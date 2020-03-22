from torch import nn
import torch
import numpy as np
from kkpthlib.datasets import fetch_norvig_words
from kkpthlib.datasets import ListIterator
from kkpthlib import TransformerPositionalEncoding
from kkpthlib import TransformerEncoderBlock
from kkpthlib import TransformerDecoderBlock
from kkpthlib import CategoricalCrossEntropy
from kkpthlib import Embedding
from kkpthlib import Linear
from kkpthlib import HParams
from kkpthlib import softmax
from kkpthlib import clipping_grad_norm_
from kkpthlib import ListIterator
from kkpthlib import run_loop
from kkpthlib import HParams
from kkpthlib import NoamOpt
from kkpthlib import get_logger

#vocab = "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
vocab = "_0123456789abcdefghijklmnopqrstuvwxyz"
hp = HParams(word_length_limit=10,
             train_seed=1122,
             valid_seed=12,
             random_seed=1999,
             batch_size=64,
             clip=10.,
             n_syms=len(vocab),
             dim=512,
             n_layers=3,
             split=250000,
             dropout_keep_prob=.9,
             use_device='cuda' if torch.cuda.is_available() else 'cpu')

def get_hparams():
    return hp

norvig = fetch_norvig_words()
words = norvig["data"]
maxlen = max([len(words_i) for words_i in words])

word_length_limit = hp.word_length_limit

words = [words_i for words_i in words if len(words_i) <= word_length_limit]
v2i = {v: k for k, v in enumerate(vocab)}
i2v = {v: k for k, v in v2i.items()}
word_inds = [np.array([v2i[wi] for wi in word_i] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]
rev_word_inds = [np.array([v2i[wi] for wi in word_i][::-1] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]

train_itr_random_state = np.random.RandomState(hp.train_seed)
valid_itr_random_state = np.random.RandomState(hp.valid_seed)
random_state = np.random.RandomState(hp.random_seed)

batch_size = hp.batch_size
n_syms = hp.n_syms

shuffled_inds = list(range(len(word_inds)))
train_itr_random_state.shuffle(shuffled_inds)
split = hp.split
train_inds = shuffled_inds[:split]
valid_inds = shuffled_inds[split:]
train_word_inds = [word_inds[i] for i in train_inds]
train_rev_word_inds = [rev_word_inds[i] for i in train_inds]

valid_word_inds = [word_inds[i] for i in valid_inds]
valid_rev_word_inds = [rev_word_inds[i] for i in valid_inds]

train_itr = ListIterator([train_word_inds, train_rev_word_inds], batch_size, random_state=train_itr_random_state)
valid_itr = ListIterator([valid_word_inds, valid_rev_word_inds], batch_size, random_state=valid_itr_random_state)

def build_model(hp):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.input_positional_encoding = TransformerPositionalEncoding(hp.dim, name="inp_pos_emb", device=hp.use_device)
            self.input_token_embedding = Embedding(len(vocab), hp.dim,
                                             random_state=random_state,
                                             name="inp_embed",
                                             device=hp.use_device)
            self.target_positional_encoding = TransformerPositionalEncoding(hp.dim, name="target_pos_emb", device=hp.use_device)
            self.target_token_embedding = Embedding(len(vocab), hp.dim,
                                             random_state=random_state,
                                             name="target_embed",
                                             device=hp.use_device)
            self.encode = TransformerEncoderBlock([hp.dim], n_layers=hp.n_layers, dropout_keep_prob=hp.dropout_keep_prob, random_state=random_state, name="transformer_encoder", device=hp.use_device)
            self.decode = TransformerDecoderBlock([hp.dim], n_layers=hp.n_layers, dropout_keep_prob=hp.dropout_keep_prob, random_state=random_state, name="transformer_decoder", device=hp.use_device)
            self.linear = Linear([hp.dim],
                                  len(vocab),
                                  random_state=random_state,
                                  name="output_logits")

        def forward(self, input_, target, input_mask, target_mask):
            input_embed, e_v = self.input_token_embedding(input_)
            input_pe, pe_v = self.input_positional_encoding(input_embed)

            target_embed, t_v = self.target_token_embedding(target)
            target_pe, pt_v = self.target_positional_encoding(target_embed)

            input_pe = input_mask[..., None] * input_pe
            target_pe = target_mask[..., None] * target_pe

            encoded = self.encode([input_pe], input_mask)
            decoded = self.decode([target_pe], [encoded], target_mask, input_mask)
            return self.linear([decoded])
    model = Model().to(hp.use_device)
    return model

if __name__ == "__main__":
    model = build_model(hp)
    loss_fun = CategoricalCrossEntropy()

    def get_std_noam_opt(model):
        return NoamOpt(hp.dim, 1, 4000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1E-9))
    optimizer = get_std_noam_opt(model)

    def loop(itr, extras, stateful_args):
        optimizer.zero_grad()
        if extras["train"]:
            model.train()
        else:
            model.eval()

        x, y = next(itr)

        x = x.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)
        new_y = np.zeros((y.shape[0] + 1, y.shape[1], y.shape[2]))
        new_y[1:] = y

        x_mask = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
        x_mask[x > 0] = 1.
        x_mask = x_mask[..., 0]

        y_mask = np.zeros((y.shape[0] + 1, y.shape[1], y.shape[2]))
        y_mask[new_y > 0] = 1.
        y_mask[0] = 1.
        y_mask = y_mask[..., 0]
        y = new_y

        t_x = torch.Tensor(x).to(hp.use_device).detach()
        mask_t_x = torch.Tensor(x_mask).to(hp.use_device).detach()
        # t_y is 0 padded at the front for AR prediction
        t_y = torch.Tensor(y).to(hp.use_device).detach()
        mask_t_y = torch.Tensor(y_mask).to(hp.use_device).detach()

        pred_logit = model(t_x, t_y[:-1], mask_t_x, mask_t_y[:-1])
        pred_prob = softmax(pred_logit)
        loss_batch = loss_fun(pred_prob, t_y[1:])
        #loss = (mask_t_y[1:] * loss_batch).sum(dim=0).mean()
        loss = loss_batch.sum(dim=0).mean()
        l = loss.cpu().data.numpy()
        if extras["train"]:
            loss.backward()
            #clipping_grad_norm_(model.parameters(), hp.clip)
            optimizer.step()
        return l, None

    s = {"model": model,
         "optimizer": optimizer,
         "hparams": hp}

    run_loop(loop, train_itr,
             loop, valid_itr,
             s,
             n_train_steps_per=1000,
             n_valid_steps_per=50)
