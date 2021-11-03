from __future__ import print_function # Only Python 2.x
import subprocess
import sys
import os

if len(sys.argv) > 1:
    input_conditioning_json = sys.argv[1]
else:
    input_conditioning_json = None

class SamplingArguments(object):
    def __init__(self,
                 model_path,
                 output_dir,
                 tier_input_tag,
                 size_at_depth,
                 n_layers,
                 hidden_size,
                 stored_sampled_tier_data=None,
                 cuda_device=0,
                 custom_conditioning_json=None,
                 script_name="sample_attention_melnet_cmdline.py",
                 axis_split="21212",
                 experiment_name="sampled_model",
                 use_sample_index="0,0",
                 tier_condition_tag=None,
                 learning_rate="1E-4",
                 optimizer="adam",
                 real_batch_size=1,
                 virtual_batch_size=1,
                 cell_type="gru"):
        super(SamplingArguments, self).__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.cuda_device = cuda_device
        self.script_name = script_name
        self.axis_split = axis_split
        self.experiment_name = experiment_name
        self.use_sample_index = use_sample_index
        self.tier_condition_tag = tier_condition_tag
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.real_batch_size = real_batch_size
        self.virtual_batch_size = virtual_batch_size
        self.tier_input_tag = tier_input_tag
        self.size_at_depth = size_at_depth
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.stored_sampled_tier_data = stored_sampled_tier_data
        self.custom_conditioning_json = custom_conditioning_json

    def format_args_as_string(self):
        base_str = "CUDA_VISIBLE_DEVICES={} python sample_attention_melnet_cmdline.py".format(self.cuda_device)
        base_str += " --axis_split=" + str(self.axis_split)
        base_str += " --tier_input_tag=" + str(self.tier_input_tag)
        if self.tier_condition_tag is not None:
            base_str += " --tier_condition_tag=" + str(self.tier_condition_tag)
        base_str += " --size_at_depth=" + str(self.size_at_depth)
        base_str += " --n_layers=" + str(self.n_layers)
        base_str += " --hidden_size=" + str(self.hidden_size)
        base_str += " --cell_type=" + str(self.cell_type)
        base_str += " --learning_rate=" + str(self.learning_rate)
        base_str += " --optimizer=" + str(self.optimizer)
        base_str += " --real_batch_size=" + str(self.real_batch_size)
        base_str += " --virtual_batch_size=" + str(self.virtual_batch_size)
        base_str += " --use_sample_index=" + str(self.use_sample_index)
        base_str += " --output_dir=" + str(self.output_dir)
        base_str += " --experiment_name=" + str(self.experiment_name)
        if self.custom_conditioning_json is not None:
            base_str += " --custom_conditioning_json=" + str(self.custom_conditioning_json)
        base_str += " {}".format(self.model_path)
        if self.stored_sampled_tier_data is not None:
            base_str += " --stored_sampled_tier_data=" + str(self.stored_sampled_tier_data)
        return base_str

"""
new 0_0
CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py --axis_split=21212 --tier_input_tag=0,0 --size_at_depth=88,32 --n_layers=5 --hidden_size=256 --cell_type=gru --learning_rate=1E-4 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34 --output_dir=chain_sampled_tier0_0/ /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_23-46-12_2021-05-10_02bd5d_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_0_0_sz_88_32/saved_models/valid_model-65364.pth
"""

"""
new 0_1
CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py  --axis_split=21212 --tier_input_tag=0,1 --tier_condition_tag=0,0 --size_at_depth=88,32 --n_layers=5 --hidden_size=256 --cell_type=gru --learning_rate=1E-4 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34 --output_dir=chain_sampled_tier0_1/ /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_08-58-06_2021-06-10_ea62e3_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_0_1_cond_0_0_sz_88_32/saved_models/valid_model-56368.pth --stored_sampled_tier_data=chain_sampled_tier0_0/sampled_forced_images/unnormalized_samples.npy
"""

"""
new 1_1
CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py  --axis_split=21212 --tier_input_tag=1,1 --tier_condition_tag=1,0 --size_at_depth=88,64 --n_layers=4 --hidden_size=256 --cell_type=gru --learning_rate=1E-4 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34 --output_dir=chain_sampled_tier1_1/ /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_04-13-10_2021-13-10_d28005_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_1_1_cond_1_0_sz_88_64/saved_models/valid_model-45696.pth --stored_sampled_tier_data="chain_sampled_tier0_0/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier0_1/sampled_forced_images/unnormalized_samples.npy"
"""

"""
new 2_1
CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py  --axis_split=21212 --tier_input_tag=2,1 --tier_condition_tag=2,0 --size_at_depth=176,64 --n_layers=3 --hidden_size=256 --cell_type=gru --learning_rate=1E-4 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34 --output_dir=chain_sampled_tier2_1/ /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_05-36-09_2021-14-10_907c17_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_2_1_cond_2_0_sz_176_64/saved_models/valid_model-45625.pth --stored_sampled_tier_data="chain_sampled_tier0_0/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier0_1/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier1_1/sampled_forced_images/unnormalized_samples.npy"
"""

"""
new 3_1
CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py  --axis_split=21212 --tier_input_tag=3,1 --tier_condition_tag=3,0 --size_at_depth=176,128 --n_layers=2 --hidden_size=256 --cell_type=gru --learning_rate=1E-4 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34 --output_dir=chain_sampled_tier3_1/ /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_04-36-32_2021-18-10_2365e2_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_3_1_cond_3_0_sz_176_128/saved_models/valid_model-20000.pth --stored_sampled_tier_data="chain_sampled_tier0_0/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier0_1/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier1_1/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier2_1/sampled_forced_images/unnormalized_samples.npy"
"""

"""
new 4_1
CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py  --axis_split=21212 --tier_input_tag=4,1 --tier_condition_tag=4,0 --size_at_depth=352,128 --n_layers=2 --hidden_size=256 --cell_type=gru --learning_rate=1E-4 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34 --output_dir=chain_sampled_tier4_1/ /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_21-54-32_2021-19-10_3e9e97_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_4_1_cond_4_0_sz_352_128/saved_models/valid_model-11250.pth --stored_sampled_tier_data="chain_sampled_tier0_0/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier0_1/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier1_1/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier2_1/sampled_forced_images/unnormalized_samples.npy,chain_sampled_tier3_1/sampled_forced_images/unnormalized_samples.npy"
"""

"""
combine_all_outputs.py --axis_split=21212
"""

storage_dir="tier0_0"
sample_index="10,0"
tier0_0_model_path = os.path.expanduser("~") + "/_kkpthlib_models/attention_melnet_cmdline_23-46-12_2021-05-10_02bd5d_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_0_0_sz_88_32/saved_models/valid_model-65364.pth"

tier0_0_args = SamplingArguments(
                                 custom_conditioning_json=input_conditioning_json,
                                 model_path=tier0_0_model_path,
                                 output_dir=storage_dir,
                                 tier_input_tag="0,0",
                                 size_at_depth="88,32",
                                 use_sample_index=sample_index,
                                 n_layers=5,
                                 hidden_size=256)

sampled_string="{}unnormalized_samples.npy".format(storage_dir + "/sampled_forced_images/")
storage_dir="tier0_1_cond0_0"
tier0_1_cond0_0_model_path = os.path.expanduser("~") + "/_kkpthlib_models/attention_melnet_cmdline_08-58-06_2021-06-10_ea62e3_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_0_1_cond_0_0_sz_88_32/saved_models/valid_model-56368.pth"
tier0_1_cond0_0_args = SamplingArguments(
                                         custom_conditioning_json=input_conditioning_json,
                                         model_path=tier0_1_cond0_0_model_path,
                                         output_dir=storage_dir,
                                         tier_input_tag="0,1",
                                         tier_condition_tag="0,0",
                                         size_at_depth="88,32",
                                         use_sample_index=sample_index,
                                         n_layers=5,
                                         hidden_size=256,
                                         stored_sampled_tier_data=sampled_string)

sampled_string+=",{}unnormalized_samples.npy".format(storage_dir + "/sampled_forced_images/")
storage_dir="tier1_1_cond1_0"
tier1_1_cond1_0_model_path = os.path.expanduser("~") + "/_kkpthlib_models/attention_melnet_cmdline_04-13-10_2021-13-10_d28005_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_1_1_cond_1_0_sz_88_64/saved_models/valid_model-45696.pth"
tier1_1_cond1_0_args = SamplingArguments(
                                         custom_conditioning_json=input_conditioning_json,
                                         model_path=tier1_1_cond1_0_model_path,
                                         output_dir=storage_dir,
                                         tier_input_tag="1,1",
                                         tier_condition_tag="1,0",
                                         size_at_depth="88,64",
                                         use_sample_index=sample_index,
                                         n_layers=4,
                                         hidden_size=256,
                                         stored_sampled_tier_data=sampled_string)

sampled_string+=",{}unnormalized_samples.npy".format(storage_dir + "/sampled_forced_images/")
storage_dir="tier2_1_cond2_0"
tier2_1_cond2_0_model_path = os.path.expanduser("~") + "/_kkpthlib_models/attention_melnet_cmdline_05-36-09_2021-14-10_907c17_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_2_1_cond_2_0_sz_176_64/saved_models/valid_model-45625.pth"
tier2_1_cond2_0_args = SamplingArguments(
                                         custom_conditioning_json=input_conditioning_json,
                                         model_path=tier2_1_cond2_0_model_path,
                                         output_dir=storage_dir,
                                         tier_input_tag="2,1",
                                         tier_condition_tag="2,0",
                                         size_at_depth="176,64",
                                         use_sample_index=sample_index,
                                         n_layers=3,
                                         hidden_size=256,
                                         stored_sampled_tier_data=sampled_string)

sampled_string+=",{}unnormalized_samples.npy".format(storage_dir + "/sampled_forced_images/")
storage_dir="tier3_1_cond3_0"
tier3_1_cond3_0_model_path = os.path.expanduser("~") + "/_kkpthlib_models/attention_melnet_cmdline_04-36-32_2021-18-10_2365e2_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_3_1_cond_3_0_sz_176_128/saved_models/valid_model-20000.pth"
tier3_1_cond3_0_args = SamplingArguments(
                                         custom_conditioning_json=input_conditioning_json,
                                         model_path=tier3_1_cond3_0_model_path,
                                         output_dir=storage_dir,
                                         tier_input_tag="3,1",
                                         tier_condition_tag="3,0",
                                         size_at_depth="176,128",
                                         use_sample_index=sample_index,
                                         n_layers=2,
                                         hidden_size=256,
                                         stored_sampled_tier_data=sampled_string)

sampled_string+=",{}unnormalized_samples.npy".format(storage_dir + "/sampled_forced_images/")
storage_dir="tier4_1_cond4_0"
tier4_1_cond4_0_model_path = os.path.expanduser("~") + "/_kkpthlib_models/attention_melnet_cmdline_21-54-32_2021-19-10_3e9e97_attn_tts_robovoice_mae_preproc_fp16_adam_gru_256_ramplr_round34_tier_4_1_cond_4_0_sz_352_128/saved_models/valid_model-11250.pth"
tier4_1_cond4_0_args = SamplingArguments(
                                         custom_conditioning_json=input_conditioning_json,
                                         model_path=tier4_1_cond4_0_model_path,
                                         output_dir=storage_dir,
                                         tier_input_tag="4,1",
                                         tier_condition_tag="4,0",
                                         size_at_depth="352,128",
                                         use_sample_index=sample_index,
                                         n_layers=2,
                                         hidden_size=256,
                                         stored_sampled_tier_data=sampled_string)


# https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

scripts = [tier0_0_args, tier0_1_cond0_0_args, tier1_1_cond1_0_args, tier2_1_cond2_0_args, tier3_1_cond3_0_args, tier4_1_cond4_0_args]
for _i, cmd_args in enumerate(scripts):
    #print(cmd_args.format_args_as_string())
    for el in execute(cmd_args.format_args_as_string()):
        print(el, end="")
