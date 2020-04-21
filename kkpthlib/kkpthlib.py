import numpy as np
import torch
from scipy import linalg
from scipy.stats import truncnorm
import math

import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

import copy

from .hparams import HParams

def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros
    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize
    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype("float32")


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01
    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype("float32")


def np_truncated_normal(shape, random_state, scale=0.075):
    """
    Builds a numpy variable filled with truncated normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.075)
        default of 0.075
    Returns
    -------
    initialized_normal, array-like
        Array-like of truncated normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape

    sigma = scale
    lower = -2 * sigma
    upper = 2 * sigma
    mu = 0
    N = np.prod(shp)
    samples = truncnorm.rvs(
              (lower - mu) / float(sigma), (upper - mu) / float(sigma),
              loc=mu, scale=sigma, size=N, random_state=random_state)
    return samples.reshape(shp).astype("float32")


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale
    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype("float32")


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / float(kern_sum))  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_glorot_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1. * sqrt(6 / (n_in + n_out)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        from IPython import embed; embed(); raise ValueError()
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prod(shp[1:]))
    else:
        shp = shape
        kern_sum = sum(shp)
    bound = scale * np.sqrt(6. / float(kern_sum))
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.
    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prod(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims, name=""):
    logger.info("Initializing {} with {} init".format(name, "zero"))
    #return [np.random.randn(dim,).astype("float32") for dim in bias_dims]
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default", name=""):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)

    linear example:
            weight_values, = make_numpy_weights(input_dim, [output_dim],
                                                random_state=random_state,
                                                init=init, scale=scale, name=name_w)

    conv example:
            shape usually constructed internally as:
            shp = (shape[1][0], shape[0][0]) + shape[1][1:]

            weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                                [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                                init=init,
                                                scale=scale,
                                                random_state=random_state, name=name_w)

            this means input_width, input_height are ignored for most initializers
    """
    ff = [None] * len(out_dims)
    fs = [scale] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            ff[i] = np_ortho
            fs[i] = 1.
            '''
            if in_dim == out_dim:
                logger.info("Initializing {} with {} init".format(name, "ortho"))
                ff[i] = np_ortho
                fs[i] = 1.
            else:
                logger.info("Initializing {} with {} init".format(name, "variance_scaled_uniform"))
                ff[i] = np_variance_scaled_uniform
                fs[i] = 1.
            '''
        elif init == "ortho":
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            if in_dim != out_dim:
                raise ValueError("Unable to use ortho init for non-square matrices!")
            ff[i] = np_ortho
            fs[i] = 1.
        elif init == "glorot_uniform":
            logger.info("Initializing {} with {} init".format(name, "glorot_uniform"))
            ff[i] = np_glorot_uniform
        elif init == "normal":
            logger.info("Initializing {} with {} init".format(name, "normal"))
            ff[i] = np_normal
            fs[i] = 0.01
        elif init == "truncated_normal":
            logger.info("Initializing {} with {} init".format(name, "truncated_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 0.075
        elif init == "embedding_normal":
            logger.info("Initializing {} with {} init".format(name, "embedding_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 1. / np.sqrt(out_dim)
        else:
            raise ValueError("Unknown init type %s" % init)

    ws = []
    for i, out_dim in enumerate(out_dims):
        if fs[i] == "default":
            wi = ff[i]((in_dim, out_dim), random_state)
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
        else:
            wi = ff[i]((in_dim, out_dim), random_state, scale=fs[i])
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
    return ws

from scipy.stats import truncnorm
import sys
import uuid
from .core import get_logger
from collections import OrderedDict

logger = get_logger()

# Storage of internal shared
_lib_shared_params = OrderedDict()
has_warned = {}

def _shape(arr):
    return tuple(arr.shape)

def _ndim(arr):
    return len(_shape(arr))

def _get_name():
    return str(uuid.uuid4())

def get_params_dict():
    return _lib_shared_params

def _get_shared(name):
    if name in _lib_shared_params.keys():
        if name not in has_warned:
            logger.info("Found name %s in shared parameters" % name)
            has_warned[name] = True
        return _lib_shared_params[name]
    else:
        raise NameError("Name not found in shared params!")

def _check_shared(name):
    return name in _lib_shared_params.keys()

def _set_shared(name, variable):
    if name in _lib_shared_params.keys():
        raise ValueError("Trying to set key %s which already exists!" % name)
    _lib_shared_params[name] = variable

weight_norm_default = False
def get_weight_norm_default():
    return weight_norm_default

strict_mode_default = False
def get_strict_mode_default():
    return strict_mode_default


device_default = "cpu"
def get_device_default():
    return device_default

def set_device_default(device):
    global device_default
    device_default = device


dtype_default = "float32"
def get_dtype_default():
    return dtype_default

def set_dtype_default(dtype):
    global dtype_default
    dtype_default = dtype


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return torch.nn.functional.relu(x)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def softmax(x):
    # should work for both 2D and 3D
    e_x = torch.exp(x - x.max(dim=-1, keepdims=True)[0])
    out = e_x / e_x.sum(dim=-1, keepdims=True)
    return out

def softmax_np(x):
    # should work for both 2D and 3D
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)
    return out


def make_tensor(arr, dtype, device, requires_grad=True):
    if device == "default":
        device = get_device_default()
    else:
        device = device

    if dtype == "default":
        dtype = get_dtype_default()

    if dtype == "float32":
        tensor = torch.from_numpy(arr.astype("float32")).to(device)
    elif dtype == "float64":
        tensor = torch.from_numpy(arr.astype("float64")).to(device)
    else:
        raise ValueError("Not yet implemented for dtype {}".format(dtype))
    if not requires_grad:
        tensor = tensor.requires_grad_(False)
    return tensor

def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    a_tup = _shape(a)
    b_tup = _shape(b)
    if len(a_tup) == 2 and len(b_tup) == 2:
        return torch.matmul(a, b)
    elif len(a_tup) == 3 and len(b_tup) == 2:
        # more generic, supports multiple -1 axes
        return torch.einsum("ijk,kl->ijl", a, b)
        #a_i = tf.reshape(a, [-1, a_tup[-1]])
        #a_n = tf.matmul(a_i, b)
        #a_nf = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
        #return a_nf
    else:
        raise ValueError("Shapes for arguments to dot() are {} and {}, not supported!".format(a_tup, b_tup))

_scan_infos = {}
def scan(fn, sequences, outputs_info):
    nonepos = [n for n, o in enumerate(outputs_info) if o is None]
    nonnone = [o for o in outputs_info if o is not None]
    sequences_and_nonnone = sequences + nonnone
    sliced = [s[0] for s in sequences] + nonnone
    sig = (fn.func_code.co_filename,) + (fn.func_code.co_name,) + (fn.func_code.co_firstlineno,) + fn.func_code.co_varnames + (fn.func_code.co_code,)
    lu = hash(sig)
    global _scan_infos
    if lu not in _scan_infos:
        inf_ret = fn(*sliced)
        _scan_infos[lu] = inf_ret
    else:
        inf_ret = _scan_infos[lu]
    if len(outputs_info) < len(inf_ret):
        raise ValueError("More outputs from `fn` than elements in outputs_info. Expected {} outs, given outputs_info of length {}, but `fn` returns {}. Pass None in outputs_info for returns which don't accumulate".format(len(outputs_info), len(outputs_info), len(inf_ret)))
    initializers = []
    for n in range(len(outputs_info)):
        if outputs_info[n] is not None:
            initializers.append(outputs_info[n])
        else:
            initializers.append(0. * inf_ret[n])

    def wrapwrap(nonepos, initializers):
        type_class = "list" if isinstance(initializers, list) else "tuple"
        def fnwrap(accs, inps):
            inps_then_accs = inps + [a for n, a in enumerate(accs) if n not in nonepos]
            fn_rets = fn(*inps_then_accs)
            return [fr for fr in fn_rets]
        return fnwrap

    this_fn = wrapwrap(nonepos, initializers)
    def _scan(lclfn, seqs, inits):
        all_r = [[] for i in range(len(inits))]
        last_out = inits
        for i in range(len(seqs[0])):
            ri = lclfn(last_out, [seqs[n][i] for n in range(len(seqs))])
            last_out = ri
            if not hasattr(ri, "__len__"):
                ri = [ri]
            else:
                [all_r[j].append(ri[j]) for j in range(len(ri))]
        return all_r

    r = _scan(this_fn, sequences, initializers)
    return [torch.stack(rj) for rj in r]


def clipping_grad_norm_(parameters, rescale, named_parameters=False, named_check=False):
    # is a generator... get a static reference so the second iteration isn't empty
    if named_check:
        for n, p in parameters:
            print("Checking {} grad.data".format(n))
            assert p.grad.data is not None
            print("{}, OK".format(n))
        raise ValueError("named_check complete!")
    if not named_parameters:
        _params = [p for p in parameters]
    else:
        _params = [p[1] for p in parameters]

    grad_norm = torch.sqrt(sum([torch.sqrt(torch.pow(p.grad.data, 2).sum()) for p in _params]))
    scaling_num = rescale
    scaling_den = max([1.0 * rescale, grad_norm])
    scaling = scaling_num / scaling_den
    for p in _params:
        p.grad.data.mul_(scaling)


def clipping_grad_value_(parameters, clip_value, named_parameters=False, named_check=False):
    # is a generator... get a static reference so the second iteration isn't empty
    if named_check:
        for n, p in parameters:
            print("Checking {} grad.data".format(n))
            assert p.grad.data is not None
            print("{}, OK".format(n))
        raise ValueError("named_check complete!")
    if not named_parameters:
        _params = [p for p in parameters]
    else:
        _params = [p[1] for p in parameters]

    clip_value = float(clip_value)
    for p in _params:
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


class Embedding(torch.nn.Module):
    def __init__(self,
                 n_symbols,
                 output_dim,
                 random_state=None,
                 init="embedding_normal",
                 scale=1.,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        """
        Last dimension of indices tensor must be 1!!!!
        """
        super(Embedding, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state argument to Embedding")

        name_w = name + "_embedding_w"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

        if init != "embedding_normal":
            raise ValueError("Currently unsupported init type {}".format(init))

        try:
            vectors = _get_shared(name_w)
        except NameError:
            vectors_weight, = make_numpy_weights(n_symbols, [output_dim],
                                                 random_state, init=init,
                                                 scale=scale, name=name_w)
            vectors = make_tensor(vectors_weight, dtype=dtype, device=device)
            #vectors = torch.from_numpy(vectors_weight).to(lcl_device)
            _set_shared(name_w, vectors)
        self.vectors = vectors

        th_embed = torch.nn.Embedding(n_symbols, output_dim)
        th_embed.weight.data.copy_(vectors)
        self.th_embed = th_embed

    def forward(self,
                indices):
        ii = indices.long()
        shp = _shape(ii)
        nd = _ndim(ii)
        if shp[-1] != 1:
            if nd < 3:
                logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
                ii = ii[..., None]
            else:
                raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

        shp = _shape(ii)
        nd = len(shp)
        # force 3d for consistency, then slice
        lu = self.th_embed(ii[..., 0])
        return lu, self.vectors


class Linear(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 output_dim,
                 random_state=None,
                 name=None,
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 dropout_flag_prob_keep=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(Linear, self).__init__()

        if random_state is None:
            raise ValueError("must pass instance of np.random.RandomState!")
        input_dim = sum(list_of_input_dims)

        if name is None:
            name = _get_name()

        name_w = name + "_linear_w"
        name_b = name + "_linear_b"
        name_out = name + "_linear_out"

        if init is None or type(init) is str:
            #logger.info("Linear layer {} initialized using init {}".format(name, init))
            weight_values, = make_numpy_weights(input_dim, [output_dim],
                                                random_state=random_state,
                                                init=init, scale=scale, name=name_w)
        else:
            # rely on announcement from parent class
            weight_values=init[0]

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)

        self.weight = torch.nn.Parameter(weight)
        self.biases = None

        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([output_dim], name=name_b)
            else:
                b = init[1]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device)
                _set_shared(name_b, biases)
            self.biases = torch.nn.Parameter(biases)

    def forward(self,
                list_of_inputs,
                bias_offset=0.,
                dropout_flag_prob_keep=None):

        nd = _ndim(list_of_inputs[0])
        input_var = torch.cat(list_of_inputs, dim=nd - 1)
        if dropout_flag_prob_keep is not None:
            # no seed set here, it might not be repeatable
            input_var = torch.nn.functional.dropout(input_var, p=1. - dropout_flag_prob_keep, inplace=False)

        out = dot(input_var, self.weight)

        if self.biases is not None:
            out = out + self.biases
        return out


class Conv2d(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 kernel_size=(3, 3),
                 dilation=[1, 1],
                 strides=[1, 1],
                 input_height_width_init_tuple=(1, 1),
                 border_mode="same",
                 custom_weight_mask=None,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None,
                 random_state=None, strict=None,
                 dtype="default", device="default"):
        super(Conv2d, self).__init__()

        if strides != [1, 1]:
            raise ValueError("Alternate strides not yet supported in conv2d")
        if dilation != [1, 1]:
            raise ValueError("Alternate dilation not yet supported in conv2d")
        # kernel is H, W
        # input assumption is N C H W
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            raise ValueError("Changing dilation not yet supported")

        input_channels = sum(list_of_input_dims)
        #input_height, input_width = input_height_width_tuple
        # these numbers don't matter
        input_height, input_width = input_height_width_init_tuple

        if type(name) is str:
            name_w = name + "_conv2d_w"
            name_b = name + "_conv2d_b"
            name_out = name + "_conv2d_out"
            name_mask = name + "_conv2d_mask"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        if init is None or type(init) is str:
            weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                                [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                                init=init,
                                                scale=scale,
                                                random_state=random_state, name=name_w)
        else:
            weight_values = init[0]
            name_w = name[0]
        weight_values = weight_values.transpose(3, 2, 0, 1)
        #weight_values = weight_values[::-1, ::-1].copy()

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)

        self.weight = torch.nn.Parameter(weight)

        if custom_weight_mask is not None:
            """
            try:
                mask = _get_shared(name_mask)
            except NameError:
                mask = tf.Variable(custom_weight_mask, trainable=False, name=name_mask)
                _set_shared(name_mask, mask)
            """
            raise ValueError("custom_weight_mask not yet implemented in conv")
            weight = tf.constant(custom_weight_mask) * weight


        # need to custom handle SAME and VALID
        # rip
        # NCHW input, weights are out_chan, in_chan, H, W
        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([num_feature_maps], name=name_b)
            else:
                b = init[1]
                name_b = name[1]
                name_out = name[2]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device)
                _set_shared(name_b, biases)
            self.biases = torch.nn.Parameter(biases)

        self.strides = strides
        self.dilation = dilation
        self.input_channels = input_channels
        # these numbers don't matter
        self.input_height = input_height
        self.input_width = input_width
        self.border_mode = border_mode
        self.kernel_size = kernel_size

    def forward(self,
                list_of_inputs):
        dilation = self.dilation
        strides = self.strides
        input_channels = self.input_channels
        input_height = self.input_height
        input_width = self.input_width
        border_mode = self.border_mode
        weight = self.weight
        biases = self.biases
        kernel_size = self.kernel_size

        if strides != [1, 1]:
            raise ValueError("Alternate strides not yet supported in conv2d")
        if dilation != [1, 1]:
            raise ValueError("Alternate dilation not yet supported in conv2d")
        # kernel is H, W
        # input assumption is N C H W

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            raise ValueError("Changing dilation not yet supported")

        input_t = torch.cat(list_of_inputs, dim=-1)

        if border_mode == "same":
            pad = "same"
        elif border_mode == "valid":
            pad = "valid"
        else:
            pad = border_mode
            if hasattr(pad, "__len__") and len(pad) == 2:
                pass
            else:
                try:
                    int(pad)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Pad must be integer, tuple of integer (hpad, wpad), or string 'same', 'valid'")

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim(in_dim, padding, ks, stride, dilation):
            if isinstance(padding, int) or isinstance(padding, tuple):
                return conv_outdim_general(in_dim, padding, ks, stride, dilation)
            elif isinstance(padding, str):
                assert padding in ['same', 'valid']
                if padding == 'same':
                    return conv_outdim_samepad(in_dim, stride)
                else:
                    return conv_outdim_general(in_dim, 0, ks, stride, dilation)
            else:
                raise TypeError('Padding can be int/tuple or str=same/valid')

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_general(in_dim, padding, ks, stride, dilation=1):
            # See https://arxiv.org/pdf/1603.07285.pdf, eq (15)
            return ((in_dim + 2 * padding - ks - (ks - 1) * (dilation - 1)) // stride) + 1

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_samepad(in_dim, stride):
            return (in_dim + stride - 1) // stride

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def pad_same(in_dim, ks, stride, dilation=1):
            """
            References:
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
            """
            assert stride > 0
            assert dilation >= 1
            effective_ks = (ks - 1) * dilation + 1
            out_dim = (in_dim + stride - 1) // stride
            p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

            padding_before = p // 2
            padding_after = p - padding_before
            return padding_before, padding_after


        if pad == "same":
            ph = pad_same(input_t.shape[-2], kernel_size[0], strides[-2], dilation[-2])[0]
            pw = pad_same(input_t.shape[-1], kernel_size[1], strides[-1], dilation[-1])[0]
        elif pad == "valid":
            raise ValueError("valid pad NYI")
            from IPython import embed; embed(); raise ValueError()

        # NCHW input, weights are out_chan, in_chan, H, W
        out = torch.nn.functional.conv2d(input_t, weight, stride=strides, dilation=dilation, padding=(ph, pw), bias=biases)
        return out


class BatchNorm2d(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 train_test_flag,
                 gamma_init=1., beta_init=0.,
                 decay=0.9,
                 eps=1E-3,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        super(BatchNorm2d, self).__init__()
        # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        # NCHW convention
        if name is None:
            name = _get_name()

        name_scale = name + "_batchnorm_s"
        name_beta = name + "_batchnorm_b"
        name_out = name + "_batchnorm_out"
        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_scale in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_scale))

            if name_beta in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_beta))

        try:
            scale = _get_shared(name_scale)
        except NameError:
            scale_values = gamma_init * np.ones((input_dim,))
            scale = make_tensor(scale_values, dtype=dtype, device=device)
            _set_shared(name_scale, scale)

        try:
            beta = _get_shared(name_beta)
        except NameError:
            # init with ones? it's what I did in TF
            beta_values = beta_init * np.ones((input_dim,))
            beta = make_tensor(beta_values, dtype=dtype, device=device)
            _set_shared(name_beta, beta)

        self.beta = torch.nn.Parameter(beta)
        self.scale = torch.nn.Parameter(scale)
        self.decay = decay
        self.eps = eps
        self.dtype = dtype
        self.device = device

    def forward(self, input_tensor, train_test_flag):
        # 0 train, 1 test
        # https://stackoverflow.com/questions/44887446/pytorch-nn-functional-batch-norm-for-2d-input
        scale = self.scale
        beta = self.beta
        eps = self.eps
        decay = self.decay
        dtype = self.dtype
        device = self.device

        pop_mean = make_tensor(np.zeros((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)
        pop_var = make_tensor(np.ones((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)

        shp = _shape(input_tensor)
        def left():
            return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, weight=scale, bias=beta, momentum=1. - decay, eps=eps, training=True)

        def right():
            return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, training=False, weight=scale, bias=beta, eps=eps)

        if train_test_flag <= 0.5:
            out = left()
        else:
            out = right()
        return out


class SequenceConv1dStack(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 batch_norm_flag=None,
                 n_stacks=1,
                 residual=True,
                 activation="relu",
                 kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                 border_mode="same",
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(SequenceConv1dStack, self).__init__()

        if name is None:
            name = _get_name()

        # assuming they come in as length, batch, features
        #tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        c = Conv2d(list_of_input_dims, len(kernel_sizes) * num_feature_maps,
                   kernel_size=(1, 1),
                   name=name + "_convpre", random_state=random_state,
                   border_mode=border_mode, init=init, scale=scale, biases=biases,
                   bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
        layers = torch.nn.ModuleList()
        layers.append(torch.nn.ModuleList([c,]))

        for ii in range(n_stacks):
            cs = torch.nn.ModuleList()
            for jj, ks in enumerate(kernel_sizes):
                c = Conv2d([len(kernel_sizes) * num_feature_maps], num_feature_maps,
                           kernel_size=ks,
                           name=name + "_conv{}_ks{}".format(ii, jj), random_state=random_state,
                           border_mode=border_mode, init=init, scale=scale, biases=biases,
                           bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
                cs.append(c)
            # cat along channel axis
            bn_l = BatchNorm2d(len(cs) * num_feature_maps, batch_norm_flag, name="bn_conv{}".format(ii), dtype=dtype, device=device)
            cs.append(bn_l)
            # ????
            #r_l = ReLU(bn_l)
            layers.append(cs)
            #prev_layer = prev_layer + r_l

        post = Conv2d([len(kernel_sizes) * num_feature_maps], num_feature_maps,
                       kernel_size=(1, 1),
                       name=name + "_convpost", random_state=random_state,
                       border_mode=border_mode, init=init, scale=scale, biases=biases,
                       bias_offset=bias_offset, strict=strict,
                       dtype=dtype, device=device)

        li = torch.nn.ModuleList()
        li.append(post)
        layers.append(li)
        self.layers = layers
        self.n_stacks = n_stacks
        self.kernel_sizes = kernel_sizes

    def forward(self,
                list_of_inputs,
                batch_norm_flag=None):

        # assuming they come in as length, batch, features
        tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        pre = self.layers[0][0](tlist)
        n_stacks = self.n_stacks
        kernel_sizes = self.kernel_sizes
        prev_layer = pre
        for ii in range(n_stacks):
            cs = []
            for jj, ks in enumerate(kernel_sizes):
                # off by one to account for pre
                c = self.layers[ii + 1][jj]([prev_layer])
                cs.append(c)
            c_layer = torch.cat(cs, dim=1)
            # cat along channel axis
            # off by one to account for bn layer last
            bn_l = self.layers[ii + 1][jj + 1](c_layer, batch_norm_flag)
            r_l = ReLU(bn_l)
            prev_layer = prev_layer + r_l
        post = self.layers[ii + 2][0]([prev_layer])
        return post[:, :, 0].permute(2, 0, 1)


class LSTMCell(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 input_mask=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None):
        super(LSTMCell, self).__init__()
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call
        if random_state is None:
            raise ValueError("Must pass random_state")

        if name is None:
            name = _get_name()

        input_dim = sum(list_of_input_dims)
        hidden_dim = 4 * num_units

        if init is None:
            inp_init = None
            h_init = None
            out_init = None
        elif init == "truncated_normal":
            inp_init = "truncated_normal"
            h_init = "truncated_normal"
            out_init = "truncated_normal"
        elif init == "glorot_uniform":
            inp_init = "glorot_uniform"
            h_init = "glorot_uniform"
            out_init = "glorot_uniform"
        elif init == "normal":
            inp_init = "normal"
            h_init = "normal"
            out_init = "normal"
        else:
            raise ValueError("Unknown init argument {}".format(init))

        name_proj = name + "_lstm_proj"
        name_w = name + "_lstm_proj_w"
        name_b = name + "_lstm_proj_b"
        comb_w_np, = make_numpy_weights(input_dim + num_units, [hidden_dim],
                                        random_state=random_state,
                                        init=inp_init, name=name_w)
        comb_b_np, = make_numpy_biases([hidden_dim], name=name_b)

        logger.info("LSTMCell {} input to hidden initialized using init {}".format(name, inp_init))
        logger.info("LSTMCell {} hidden to hidden initialized using init {}".format(name, h_init))

        lstm_proj_obj = Linear(list_of_input_dims + [hidden_dim],
                               hidden_dim,
                               random_state=random_state,
                               name=name_proj,
                               init=(comb_w_np, comb_b_np), strict=strict)

        if output_dim is not None:
            name_out = name + "_lstm_h_to_out",
            name_out_w = name + "_lstm_h_to_out_w",
            name_out_b = name + "_lstm_h_to_out_b",
            h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                                random_state=random_state,
                                                init=out_init, name=name_out_w)
            h_to_out_b_np, = make_numpy_biases([output_dim], name=name_out_b)
            h_to_out_obj = Linear([num_units], output_dim, random_state=random_state,
                              name=name_out,
                              init=(h_to_out_w_np, h_to_out_b_np), strict=strict)
            self.h_to_out_obj = h_to_out_obj
        self.lstm_proj_obj = lstm_proj_obj
        self.num_units = num_units
        self.input_dim = input_dim
        self.forget_bias = forget_bias

    def forward(self,
                list_of_inputs,
                previous_hidden, previous_cell,
                output_dim=None,
                input_mask=None,
                cell_dropout=None):
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call

        input_dim = self.input_dim
        num_units = self.num_units
        forget_bias = self.forget_bias

        ph = previous_hidden
        pc = previous_cell

        lstm_proj = self.lstm_proj_obj(list_of_inputs + [ph])

        i, j, f, o = torch.split(lstm_proj, num_units, dim=-1)


        if cell_dropout is not None:
            pj = torch.nn.functional.dropout(tanh(j), 1. - cell_dropout)
        else:
            pj = tanh(j)

        c = sigmoid(f + forget_bias) * pc + sigmoid(i) * pj
        if input_mask is not None:
            c = input_mask[:, None] * c + (1. - input_mask[:, None]) * pc

        h = sigmoid(o) * tanh(c)
        if input_mask is not None:
            # this line was bugged in released / trained version!
            # https://github.com/kastnerkyle/representation_mixing/blob/master/code/lib/tfbldr/nodes/nodes.py#L1554
            # fixed here but will mean things are different
            # when masks are used
            h = input_mask[:, None] * h + (1. - input_mask[:, None]) * ph

        if output_dim is not None:
            h_to_out = self.h_to_out_obj([h])
            final_out = h_to_out
        else:
            final_out = h
        return final_out, (h, c)


class BiLSTMLayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None):
        super(BiLSTMLayer, self).__init__()
        if name is None:
            name = _get_name()
        name = name + "_bidirlstm_layer"
        name_proj = name + "_proj"
        hidden_dim = 4 * num_units
        in_proj_obj = Linear(list_of_input_dims,
                             hidden_dim,
                             random_state=random_state,
                             name=name_proj,
                             init=init, strict=strict)

        fwd_cell_obj = LSTMCell([hidden_dim],
                                num_units,
                                random_state=random_state,
                                name=name + "forward_rnn",
                                init=init)

        rev_cell_obj = LSTMCell([hidden_dim],
                                 num_units,
                                 random_state=random_state,
                                 name=name + "reverse_rnn",
                                 init=init)

        self.in_proj_obj = in_proj_obj
        self.fwd_cell_obj = fwd_cell_obj
        self.rev_cell_obj = rev_cell_obj
        self.num_units = num_units

    def forward(self, list_of_inputs,
                previous_forward_hidden=None, previous_forward_cell=None,
                previous_reverse_hidden=None, previous_reverse_cell=None,
                input_mask=None,
                cell_dropout=None,
                strict=None):

        num_units = self.num_units
        if input_mask is None:
            raise ValueError("No input mask currently unsupported")

        in_proj = self.in_proj_obj(list_of_inputs)

        if previous_forward_hidden == None:
            h1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_f_init = previous_forward_hidden
        if previous_reverse_hidden == None:
            h1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_b_init = previous_reverse_hidden
        if previous_forward_cell == None:
            c1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_f_init = previous_forward_cell
        if previous_reverse_cell == None:
            c1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_b_init = previous_reverse_cell

        def step(inp_t, inp_mask_t,
                 rev_inp_t, rev_inp_mask_t,
                 h1_f_tm1, c1_f_tm1, h1_b_tm1, c1_b_tm1):
            output, s = self.fwd_cell_obj([inp_t],
                                          h1_f_tm1, c1_f_tm1,
                                          input_mask=inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_f_t = s[0]
            c1_f_t = s[1]

            output, s = self.rev_cell_obj([rev_inp_t],
                                          h1_b_tm1, c1_b_tm1,
                                          input_mask=rev_inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_b_t = s[0]
            c1_b_t = s[1]
            return h1_f_t, c1_f_t, h1_b_t, c1_b_t

        # should this be a "proper" flip with mask on the end
        r = scan(step,
                 [in_proj, input_mask, torch.flip(in_proj, (0,)), torch.flip(input_mask, (0,))],
                 [h1_f_init, c1_f_init, h1_b_init, c1_b_init])
        return torch.cat([r[0], torch.flip(r[2], (0,))], dim=-1)


class GaussianAttentionCell(torch.nn.Module):
    def __init__(self, list_of_step_input_dims,
                 full_conditioning_tensor_dim,
                 num_units,
                 att_dim=10,
                 attention_scale=1.,
                 step_op="exp",
                 cell_type="lstm",
                 name=None,
                 random_state=None,
                 strict=None, init=None):
        super(GaussianAttentionCell, self).__init__()
        #returns w_t, k_t, phi_t, state
        # where state is the state tuple returned by the inner cell_type

        if name is None:
            name = _get_name()

        name = name + "_gaussian_attention"

        #check = any([len(_shape(si)) != 2 for si in list_of_step_inputs])
        #if check:
        #    raise ValueError("Unable to support step_input with n_dims != 2")

        if init is None or init == "truncated_normal":
            rnn_init = "truncated_normal"
            forward_init = "truncated_normal"
        else:
            raise ValueError("init != None not supported")

        random_state = np.random.RandomState(1442)
        if cell_type == "gru":
            raise ValueError("NYI")
        elif cell_type == "lstm":
            self.attn_rnn_cell = LSTMCell(list_of_step_input_dims + [full_conditioning_tensor_dim],
                                          num_units,
                                          random_state=random_state,
                                          name=name + "_gauss_att_lstm",
                                          init=rnn_init)
        else:
            raise ValueError("Unsupported cell_type %s" % cell_type)

        random_state = np.random.RandomState(1442)
        self.ret_obj = Linear(
            list_of_input_dims=[num_units],
            output_dim=3 * att_dim, name=name + "_group",
            random_state=random_state,
            strict=strict, init=forward_init)
        self.att_dim = att_dim
        self.full_conditioning_tensor_dim = full_conditioning_tensor_dim
        self.step_op = step_op
        self.attention_scale = attention_scale

    def forward(self,
                list_of_step_inputs,
                previous_state_list,
                previous_attention_position,
                full_conditioning_tensor,
                previous_attention_weight,
                input_mask=None,
                conditioning_mask=None,
                cell_dropout=None):

        att_dim = self.att_dim
        full_conditioning_tensor_dim = self.full_conditioning_tensor_dim
        step_op = self.step_op
        attention_scale = self.attention_scale

        attn_rnn_out, state = self.attn_rnn_cell(list_of_step_inputs + [previous_attention_weight],
                                                 previous_state_list[0],
                                                 previous_state_list[1],
                                                 input_mask=input_mask,
                                                 cell_dropout=cell_dropout)

        ret = self.ret_obj([attn_rnn_out])
        a_t = ret[:, :att_dim]
        b_t = ret[:, att_dim:2 * att_dim]
        k_t = ret[:, 2 * att_dim:]

        k_tm1 = previous_attention_position
        cond_dim = full_conditioning_tensor_dim
        ctx = full_conditioning_tensor
        ctx_mask = conditioning_mask

        """
        ctx = Linear(
            list_of_inputs=[full_conditioning_tensor],
            list_of_input_dims=[full_conditioning_tensor_dim],
            output_dim=next_proj_dim, name=name + "_proj_ctx",
            weight_norm=weight_norm,
            random_state=random_state,
            strict=strict, init=ctx_forward_init)
        """
        if step_op == "exp":
            a_t = torch.exp(a_t)
            b_t = torch.exp(b_t)
            step_size = attention_scale * torch.exp(k_t)
            k_t = k_tm1 + step_size
        elif step_op == "softplus":
            a_t = torch.exp(a_t)
            b_t = torch.exp(b_t)
            step_size = attention_scale * torch.nn.functional.softplus(k_t)
            k_t = k_tm1 + step_size
        elif step_op == "relu":
            a_t = torch.exp(a_t)
            b_t = torch.exp(b_t)
            step_size = attention_scale * relu(k_t)
            k_t = k_tm1 + step_size
        else:
            raise ValueError("{} not a known step_op".format(step_op))
        u = torch.arange(0, full_conditioning_tensor.shape[0], dtype=torch.float32)
        u = u[None, None]

        def calc_phi(lk_t, la_t, lb_t, lu):
            phi = torch.exp(-torch.pow(lk_t[..., None] - lu, 2) * lb_t[..., None]) * la_t[..., None]
            phi = torch.sum(phi, dim=1)[:, None]
            return phi

        phi_t = calc_phi(k_t, a_t, b_t, u)
        if conditioning_mask is not None:
            w_t_pre = phi_t * ctx.permute(1, 2, 0)
            w_t_masked = w_t_pre * ctx_mask.permute(1, 0)[:, None]
            w_t = torch.sum(w_t_masked, dim=-1)[:, None]
        else:
            raise ValueError("Non-masked conditional context NYI")
            w_t = tf.matmul(phi_t, tf.transpose(ctx, (1, 0, 2)))
        phi_t = phi_t[:, 0]
        w_t = w_t[:, 0]
        return w_t, k_t, phi_t, state


class CategoricalCrossEntropyFromSoftmax(torch.nn.Module):
    """
    Multinomial negative log likelihood of softmax predicted compared to one hot
    true_values

    Arguments to forward 
    prediction : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer

    targets : tensor, shape 2D or 3D
        One hot ground truth values. Must be the same shape as
        predicted_values. One hot representations can be achieved using
        dagbldr.utils.convert_to_one_hot
    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    """
    def __init__(self):
        super(CategoricalCrossEntropyFromSoftmax, self).__init__()

    def forward(self, prediction, target, eps=0.):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [2, 3]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [2, 3]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D inputs, got target size {}".format(target.size()))

        shp = prediction.size()
        if len(shp) == 3:
            # seq_length, batch, 1 -> seq_length * batch
            target_t = target.permute(2, 1, 0).reshape((shp[0] * shp[1],))
            # seq_length, batch, classes -> seq_length * batch, classes
            prediction_t = prediction.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c = torch.gather(prediction_t, 1, target_t.long()[..., None])
            per_step_batch_gathered = -torch.log(prediction_c.reshape((shp[1], shp[0])).transpose(1, 0))
            return per_step_batch_gathered
        else:
            raise ValueError("NYI CategoricalCrossEntropy 2D inputs!")

def logsumexp(x, dim=0):
    # http://www.cs.toronto.edu/~rfm/code/logreg.py
    _max, _ = x.max(dim=-1, keepdims=True)
    ds = x - _max
    sum_exp = torch.exp(ds).sum(dim=dim, keepdims=True)
    return _max + torch.log(sum_exp)

class CategoricalCrossEntropyFromLogits(torch.nn.Module):
    """
    Multinomial negative log likelihood of logits compared to one hot
    true_values

    Arguments to forward 
    prediction : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer

    targets : tensor, shape 2D or 3D
        One hot ground truth values. Must be the same shape as
        predicted_values. One hot representations can be achieved using
        dagbldr.utils.convert_to_one_hot
    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    """
    def __init__(self):
        super(CategoricalCrossEntropyFromLogits, self).__init__()

    def forward(self, prediction, target, threshold=1.):
        if target.size(-1) != 1:
            raise ValueError("Last dimension of target must be 1")

        if len(prediction.size()) != len(target.size()):
            raise ValueError("prediction and target must have the same number of dimensions! Got dimensions {} and {}".format(prediction.shape, target.shape))
        if len(prediction.size()) not in [2, 3]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D inputs, got prediction size {}".format(prediction.size()))

        if len(target.size()) not in [2, 3]:
            raise ValueError("CategoricalCrossEntropy only supports 2D or 3D inputs, got target size {}".format(target.size()))

        shp = prediction.size()
        if len(shp) == 3:
            # seq_length, batch, 1 -> seq_length * batch
            target_t = target.permute(2, 1, 0).reshape((shp[0] * shp[1],))
            # seq_length, batch, classes -> seq_length * batch, classes
            bot = logsumexp(prediction, dim=-1)
            if bot.shape[-1] != 1:
                raise ValueError("logsumexp calculation returned non singleton last axis! prediction {}, bot {}".format(prediction.shape, bot.shape))
            # calculate logsoftmax internally
            log_softmax_vals = prediction - bot
            prediction_t = log_softmax_vals.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c = torch.gather(prediction_t, 1, target_t.long()[..., None])
            p = prediction_c.reshape((shp[1], shp[0])).transpose(1, 0)
            # https://medium.com/@zhang_yang/understanding-cross-entropy-implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
            # https://github.com/AliAbbasi/Numerically-Stable-Cross-Entropy-Loss-Function-Tensorflow/blob/master/Numerically-Stable-Cross-Entropy-MultiLabel.py
            large_approx = -p

            softmax_vals = softmax(prediction)
            prediction_t2 = softmax_vals.permute(2, 1, 0).reshape((shp[2], shp[1] * shp[0],)).transpose(1, 0)
            prediction_c2 = torch.gather(prediction_t2, 1, target_t.long()[..., None])
            p2 = prediction_c2.reshape((shp[1], shp[0])).transpose(1, 0)
            small_approx = -torch.log(p2)

            buff = 0. * prediction[..., 0]
            buff[large_approx > threshold] = large_approx[large_approx > threshold]
            buff[large_approx <= threshold] = small_approx[large_approx <= threshold]
            return buff
        else:
            raise ValueError("NYI CategoricalCrossEntropy 2D inputs!")


class NoamOpt(object):
    """
    def get_std_opt(model):
        return NoamOpt(192, 1, 4000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


class RampOpt(object):
    """
    Similar to NoamOpt but with specified "max" learning rate rather than derived from model size

    Factor specifies whether ramp linearly or to the X power
    Warmup describest the number of steps to ramp up learning rate

    Decay to 0 by default, using cosine decay
        terminates at decay_to_zero_at_steps

    RampOpt(target_learning_rate, ramp_power, steps, opt)
    return RampOpt(.0001, 1, 4000, 4000 * 100,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    can set decay_to_zero_at_steps to -1 to disable decay
    """
    def __init__(self, target_learning_rate, factor, warmup, decay_to_zero_at_steps, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.target_learning_rate = target_learning_rate
        self.decay_to_zero_at_steps = decay_to_zero_at_steps
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step <= self.warmup:
            return self.target_learning_rate * ((step / float(self.warmup)) ** self.factor)

        if self.decay_to_zero_at_steps == -1:
            return self.target_learning_rate

        new_rate = self.target_learning_rate * np.cos((float(step - self.warmup) / (self.decay_to_zero_at_steps - self.warmup)) * (np.pi / 2.))

        if step > self.decay_to_zero_at_steps:
            logger.info("WARNING: RampOpt optimizer has decayed to LR 0! Current step {}, so no more learning happening!".format(step))
            new_rate = 0.
        # warmup is 0 on cos curve
        # infinity is pi/2?
        return new_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


class LockedDropout(nn.Module):
    def __init__(self, dropout_keep_prob=1.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super().__init__()
        self.dropout = 1. - droput_keep_prob
        if random_state is None:
            raise ValueError("Must pass random_state to LockedDropout")
        if device == "default":
            device = get_device_default()
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint())

    def forward(self, x):
        if not self.training or self.dropout == 0.:
            return x
        m = x.data.new(1, *x.size()[1:]).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class PositionalEmbedding(nn.Module):
    # credit to transformer xl
    def __init__(self, embedding_dimension,
                 name=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(PositionalEmbedding, self).__init__()

        self.embedding_dimension = embedding_dimension
        d = embedding_dimension

        inv_freq = 1. / (10000. ** (torch.arange(0.0, d, 2.0) / float(d)))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, position_sequence, batch_size=None):
        sinusoid_inp = torch.ger(position_sequence, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]


class LayerNorm(torch.nn.Module):
    """
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self,
                 input_dim,
                 eps=1E-6,
                 name=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(LayerNorm, self).__init__()
        if name is None:
            name = _get_name()

        self.input_dim = input_dim
        self.eps = eps

        name_w = name + "_layer_norm_w"
        name_b = name + "_layer_norm_b"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))
        try:
            weight = _get_shared(name_w)
        except NameError:
            weight_values = np.ones((input_dim,)).astype(np.float32)
            bias_values = np.zeros((input_dim,)).astype(np.float32)
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            bias = make_tensor(bias_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)
            _set_shared(name_b, bias)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # eps trick wont work here - std has issues if every element is 0 on that axis
        # std = x.std(-1, keepdim=True)
        # want to use sqrt of var + eps instead
        var = x.var(-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class PositionwiseFeedforward(nn.Module):
    def __init__(self, list_of_input_dims,
                 projection_dim,
                 dropout_keep_prob=1.0,
                 name=None,
                 random_state=None,
                 strict=None, init=None,
                 device="default",
                 dtype="default"):
        super(PositionwiseFeedforward, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to PositionwiseFeedforward")

        print("TODO ADD LOCKED DROPOUT IN POSITIONWISE FEEDFORWARD")

        name_i = name + "_positionwise_input_projection"
        name_o = name + "_positionwise_ouput_projection"
        o_dim = sum(list_of_input_dims)
        self.i = Linear(list_of_input_dims,
                        projection_dim,
                        random_state=random_state,
                        name=name_i,
                        strict=strict,
                        init=init,
                        device=device,
                        dtype=dtype)

        self.o = Linear([projection_dim],
                         o_dim,
                         biases=False,
                         random_state=random_state,
                         name=name_o,
                         strict=strict,
                         init=init,
                         device=device,
                         dtype=dtype)

        self.ln = LayerNorm(o_dim,
                            device=device,
                            dtype=dtype)

        print("COME BACK AND ADD DROPOUT")

    def forward(self, list_of_inputs):
        inp = torch.cat(list_of_inputs, dim=-1)
        s1 = relu(self.i([inp]))
        s2 = self.o([s1])
        return self.ln(s2 + inp)


def _rel_shift(x):#, zero_triu=False):
    x_padded = x.reshape(x.size(1), x.size(0), *x.size()[2:])
    x = x_padded[1:].reshape(x.size(0), x.size(1) - 1, *x.size()[2:])
    #if zero_triu:
    #    ones = torch.ones((x.size(0), x.size(1)))
    #    x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]
    return x


class RelativeMultiHeadAttention(nn.Module):
     def __init__(self, list_of_input_dims,
                  n_heads=10, head_dim=38, model_dim=380,
                  attention_dropout_keep_prob=1.0,
                  name=None,
                  random_state=None,
                  strict=None, init=None,
                  device="default",
                  dtype="default"):
         super(RelativeMultiHeadAttention, self).__init__()

         if name is None:
             name = _get_name()

         if random_state is None:
             raise ValueError("Must pass random_state to RelativeMultiHeadAttention")

         self.n_heads = n_heads
         self.head_dim = head_dim
         self.model_dim = model_dim
         self.attention_dropout_keep_prob = attention_dropout_keep_prob

         qkv_name = name + "_qkv"
         o_name = name + "_out"

         # need to add dropout
         print("TODO COME BACK AND ADD DROPOUT IN MH ATTN")

         # no biases in transformer XL code
         self.qkv_net = Linear(list_of_input_dims,
                               3 * n_heads * head_dim,
                               biases=False,
                               random_state=random_state,
                               name=qkv_name,
                               strict=strict,
                               init=init,
                               device=device,
                               dtype=dtype)

         self.o_net = Linear([head_dim * n_heads],
                             model_dim,
                             biases=False,
                             random_state=random_state,
                             name=o_name,
                             strict=strict,
                             init=init,
                             device=device,
                             dtype=dtype)

         self.ln = LayerNorm(model_dim,
                             device=device,
                             dtype=dtype)
         self.scale = 1. / (head_dim ** .5)

     # relative position embedding last entry is the bias? scuffed as...
     def forward(self, list_of_inputs, relative_positional_embedding, attention_mask=None, memory=None):
         i = torch.cat(list_of_inputs, dim=-1)
         r = relative_positional_embedding
         qlen = i.size(0)
         rlen = r.size(0)
         batch_size = i.size(1)
         if memory is not None:
             i_heads = self.qkv_net([torch.cat([memory, i], 0)])
         else:
             i_heads = self.qkv_net([i])

         r_heads = self.qkv_net([r])
         i_head_q, i_head_k, i_head_v = torch.chunk(i_heads, 3, dim=-1)
         r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)
         if memory is not None:
             # slice out the memory part for query
             i_head_q = i_head_q[-qlen:]
         klen = i_head_k.size(0)
         i_head_q = i_head_q.view(qlen, batch_size, self.n_heads, self.head_dim)
         # this could be much longer
         i_head_k = i_head_k.view(klen, batch_size, self.n_heads, self.head_dim)
         i_head_v = i_head_v.view(klen, batch_size, self.n_heads, self.head_dim)

         r_head_q = r_head_q.view(rlen, batch_size, self.n_heads, self.head_dim)
         r_head_k = r_head_k.view(rlen, batch_size, self.n_heads, self.head_dim)

         # attention
         # [qlen x bsz x n_head x d_head]
         ir_head_q = i_head_q + r_head_q[-1] # bias term from pos embed
         # [klen x bsz x n_head x d_head]
         # i_head_k
         # [qlen x klen x bsz x n_head]
         AC = torch.einsum('ibnd,jbnd->ijbn', (ir_head_q, i_head_k))

         ir2_head_q = i_head_q + r_head_q[-1] # bias term
         BD = torch.einsum('ibnd,jbnd->ijbn', (ir2_head_q, r_head_k))
         # rel shift effectively removes 1 along the 1st dim
         BD = _rel_shift(BD)
         attention_score = AC + BD
         # [qlen x klen x bsz x n_head]
         attention_score *= self.scale

         if attention_mask is not None and attention_mask.any().item():
             # fill 1s with neg ing, leave 0s alone!
             if attention_mask.dim() == 2:
                 attention_score.masked_fill_(attention_mask[None, :, :, None], -float('inf'))
             # can define either 2d or 3d mask here, but will need to be very careful about what is 0 and what is 1
             elif attention_mask.dim() == 3:
                 attention_score.masked_fill_(attention_mask[:, :, :, None], -float('inf'))

         attention_prob = F.softmax(attention_score, dim=1)
         attention_weighted_values = torch.einsum('ijbn,jbnd->ibnd', (attention_prob, i_head_v))

         # [qlen x bsz x n_head x d_head]
         attention_weighted_values = attention_weighted_values.contiguous().view(
                attention_weighted_values.size(0),
                attention_weighted_values.size(1),
                self.n_heads * self.head_dim)

         o = self.o_net([attention_weighted_values])
         output = self.ln(i + o)
         return output


class RelativeDecoderLayer(nn.Module):
    def __init__(self, list_of_input_dims, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
                 attention_dropout_keep_prob=0.8, inner_dropout_keep_prob=0.8,
                 name=None,
                 random_state=None,
                 strict=None, init=None,
                 device="default",
                 dtype="default"):
        super(RelativeDecoderLayer, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoder")

        attention_name = name + "_multihead_attention"
        feedforward_name = name + "_positionwise_ff"
        self.attention = RelativeMultiHeadAttention(list_of_input_dims,
                                                    n_heads=n_heads,
                                                    head_dim=head_dim,
                                                    model_dim=model_dim,
                                                    attention_dropout_keep_prob=attention_dropout_keep_prob,
                                                    name=attention_name,
                                                    random_state=random_state,
                                                    strict=strict,
                                                    init=init,
                                                    device=device,
                                                    dtype=dtype)

        self.position_ff = PositionwiseFeedforward(list_of_input_dims,
                                                   inner_dim,
                                                   dropout_keep_prob=inner_dropout_keep_prob,
                                                   name=feedforward_name,
                                                   random_state=random_state,
                                                   strict=strict,
                                                   init=init,
                                                   device=device,
                                                   dtype=dtype)

    def forward(self, decoder_input, relative_positional_embedding, decoder_attention_mask=None, memory=None):
        output = self.attention([decoder_input], relative_positional_embedding, attention_mask=decoder_attention_mask,
                                memory=memory)
        output = self.position_ff([output])
        return output


class AWDTransformerXLDecoderBlock(nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 n_layers=16, n_heads=10, head_dim=38, model_dim=380, inner_dim=900,
                 locked_drop_keep_prob=1.0, attention_dropout_keep_prob=0.8, inner_dropout_keep_prob=0.8,
                 name=None,
                 random_state=None,
                 mem_len=0,
                 context_len=0,
                 strict=None, init=None,
                 device="default",
                 dtype="default"):
        super(AWDTransformerXLDecoderBlock, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state to AWDTransformerXLDecoderBlock")

        self.layers = nn.ModuleList()
        input_dim = sum(list_of_input_dims)
        if input_dim != model_dim:
            raise ValueError("sum of list_of_input_dims should match model_dim due to residual architecture, if this is not the case project the data or change dims! Have {}, sum = {}, model_dim = {}".format(list_of_input_dims, input_dim, model_dim))
        if n_heads * head_dim != model_dim:
            raise ValueError("head_dim * n_heads should == model_dim, have {} * {} != {}".format(head_dim, n_heads, model_dim))
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype
        for i in range(n_layers):
            layer_name = name + "_relative_decoder_layer{}".format(i)
            l = RelativeDecoderLayer(list_of_input_dims,
                                     n_heads=n_heads,
                                     head_dim=head_dim,
                                     model_dim=model_dim,
                                     inner_dim=inner_dim,
                                     attention_dropout_keep_prob=attention_dropout_keep_prob,
                                     inner_dropout_keep_prob=inner_dropout_keep_prob,
                                     name=layer_name,
                                     random_state=random_state,
                                     strict=strict,
                                     init=init,
                                     device=device,
                                     dtype=dtype)
            self.layers.append(l)

        self.mem_len = mem_len
        self.context_len = context_len
        self.pos_emb = PositionalEmbedding(model_dim)

    def init_list_of_mems(self):
        if self.device == "default":
            device = get_device_default()
        else:
            device = self.device

        if self.dtype == "default":
            dtype = get_dtype_default()
        if dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float64

        # returns None if mem_len is 0 else
        if self.mem_len > 0:
            mems = []

            for i in range(self.n_layers):
                empty = torch.empty(0, dtype=dtype, device=torch.device(device))
                mems.append(empty)
            return mems
        else:
            return None

    def update_list_of_mems(self, hiddens, list_of_mems, query_len, memory_len):
        # mlen and qlen were swapped in call vs signature in original code!
        # https://github.com/kimiyoung/transformer-xl/issues/96
        # effectively, would mean memory_len= len query
        # and query_len= 0 for PTB experiment
        # where self.context_len was 70
        # self.mem_len 0
        # we swap the call to be correct, and set hyperparameters to actually use memory
        if list_of_mems is None:
            return None

        if self.mem_len == 0:
            return None

        qlen = query_len
        mlen = memory_len

        assert len(hiddens) == len(list_of_mems), "len(list_of_mems) != len(hiddens)"

        # Copied from transformer XL main code
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.context_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hiddens)):
                cat = torch.cat([list_of_mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems


    def forward(self, input_tensor, list_of_mems=None):
        if not list_of_mems:
            list_of_mems = self.init_list_of_mems()

        shp = input_tensor.shape

        qlen = shp[0]
        mlen = list_of_mems[0].size(0) if list_of_mems is not None else 0
        klen = mlen + qlen
        # masking works internally by setting 1s to neg inf, 0s are left alone! This is slightly different than expected
        attn_mask = torch.triu(input_tensor.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]

        # relative positional embedding
        pos_seq = torch.arange(klen, -1, -1.0, device=input_tensor.device)
        pe = self.pos_emb(pos_seq, batch_size=shp[1])
        # one longer than sequence because pe[-1] is a fixed "bias"
        # and because _rel_shift reduces size by 1

        hids = []
        print("TODO ADD LOCKED DROPOUT")
        core_out = input_tensor
        for i, this_layer in enumerate(self.layers):
            hids.append(core_out)
            mems_i = list_of_mems[i] if list_of_mems is not None else None
            core_out = this_layer(core_out, pe, decoder_attention_mask=attn_mask, memory=mems_i)

        # update memory
        # mlen = 0
        # qlen = len(inpt)
        #new_mems = self.update_list_of_mems(hids, list_of_mems, mlen, qlen)
        # original code had a bug, see comments for detail
        new_mems = self.update_list_of_mems(hids, list_of_mems, qlen, mlen)

        # slice according to context_len, normally set to 0
        # in original code they do this via target size, but we don't have that information
        core_out = core_out[self.context_len:]
        return core_out, new_mems
