import torch
def space2batch(x, axis):
    if axis not in [2, 3]:
        raise ValueError("space2batch only operates on axis 2 or 3, as 2D data is assumed (n_batch, channel (depth), Y, X")
    if axis == 2:
        x_flat = x.permute(3, 1, 0, 2).reshape(x.shape[3], x.shape[1], -1)
        x_flat = x_flat.permute(0, 2, 1)
    elif axis == 3:
        x_flat = x.permute(2, 1, 0, 3).reshape(x.shape[2], x.shape[1], -1)
        x_flat = x_flat.permute(0, 2, 1)
    return x_flat.contiguous()


def batch2space(x, n_batch, axis):
    if axis not in [2, 3]:
        raise ValueError("batch2space only operates on axis 2 or 3, as 1D input is assumed (n_batch * n_original_dim, channel (depth), feat")
    x_rec = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], n_batch, x.shape[1] // n_batch)
    if axis == 2:
        x_rec = x_rec.permute(2, 1, 3, 0)
    elif axis == 3:
        x_rec = x_rec.permute(2, 1, 0, 3)
    return x_rec.contiguous()


def split(x, axis):
    assert x.shape[axis] % 2 == 0
    if axis not in [2, 3]:
        raise ValueError("split only operates on axis 2 or 3, 4D input assumed n_batch, channel (depth), Y, X")
    if axis == 2:
        return x[:, :, 0::2, :], x[:, :, 1::2, :]
    elif axis == 3:
        return x[:, :, :, 0::2], x[:, :, :, 1::2]


def interleave(x_1, x_2, axis):
    if axis == 2:
        c = torch.empty(x_1.shape[0], x_1.shape[1], 2 * x_1.shape[2], x_1.shape[3])
        c[:, :, ::2, :] = x_1
        c[:, :, 1::2, :] = x_2
        return c
    if axis == 3:
        c = torch.empty(x_1.shape[0], x_1.shape[1], x_1.shape[2], 2 * x_1.shape[3])
        c[:, :, :, ::2] = x_1
        c[:, :, :, 1::2] = x_2
        return c
