import os
import random

import h5py
import numpy as np
import paddle


def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)


def collate(X):
    N_max = max([tuple(input["node_pos"].shape)[-2] for [input, t] in X])
    E_max = max([tuple(input["edges"].shape)[-2] for [input, t] in X])
    N_all = paddle.zeros(shape=len(X))
    E_all = paddle.zeros(shape=len(X))
    mask = []
    for batch, [input, t] in enumerate(X):
        tensor = input["node_pos"]
        N, S = tuple(tensor.shape)
        input["node_pos"] = paddle.concat(
            x=[tensor, paddle.zeros(shape=[N_max - N + 1, S])], axis=-2
        )
        N_all[batch] = N
        mask_i = paddle.zeros(shape=N_max)
        mask_i[:N] = 1
        mask.append(mask_i)
        gt = input["gt"]
        if gt.ndim == 1:
            pass
        else:
            N, S = tuple(gt.shape)
            input["gt"] = paddle.concat(
                x=[gt, paddle.zeros(shape=[N_max - N + 1, S])], axis=-2
            )
            N_all[batch] = N
        edges = input["edges"].astype(paddle.float32)
        E, S = tuple(edges.shape)
        input["edges"] = paddle.concat(
            x=[edges, N_max * paddle.ones(shape=[E_max - E + 1, S])], axis=-2
        )
        E_all[batch] = E
    batch_in = {key: None for key in ["node_pos", "edges", "gt"]}
    mask = paddle.stack(x=mask, axis=0)
    for key in batch_in.keys():
        batch_in[key] = paddle.stack(x=[x[0][key] for x in X], axis=0)
    batch_in["mask"] = mask
    names = [x[1] for x in X]
    return batch_in, names
