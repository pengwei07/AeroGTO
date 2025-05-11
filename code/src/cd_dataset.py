import sys

sys.path.append("/workspace/AeroGTO/paddle_project")

import numpy as np
import paddle
import pandas as pd
from paddle_utils import *


def faces_to_edges_both(faces):
    edges = paddle.concat(x=[faces[:, :2], faces[:, 1:], faces[:, ::2]], axis=0)
    receivers, _ = paddle.min(x=edges, axis=-1), paddle.argmin(x=edges, axis=-1)
    senders, _ = paddle.max(x=edges, axis=-1), paddle.argmax(x=edges, axis=-1)
    packed_edges = paddle.stack(x=[senders, receivers], axis=-1).astype(dtype="int32")
    unique_edges = paddle.unique(x=packed_edges, axis=0)
    unique_edges = paddle.concat(
        x=[unique_edges, paddle.flip(x=unique_edges, axis=[-1])], axis=0
    )
    return unique_edges


def faces_to_edges_single(faces):
    edges = paddle.concat(x=[faces[:, :2], faces[:, 1:], faces[:, ::2]], axis=1)
    receivers, _ = paddle.min(x=edges, axis=-1), paddle.argmin(x=edges, axis=-1)
    senders, _ = paddle.max(x=edges, axis=-1), paddle.argmax(x=edges, axis=-1)
    packed_edges = paddle.stack(x=[senders, receivers], axis=-1).astype(dtype="int32")
    unique_edges = paddle.unique(x=packed_edges, axis=1)
    swap_mask = paddle.rand(shape=tuple(unique_edges.shape)[:-1]) > 0.5
    senders_swapped = paddle.where(
        condition=swap_mask, x=unique_edges[..., 1], y=unique_edges[..., 0]
    )
    receivers_swapped = paddle.where(
        condition=swap_mask, x=unique_edges[..., 0], y=unique_edges[..., 1]
    )
    randomized_unique_edges = paddle.stack(
        x=[senders_swapped, receivers_swapped], axis=-1
    )
    return randomized_unique_edges


def read_obj_file(filename):
    vertices = []
    faces = []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("v "):
                vertex = line.strip().split()[1:]
                vertices.append([float(coord) for coord in vertex])
            elif line.startswith("f "):
                face = line.strip().split()[1:]
                faces.append([int(index.split("/")[0]) for index in face])
    vertices = np.stack(vertices, axis=0)
    faces = np.stack(faces, axis=0)
    return vertices, faces


def read_csv(filename):
    data = pd.read_csv(filename)
    index = np.array(data.iloc[:, 0], dtype=np.int16)
    file_list = np.array(data.iloc[:, 1], dtype=np.str_)
    Cd = np.array(data.iloc[:, 2], dtype=np.float32)
    for i, item in enumerate(file_list):
        if item == "nan":
            valid_num = i
            break
    return index[:valid_num], file_list[:valid_num], Cd[:valid_num]


def get_data(path, item):
    vertices, faces = read_obj_file(path + item + ".obj")
    node_pos = np.array(vertices).astype(np.float32)
    cells = np.array(faces).astype(np.int32)
    return node_pos, cells


class Drag_Dataset(paddle.io.Dataset):
    def __init__(self, data_path="../", edge_type="both"):
        super(Drag_Dataset, self).__init__()
        self.fn = data_path
        self.edge_type = edge_type
        index, file_list, Cd = read_csv(data_path + "/cd_label.csv")
        self.dataloc = file_list
        self.Cd = Cd

    def __len__(self):
        return len(self.dataloc)

    def get_singel(self, item):
        node_pos, cells = get_data(self.fn, self.dataloc[item])
        node_pos = paddle.to_tensor(data=node_pos).astype(dtype="float32")
        node_pos = self.scale_pos(node_pos)
        drag = paddle.to_tensor(data=self.Cd[item]).astype(dtype="float32")
        cells = paddle.to_tensor(data=cells - 1).astype(dtype="int64")
        if self.edge_type == "both":
            edges = faces_to_edges_both(cells)
        elif self.edge_type == "single":
            edges = faces_to_edges_single(cells)
        input = {"node_pos": node_pos, "gt": drag.unsqueeze(axis=-1), "edges": edges}
        return input, self.dataloc[item]

    def scale_pos(self, pos):
        pos_min = paddle.to_tensor(data=[-1.76815, 0.005625, 0.000285]).to(pos.place)
        pos_max = paddle.to_tensor(data=[1.769428, 1.664876, 0.708111]).to(pos.place)
        node_pos = (pos - pos_min.reshape(-1, 3)) / (
            pos_max.reshape(-1, 3) - pos_min.reshape(-1, 3)
        )
        return node_pos

    def __getitem__(self, item):
        input, name = self.get_singel(item)
        return input, name


if __name__ == "__main__":
    dataset = Drag_Dataset(data_path="../../data/cd/")
    for i in range(len(dataset)):
        input, name = dataset[i]
        print(input)
        print(name)
        break
