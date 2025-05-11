import sys

sys.path.append("/workspace/AeroGTO/paddle_project")
import os

import numpy as np
import paddle
from paddle_utils import *
from plyfile import PlyData


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


def get_data(path, item):
    fea_path = path + "mesh_" + item + ".ply"
    lab_path = path + "press_" + item + ".npy"
    ply_data = PlyData.read(fea_path)
    vertex_data = ply_data["vertex"]
    vertices = [(vertex["x"], vertex["y"], vertex["z"]) for vertex in vertex_data]
    face_data = ply_data["face"]
    faces = [face["vertex_indices"] for face in face_data]
    node_pos = np.array(vertices).astype(np.float32)
    cells = np.array(faces).astype(np.int32)
    label_data = np.load(lab_path).astype(np.float32)
    pressure = np.concatenate((label_data[0:16], label_data[112:]), axis=0)
    return node_pos, cells, pressure.reshape(-1, 1)


class Pressure_Dataset(paddle.io.Dataset):
    def __init__(self, data_path, edge_type="both"):
        super(Pressure_Dataset, self).__init__()
        self.dataloc_train = []
        self.dataloc_val = []
        self.fn = data_path
        self.edge_type = edge_type
        file_list = os.listdir(f"{self.fn}")
        for file_name in file_list:
            if file_name.startswith("mesh"):
                split_string = file_name.split(".")
                split_string1 = split_string[0].split("_")
                self.dataloc_train.append(split_string1[1])
        self.dataloc_train = np.array(self.dataloc_train, dtype=np.str_)
        self.dataloc = self.dataloc_train

    def __len__(self):
        return len(self.dataloc)

    def get_singel(self, item):
        node_pos, cells, pressure = get_data(self.fn, self.dataloc[item])
        node_pos = paddle.to_tensor(data=node_pos).astype(dtype="float32")
        node_pos = self.scale_pos(node_pos)
        pressure = paddle.to_tensor(data=pressure).astype(dtype="float32")
        cells = paddle.to_tensor(data=cells).astype(dtype="int64")
        if self.edge_type == "both":
            edges = faces_to_edges_both(cells)
        elif self.edge_type == "single":
            edges = faces_to_edges_single(cells)
        input = {"node_pos": node_pos, "gt": pressure, "edges": edges}
        return input, self.dataloc[item]

    def scale_pos(self, pos):
        pos_min = paddle.to_tensor(data=[-0.9025, 0.0067, -2.8527]).to(pos.place)
        pos_max = paddle.to_tensor(data=[0.9025, 1.9583, 2.8639]).to(pos.place)
        node_pos = (pos - pos_min.reshape(-1, 3)) / (
            pos_max.reshape(-1, 3) - pos_min.reshape(-1, 3)
        )
        return node_pos

    def __getitem__(self, item):
        input, name = self.get_singel(item)
        return input, name


if __name__ == "__main__":
    dataset = Pressure_Dataset(data_path="../../data/pressure/")
    for i in range(len(dataset)):
        input, name = dataset[i]
        print(input)
        print(name)
        break
