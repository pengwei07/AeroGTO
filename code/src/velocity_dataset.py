import sys

sys.path.append("/workspace/AeroGTO/paddle_project")
import os

import numpy as np
import paddle
import vtk
from paddle_utils import *
from vtk.util.numpy_support import vtk_to_numpy


def read_vtk(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    num_points = points.GetNumberOfPoints()
    point_coords = np.array(
        [points.GetPoint(i) for i in range(num_points)], dtype=np.float32
    )
    cells = data.GetCells()
    edges = []
    cell_types = data.GetCellTypesArray()
    num_cells = cell_types.GetNumberOfTuples()
    hexahedron_type = vtk.VTK_HEXAHEDRON
    for i in range(num_cells):
        if cell_types.GetValue(i) == hexahedron_type:
            cell = vtk.vtkIdList()
            cells.GetCellAtId(i, cell)
            cell_ids = [cell.GetId(j) for j in range(cell.GetNumberOfIds())]
            edges.extend(
                [(cell_ids[j], cell_ids[j + 1]) for j in range(len(cell_ids) - 1)]
            )
            edges.append((cell_ids[-1], cell_ids[0]))
    edges = np.array(edges, dtype=np.int32)
    velocity = vtk_to_numpy(data.GetPointData().GetArray("point_vectors")).astype(
        np.float32
    )
    return point_coords, edges, velocity


def get_data(path, item):
    file_path = path + "vel_" + item + ".vtk"
    point_coords, edges, velocity = read_vtk(file_path)
    return point_coords, edges, velocity


class Velocity_Dataset(paddle.io.Dataset):
    def __init__(self, data_path):
        super(Velocity_Dataset, self).__init__()
        self.dataloc_train = []
        self.dataloc_val = []
        self.fn = data_path
        file_list = os.listdir(f"{self.fn}")
        for file_name in file_list:
            if file_name.startswith("vel"):
                split_string = file_name.split(".")
                split_string1 = split_string[0].split("_")
                self.dataloc_train.append(split_string1[1])
        self.dataloc_train = np.array(self.dataloc_train, dtype=np.str_)
        self.dataloc = self.dataloc_train

    def __len__(self):
        return len(self.dataloc)

    def get_singel(self, item):
        node_pos, edges, velocity = get_data(self.fn, self.dataloc[item])
        node_pos = paddle.to_tensor(data=node_pos).astype(dtype="float32")
        node_pos = self.scale_pos(node_pos)
        velocity = paddle.to_tensor(data=velocity).astype(dtype="float32")
        edges = paddle.to_tensor(data=edges).astype(dtype="int64")
        input = {"node_pos": node_pos, "gt": velocity, "edges": edges}
        return input, self.dataloc[item]

    def scale_pos(self, pos):
        pos_min = paddle.to_tensor(data=[-1.5696, -0.2591, -3.148]).to(pos.place)
        pos_max = paddle.to_tensor(data=[1.5658, 2.0213, 4.1869]).to(pos.place)
        node_pos = (pos - pos_min.reshape(-1, 3)) / (
            pos_max.reshape(-1, 3) - pos_min.reshape(-1, 3)
        )
        return node_pos

    def __getitem__(self, item):
        input, name = self.get_singel(item)
        return input, name


if __name__ == "__main__":
    dataset = Velocity_Dataset(data_path="../../data/velocity/")
    for i in range(len(dataset)):
        input, name = dataset[i]
        print(input)
        print(name)
        break
