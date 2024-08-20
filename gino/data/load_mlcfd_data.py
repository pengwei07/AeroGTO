import os
import numpy as np
from pathlib import Path
import meshio

import torch


def load_mlcfd_data(path: Path, selection="pressure"):

    if selection == "pressure":
        mesh = meshio.read(path / "quadpress_smpl.vtk")
        vertice = mesh.points
        vertice = torch.tensor(vertice, dtype=torch.float)
        press = np.load(path / "press.npy").reshape(-1, 1)
        press = torch.tensor(press, dtype=torch.float)
        return vertice, press
    else:
        mesh = meshio.read(path / "hexvelo_smpl.vtk")
        vertice = mesh.points
        vertice = torch.tensor(vertice, dtype=torch.float)
        velo = np.load(path / "velo.npy").reshape(-1, 3)
        return vertice, velo


def load_mlcfd_dataset(path, n):
    # get list of dirs
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs.sort()
    print(f"Loading {n} folders from {path} ...")

    # load all data
    vertices_list = []
    press_list = []
    for d in dirs[:n]:
        vertice, press = load_mlcfd_data(path + d)
        vertices_list.append(vertice)
        press_list.append(press)
        print(d, vertice.shape, press.shape)
    return vertices_list, press_list


def load_mlcfd_dataset_tensor(path: Path, n: int = None, selection: str = "pressure"):
    """
    Load mlcfd dataset into tensors
    :param path: Path to the dataset
    :param n: Number of folders to load. If None, load all folders
    :param selection: "pressure" or "velocity"
    """
    assert path.exists(), f"Path {path} does not exist"
    # get list of dirs
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs.sort()

    # set n to number of dirs if n is None
    if n == None:
        n = len(dirs)
    print(f"Loading {n} folders from {path} ...")

    # load all data
    if selection == "pressure":
        vertices_list = torch.zeros(n, 3682, 3)
        press_list = torch.zeros(n, 3682, 1)
        for i, d in enumerate(dirs[:n]):
            vertice, press = load_mlcfd_data(path / d, selection=selection)
            vertices_list[i] = vertice
            press_list[i] = press
            # print(i, d, vertice.shape, press.shape)
        return vertices_list, press_list
    else:
        vertices_list = torch.zeros(n, 29498, 3)
        velo_list = torch.zeros(n, 29498, 3)
        for i, d in enumerate(dirs[:n]):
            vertice, velo = load_mlcfd_data(path / d, selection=selection)
            vertices_list[i] = vertice
            velo_list[i] = torch.FloatTensor(velo)
            # print(i, d, vertice.shape, press.shape)
        return vertices_list, velo_list
    # return vertices_list[:,112:3682], press_list[:,112:3682]


def load_mlcfd_dataset_full(path: Path, selection: str = "pressure"):
    # get list of dirs
    vertices_list = []
    press_list = []
    if isinstance(path, str):
        path = Path(path)

    # expand user
    path = path.expanduser()

    for i in range(9):
        path_i = path / f"param{i}"
        vertices, press = load_mlcfd_dataset_tensor(path_i, selection=selection)
        vertices_list.append(vertices)
        press_list.append(press)

    vertices_list = torch.cat(vertices_list, dim=0)
    press_list = torch.cat(press_list, dim=0)
    return vertices_list, press_list
