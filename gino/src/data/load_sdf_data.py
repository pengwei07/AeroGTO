import torch
import open3d as o3d
import numpy as np
import os
from pathlib import Path
from neuralop.utils import UnitGaussianNormalizer

def load_car_pressure_sdf(
    path: Path,
    n_train=500,
    n_test=111,
    spatial_resolution=(64, 64, 64),
    pos_encoding=True,
    vertex_norm_range=(-1, 1),
    eps=0.01,
    norm_pressure=True,
):
    if isinstance(path, str):
        path = Path(path)

    path = path.expanduser()
    assert path.exists(), "Path does not exist"
    assert path.is_dir(), "Path is not a directory"

    with open(path / "watertight_global_bounds.txt", "r") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")

        min_bounds = [float(a) - eps for a in min_bounds]
        max_bounds = [float(a) + eps for a in max_bounds]

    with open(path / "watertight_meshes.txt", "r") as fp:
        mesh_ind = fp.read().split("\n")
        mesh_ind = [int(a) for a in mesh_ind]

    n = len(mesh_ind)
    assert n_train + n_test <= n, "Not enough data"

    if pos_encoding:
        train_x = torch.zeros(n_train, 4, *spatial_resolution)
        test_x = torch.zeros(n_test, 4, *spatial_resolution)
    else:
        train_x = torch.zeros(n_train, 1, *spatial_resolution)
        test_x = torch.zeros(n_test, 1, *spatial_resolution)

    train_y = torch.zeros(n_train, 3586, 4)
    test_y = torch.zeros(n_test, 3586, 4)

    tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
    ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
    tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
    query_points = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(
        np.float32
    )

    if pos_encoding:
        tx = np.linspace(0, 1, spatial_resolution[0])
        ty = np.linspace(0, 1, spatial_resolution[1])
        tz = np.linspace(0, 1, spatial_resolution[2])
        pos_enc = torch.from_numpy(
            np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1)
            .astype(np.float32)
            .transpose((3, 0, 1, 2))
        )

    for j in range(n_train):
        mesh_path = path / "data" / ("mesh_" + str(mesh_ind[j]).zfill(3) + ".ply")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(query_points).numpy()

        train_x[j, 0, ...] = torch.from_numpy(signed_distance)
        if pos_encoding:
            train_x[j, 1:, ...] = pos_enc

        press_path = path / "data" / ("press_" + str(mesh_ind[j]).zfill(3) + ".npy")

        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        press = np.concatenate((press[0:16], press[112:]), axis=0)
        train_y[j, :, 0:3] = torch.from_numpy(mesh.vertex.positions.numpy())
        train_y[j, :, 3] = torch.from_numpy(press)

    for j in range(n_test):
        mesh_path = (
            path / "data" / ("mesh_" + str(mesh_ind[-(j + 1)]).zfill(3) + ".ply")
        )

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(query_points).numpy()

        test_x[j, 0, ...] = torch.from_numpy(signed_distance)
        if pos_encoding:
            test_x[j, 1:, ...] = pos_enc

        press_path = (
            path / "data" / ("press_" + str(mesh_ind[-(j + 1)]).zfill(3) + ".npy")
        )

        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        press = np.concatenate((press[0:16], press[112:]), axis=0)
        test_y[j, :, 0:3] = torch.from_numpy(mesh.vertex.positions.numpy())
        test_y[j, :, 3] = torch.from_numpy(press)

    if vertex_norm_range is not None:
        scale = vertex_norm_range[1] - vertex_norm_range[0]
        shift = vertex_norm_range[0]
        for j in range(3):
            train_y[:, :, j] = (
                scale
                * ((train_y[:, :, j] - min_bounds[j]) / (max_bounds[j] - min_bounds[j]))
                + shift
            )
            test_y[:, :, j] = (
                scale
                * ((test_y[:, :, j] - min_bounds[j]) / (max_bounds[j] - min_bounds[j]))
                + shift
            )

    encoder = None
    if norm_pressure:
        encoder = UnitGaussianNormalizer(
            train_y[:, :, 3], eps=1e-6, reduce_dim=[0, 1], verbose=False
        )
        train_y[:, :, 3] = encoder.encode(train_y[:, :, 3])

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    test_ds = torch.utils.data.TensorDataset(test_x, test_y)

    return train_ds, test_ds, encoder
