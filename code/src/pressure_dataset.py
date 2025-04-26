import os.path
import numpy as np
from plyfile import PlyData

import torch
from torch.utils.data import Dataset

def faces_to_edges_both(faces):
    edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=0)
    
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0)
    return unique_edges

def faces_to_edges_single(faces):
    edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)

    # a random boolean tensor
    swap_mask = torch.rand(unique_edges.shape[:-1]) > 0.5

    senders_swapped = torch.where(swap_mask, unique_edges[..., 1], unique_edges[..., 0])
    receivers_swapped = torch.where(swap_mask, unique_edges[..., 0], unique_edges[..., 1])
    randomized_unique_edges = torch.stack([senders_swapped, receivers_swapped], dim=-1)

    return randomized_unique_edges

def get_data(path, item):
    
    fea_path = path + "mesh_" + item + ".ply"
    lab_path = path + "press_" + item + ".npy"
    
    ply_data = PlyData.read(fea_path)
    
    vertex_data = ply_data['vertex']
    vertices = [(vertex['x'], vertex['y'], vertex['z']) for vertex in vertex_data]
    
    face_data = ply_data['face']
    faces = [face['vertex_indices'] for face in face_data]

    node_pos = np.array(vertices).astype(np.float32)
    cells = np.array(faces).astype(np.int32)
    # (3586, 3) (7168, 3)
    label_data = np.load(lab_path).astype(np.float32) # (3682,)
    pressure = np.concatenate((label_data[0:16], label_data[112:]), axis=0) # (3586,)
        
    return node_pos, cells, pressure.reshape(-1,1)

class Pressure_Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 edge_type="both",
                 ):

        super(Pressure_Dataset, self).__init__()
        
        self.dataloc_train = []
        self.dataloc_val = []
        
        self.fn = data_path
        self.edge_type = edge_type
        
        file_list = os.listdir(f"{self.fn}") 
        for file_name in file_list:
            if file_name.startswith('mesh'):
                split_string = file_name.split('.')
                split_string1 = split_string[0].split('_')
                self.dataloc_train.append(split_string1[1]) 
        self.dataloc_train = np.array(self.dataloc_train, dtype=np.str_)
        
        self.dataloc = self.dataloc_train
        

    def __len__(self):
        return len(self.dataloc)
    
    def get_singel(self, item):
        
        node_pos, cells, pressure = get_data(
            self.fn,
            self.dataloc[item]
            )
        node_pos = torch.from_numpy(node_pos).float()
        node_pos = self.scale_pos(node_pos)
            
        pressure = torch.from_numpy(pressure).float() 

        cells = torch.from_numpy(cells).long()
        if self.edge_type == "both":
            edges = faces_to_edges_both(cells)
        elif self.edge_type == "single":
            edges = faces_to_edges_single(cells)
            
        input = {
            'node_pos': node_pos,
            'gt': pressure,
            'edges': edges
            }
        
        return input, self.dataloc[item]
    
    def scale_pos(self, pos):

        pos_min = torch.tensor([-0.9025, 0.0067, -2.8527]).to(pos.device)
        pos_max = torch.tensor([0.9025, 1.9583, 2.8639]).to(pos.device)
        
        node_pos = (pos - pos_min.reshape(-1,3)) / (pos_max.reshape(-1,3) - pos_min.reshape(-1,3))
       
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