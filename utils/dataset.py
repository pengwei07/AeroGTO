import os.path
import random
from torch.utils.data import Dataset
import torch
import numpy as np
from plyfile import PlyData
from sklearn.neighbors import KDTree

def get_data(path, item, mode, if_sample):
    # Define paths to different data files
    info_path = path + '/' + mode + "/" + "info_" + item + ".pt"
    press_path = path + '/' + mode + "/" + "press_" + item + ".npy"
    centroids_path = path + '/data/' + mode + "/" + "centroids_" + item + ".npy"
    areas_path = path + '/data/' + mode + "/" + "areas_" + item + ".npy"
    edges_path = path + '/edges/' + mode + "/" + "cell_edges_" + item + ".npy"
    dragWeight_path = path + '/dragWeight_test/' + "dragWeight_" + item + ".npy"
    
    # Load data from files
    centroids = np.load(centroids_path).astype(np.float32) 
    areas = np.load(areas_path).astype(np.float32) 
    pressure = np.load(press_path).reshape(-1,1).astype(np.float32) 
    info = torch.load(info_path)
    
    # Determine which data to return based on if_sample and mode
    if if_sample:
        if mode == "train":
            inputs_all = {
                "centroids": centroids,
                "areas": areas,
                "pressure": pressure,
                "info": info
            }
        elif mode == "test":
            dragWeight = np.load(dragWeight_path).reshape(-1,1).astype(np.float32) 
            inputs_all = {
                "centroids": centroids,
                "areas": areas,
                "pressure": pressure,
                "info": info,
                "dragWeight": dragWeight
            }
    else:
        edges = np.load(edges_path).astype(np.int32)
        if mode == "train":
            inputs_all = {
                "centroids": centroids,
                "areas": areas,
                "edges": edges,
                "pressure": pressure,
                "info": info
            }
        elif mode == "test":
            dragWeight = np.load(dragWeight_path).reshape(-1,1).astype(np.float32) 
            inputs_all = {
                "centroids": centroids,
                "areas": areas,
                "edges": edges,
                "pressure": pressure,
                "info": info,
                "dragWeight": dragWeight
            }   
            
    return inputs_all

class Car_Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 mode="train",
                 if_sample = True,
                 adj_num = 15,
                 edges_ratio = 1,
                 node_ratio = 2,
                 if_zero_shot = False
                 ):

        super(Car_Dataset, self).__init__()
        assert mode in ["train", "test"]

        self.dataloc = []
        self.fn = data_path
        self.mode = mode
        self.if_sample = if_sample
        self.edges_ratio = 1 / edges_ratio
        self.node_ratio = 1 / node_ratio
        if self.if_sample:
            self.adj_num = 15
            
        self.if_zero_shot = if_zero_shot
        
        # Populate the dataset based on the mode
        if self.mode == "train":
            for i in range(500):
                str_i = i + 1
                str_ii = "{:03}".format(str_i)
                self.dataloc.append(str_ii) 
        elif self.mode == "test":
            for i in range(51):
                str_i = i + 1
                str_ii = "{:03}".format(str_i)
                self.dataloc.append(str_ii) 

    def __len__(self):
        return len(self.dataloc)
    
    def get_edges(self, new_edges):
        
        # Flatten the new_edges tensor and remove invalid edges
        flat_edges = new_edges[(new_edges[:,:,0] != -1) & (new_edges[:,:,1] != -1)]
        # Reshape the flattened tensor to have shape [L, 2]
        flat_edges = flat_edges.view(-1, 2)
        
        # Get unique edges
        unique_edges = torch.unique(flat_edges, dim=0)
                
        return unique_edges
    
    def get_singel(self, item):
        
        inputs_all = get_data(
            self.fn,
            self.dataloc[item],
            self.mode,
            self.if_sample
            )

        centroids = torch.from_numpy(inputs_all["centroids"]).float()
        areas = torch.from_numpy(inputs_all["areas"]).float()
        pressure = torch.from_numpy(inputs_all["pressure"]).float()
        
        centroids_01 = self.scale_pos(centroids)
        areas_01 = self.scale_area(areas)
        info_01 = self.scale_info(inputs_all["info"])
        
        if self.if_sample:
            
            if self.mode == "test":
                
                if self.if_zero_shot:
                    dragWeight = torch.from_numpy(inputs_all["dragWeight"]).float()
                    # Perform node and edge sampling
                    all_sampled_centroids, all_batches_indices = self.sample_node_all(centroids, self.node_ratio)
                    
                    all_nodes = []
                    all_edges = []
                    all_press = []
                    all_areas = []
                    all_dragWeight = []
                    
                    for index, centroids_i in enumerate(all_sampled_centroids):
                        
                        indices = all_batches_indices[index]
                        
                        # Edge sampling
                        edges_ = self.knn_edges(centroids_i, self.adj_num)
                        edges = self.sample_edges_0(edges_, self.edges_ratio)
                        all_edges.append(edges)
                        
                        # Node processing
                        all_nodes.append(centroids_01[indices])
                        
                        # Pressure processing
                        all_press.append(pressure[indices])
                        
                        # Area processing
                        all_areas.append(areas_01[indices])
                        
                        # Drag weight processing
                        all_dragWeight.append(inputs_all["dragWeight"][indices])
                    
                    input = {'node_pos': all_nodes,
                             'edges': all_edges,
                             'pressure': all_press,
                             'info': info_01,
                             'areas': all_areas,
                             'dragWeight': all_dragWeight
                            }
                else:
                    # Perform node and edge sampling
                    centroids_sampled, indices = self.sample_node(centroids, self.node_ratio)
                    
                    edges_ = self.knn_edges(centroids_sampled, self.adj_num)
                    edges = self.sample_edges_0(edges_, self.edges_ratio)
                    
                    input = {'node_pos': centroids_01[indices],
                            'edges': edges,
                            'pressure': pressure[indices],
                            'info': info_01,
                            'areas': areas_01[indices]
                            }

            else:
                # Perform node and edge sampling
                centroids_sampled, indices = self.sample_node(centroids, self.node_ratio)
                
                edges_ = self.knn_edges(centroids_sampled, self.adj_num)
                edges = self.sample_edges_0(edges_, self.edges_ratio)
                
                input = {'node_pos': centroids_01[indices],
                        'edges': edges,
                        'pressure': pressure[indices],
                        'info': info_01,
                        'areas': areas_01[indices]
                        }
        else:   
            edges = torch.from_numpy(inputs_all["edges"]).long()
            edges = self.get_edges(edges)
            edges = self.sample_edges_1(edges, self.edges_ratio)
            
            if self.mode == "test":
                dragWeight = torch.from_numpy(inputs_all["dragWeight"]).float()
                input = {'node_pos': centroids_01,
                        'edges': edges,
                        'pressure': pressure,
                        'info': info_01,
                        'areas': areas_01,
                        'dragWeight': dragWeight
                        }
            else:
                input = {'node_pos': centroids_01,
                        'edges': edges,
                        'pressure': pressure,
                        'info': info_01,
                        'areas': areas_01
                        }
            
        return input, self.dataloc[item] 
    
    def knn_edges(self, node_pos, k):        
        # Construct KDTree for k-nearest neighbors
        node_pos = node_pos.numpy()
        tree = KDTree(node_pos)
        # Query the nearest k+1 neighbors (including the point itself)
        dists, indices = tree.query(node_pos, k=k+1) 
                
        # Create edge indices
        num_points = node_pos.shape[0]
        i_indices = torch.arange(num_points).view(-1, 1).repeat(1, k+1)
        
        # Reshape edges array to [N, k, 2]
        edges = torch.stack([i_indices, torch.from_numpy(indices)], dim=-1)
        
        return edges.long()
    
    def sample_edges_0(self, edges, ratio):
        # Get number of nodes N
        N = edges.shape[0]
        k = edges.shape[1]
        # Generate random indices for edge sampling
        random_indices = torch.randint(0, k, (N,))
        # Sample edges based on random indices
        sample_edges = edges[torch.arange(N), random_indices]
        
        if int(ratio) == 1:
            return sample_edges
        else:
            # Calculate the sample size
            sample_size = int(N * ratio)
            # Generate random indices for sampling
            indices = torch.randperm(N)[:sample_size]
            # Sample edges
            sampled = sample_edges[indices]
            return sampled
    
    def sample_edges_1(self, edges, ratio):
        # Get number of edges L
        L = edges.shape[0]
        # Calculate the sample size
        sample_size = int(L * ratio)
        # Generate random indices for sampling
        indices = torch.randperm(L)[:sample_size]
        # Sample edges
        sampled = edges[indices]
        
        return sampled

    def sample_node(self, nodes, ratio):
        # Get number of nodes N
        N = nodes.shape[0]
        # Calculate the sample size
        sample_size = int(N * ratio)
        # Generate random indices for sampling
        indices = torch.randperm(N)[:sample_size]
        # Sample nodes
        sampled = nodes[indices]
        
        return sampled, indices
    
    def sample_node_all(self, nodes, ratio):
        all_sampled_nodes = []
        all_batches_indices = []
        
        # Get number of nodes N
        N = nodes.shape[0]
        
        # Calculate the sample size for each batch
        sample_size = int(N * ratio)
        
        # Shuffle all node indices
        shuffled_indices = torch.randperm(N)
        
        # Calculate the number of batches
        num_batches = N // sample_size
        
        for i in range(num_batches):
            # Get the current batch indices
            batch_indices = shuffled_indices[i * sample_size:(i + 1) * sample_size]
            
            # Sample nodes using the batch indices
            sampled = nodes[batch_indices]
            
            # Store sampled nodes and indices
            all_sampled_nodes.append(sampled)
            all_batches_indices.append(batch_indices)
        
        return all_sampled_nodes, all_batches_indices
        
    def scale_info(self, info):
        # Scale the information data into the range [0, 1]
        info_tensor = torch.tensor([info['length'], info['width'], info['height'], info['clearance'], info['slant'], info['radius'], info['velocity'],info['re']], dtype=torch.float32)
        info_tensor = info_tensor.reshape(-1,8)
        
        info_min = torch.tensor([744.0, 269.0, 228.0, 30.0, 0.0, 80.0, 20.0, 1039189.1])
        info_max = torch.tensor([1344.0, 509.0, 348.0, 90.0, 40.0, 120.0, 60.0, 5347297.3])

        info_01 =  (info_tensor - info_min.reshape(-1,8)) / (info_max.reshape(-1,8) - info_min.reshape(-1,8))
        
        return info_01
    
    def scale_area(self, area):
        # Scale the area data into the range [0, 1]
        area_min = 1.803180e-11
        area_max = 1.746826e-05

        area_01 =  (area - area_min) / (area_max - area_min)
        
        return area_01
    
    def scale_pos(self, pos):
        # Scale the position data into the range [0, 1]
        pos_min = torch.tensor([-1.344000e+00, 9.722890e-04, 9.743084e-04]).to(pos.device)
        pos_max = torch.tensor([2.718600e-04, 2.545020e-01, 4.305006e-01]).to(pos.device)
        
        node_pos = (pos - pos_min.reshape(-1,3)) / (pos_max.reshape(-1,3) - pos_min.reshape(-1,3))
       
        return node_pos

    def __getitem__(self, item):
        # Get the input data for the current item
        input = self.get_singel(item)
        
        return input