import os.path
import random
from torch.utils.data import Dataset
import torch
import numpy as np
from plyfile import PlyData
from sklearn.neighbors import KDTree

def get_data(path, item, mode, if_sample):

    info_path = path + '/' + mode + "/" + "info_" + item + ".pt"
    press_path = path + '/' + mode + "/" + "press_" + item + ".npy"
    centroids_path = path + '/data/' + mode + "/" + "centroids_" + item + ".npy"
    areas_path = path + '/data/' + mode + "/" + "areas_" + item + ".npy"
    edges_path = path + '/edges/' + mode + "/" + "cell_edges_" + item + ".npy"
    dragWeight_path = path + '/dragWeight_test/' + "dragWeight_" + item + ".npy"
    
    ##################
    centroids = np.load(centroids_path).astype(np.float32) 
    areas = np.load(areas_path).astype(np.float32) 
    pressure = np.load(press_path).reshape(-1,1).astype(np.float32) 
    info = torch.load(info_path)
    ##################
    if if_sample:
        if mode == "train":
            inputs_all = {
                "centroids": centroids,
                "areas":areas,
                "pressure":pressure,
                "info": info
            }
        elif mode == "test":
            dragWeight = np.load(dragWeight_path).reshape(-1,1).astype(np.float32) 
            
            inputs_all = {
                "centroids": centroids,
                "areas":areas,
                "pressure":pressure,
                "info": info,
                "dragWeight": dragWeight
            }
    else:
        edges = np.load(edges_path).astype(np.int32)
        if mode == "train":
            inputs_all = {
                "centroids": centroids,
                "areas":areas,
                "edges": edges,
                "pressure":pressure,
                "info": info
            }
        elif mode == "test":
            
            dragWeight = np.load(dragWeight_path).reshape(-1,1).astype(np.float32) 
            inputs_all = {
                "centroids": centroids,
                "areas":areas,
                "edges": edges,
                "pressure":pressure,
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
        
        self.edges_ratio = 1/edges_ratio
        self.node_ratio = 1/node_ratio
        if self.if_sample:
            self.adj_num = 15
            
        self.if_zero_shot = if_zero_shot
        
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
        
        # Flatten the new_edges tensor and remove edges with values [-1, -1]        
        flat_edges = new_edges[(new_edges[:,:,0] != -1) & (new_edges[:,:,1] != -1)]
        # Reshape the flattened tensor to have shape [L, 2]
        flat_edges = flat_edges.view(-1, 2)
        
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
                    # nodes sampling + edges sampling
                    all_sampled_centroids, all_batches_indices = self.sample_node_all(centroids, self.node_ratio)
                    
                    all_nodes = []
                    all_edges = []
                    all_press = []
                    all_areas = []
                    all_dragWeight = []
                    
                    for index, centroids_i in enumerate(all_sampled_centroids):
                        
                        indices = all_batches_indices[index]
                        
                        # edges sampling
                        edges_ = self.knn_edges(centroids_i, self.adj_num)
                        edges = self.sample_edges_0(edges_, self.edges_ratio)
                        all_edges.append(edges)
                        
                        # nodes
                        all_nodes.append(centroids_01[indices])
                        
                        # pressure
                        all_press.append(pressure[indices])
                        
                        # areas
                        all_areas.append(areas_01[indices])
                        
                        # dragWeight
                        all_dragWeight.append(inputs_all["dragWeight"][indices])
                    
                    input = {'node_pos': all_nodes,
                             'edges': all_edges,
                             'pressure': all_press,
                             'info': info_01,
                             'areas': all_areas,
                             'dragWeight': all_dragWeight
                            }
                else:
                    # nodes sampling + edges sampling
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
                # nodes sampling + edges sampling
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

        # 构建 KD 树
        node_pos = node_pos.numpy()
        tree = KDTree(node_pos)
        # 查询每个点的最近 k 个邻居，返回距离和索引
        dists, indices = tree.query(node_pos, k=k+1)  # k+1 是因为最近的一个邻居是点本身
                
        # 构建边的索引
        num_points = node_pos.shape[0]
        i_indices = torch.arange(num_points).view(-1, 1).repeat(1, k+1)
        
        # 重新整理边数组的形状, 保留结构 [N, k, 2]
        edges = torch.stack([i_indices, torch.from_numpy(indices)], dim=-1)
        
        return edges.long()
    
    def sample_edges_0(self, edges, ratio):
        
        # 1.
        # 获取节点数 N
        N = edges.shape[0]
        k = edges.shape[1]
        # 为每个节点生成一个从 0 到 k-1 的随机索引
        random_indices = torch.randint(0, k, (N,))
        # 采样 edges
        sample_edges = edges[torch.arange(N), random_indices]
        
        if int(ratio)==1:
            return sample_edges
        else:
            # 计算采样点的数量
            sample_size = int(N * ratio)
            # 在dim=0上生成随机索引
            indices = torch.randperm(N)[:sample_size]
            # 使用随机索引采样
            sampled = sample_edges[indices]
            return sampled
    
    def sample_edges_1(self, edges, ratio):
        
        # 1.
        # 获取边数 L
        L = edges.shape[0]
        # 2. 
        # 计算采样点的数量
        sample_size = int(L * ratio)
        # 在dim=0上生成随机索引
        indices = torch.randperm(L)[:sample_size]
        # 使用随机索引采样
        sampled = edges[indices]
        
        return sampled

    def sample_node(self, nodes, ratio):
        
        # 1.
        # 获取点数 N
        N = nodes.shape[0]
        # 2. 
        # 计算采样点的数量
        sample_size = int(N * ratio)
        # 在dim=0上生成随机索引
        indices = torch.randperm(N)[:sample_size]
        # 使用随机索引采样
        sampled = nodes[indices]
        
        return sampled, indices
    
    def sample_node_all(self, nodes, ratio):
        
        all_sampled_nodes = []
        all_batches_indices = []
        
        # 获取点数 N
        N = nodes.shape[0]
        
        # 计算每个批次的样本数
        sample_size = int(N * ratio)
        
        # 随机打乱所有节点的索引
        shuffled_indices = torch.randperm(N)
        
        # 计算可以分成的批次数量
        num_batches = N // sample_size
        
        for i in range(num_batches):
            # 获取当前批次的索引范围
            batch_indices = shuffled_indices[i * sample_size:(i + 1) * sample_size]
            
            # 使用这些索引进行采样
            sampled = nodes[batch_indices]
            
            # 将采样的节点和索引分别存储到列表中
            all_sampled_nodes.append(sampled)
            all_batches_indices.append(batch_indices)
        
        return all_sampled_nodes, all_batches_indices
        
    def scale_info(self, info):
        
        # Example: {'length': 1319.0, 'width': 339.0, 'height': 323.0, 'clearance': 70.0, 'slant': 40.0, 'radius': 82.5, 'velocity': 50.0, 're': 4456081.0}
        info_tensor = torch.tensor([info['length'], info['width'], info['height'], info['clearance'], info['slant'], info['radius'], info['velocity'],info['re']], dtype=torch.float32)
        info_tensor = info_tensor.reshape(-1,8)
        
        info_min = torch.tensor([744.0, 269.0, 228.0, 30.0, 0.0, 80.0, 20.0, 1039189.1])
        info_max = torch.tensor([1344.0, 509.0, 348.0, 90.0, 40.0, 120.0, 60.0, 5347297.3])

        info_01 =  (info_tensor - info_min.reshape(-1,8)) / (info_max.reshape(-1,8) - info_min.reshape(-1,8))
        
        return info_01
    
    def scale_area(self, area):
        
        area_min = 1.803180e-11
        area_max = 1.746826e-05

        area_01 =  (area - area_min) / (area_max - area_min)
        
        return area_01
    
    def scale_pos(self, pos):

        # pos_min = torch.tensor([-1.3440, 0.0, 0.0]).to(pos.device)
        # pos_max = torch.tensor([0.0005, 0.2545, 0.4305]).to(pos.device)
        
        pos_min = torch.tensor([-1.344000e+00, 9.722890e-04, 9.743084e-04]).to(pos.device)
        pos_max = torch.tensor([2.718600e-04, 2.545020e-01, 4.305006e-01]).to(pos.device)
        
        node_pos = (pos - pos_min.reshape(-1,3)) / (pos_max.reshape(-1,3) - pos_min.reshape(-1,3))
       
        return node_pos

    def __getitem__(self, item):
        
        input = self.get_singel(item)
         
        return input