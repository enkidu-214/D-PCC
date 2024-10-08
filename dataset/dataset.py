import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pickle as pkl
import random




class CompressDataset(Dataset):
    def __init__(self, data_path, map_size=100, cube_size=5, batch_size=1, points_num=1024):
        self.data_path = data_path
        with open(self.data_path, 'rb')as f:
            self.data = pkl.load(f)
        self.init_data()

        self.map_size = map_size
        self.cube_size = cube_size
        self.dim_cube_num = np.ceil(map_size / cube_size).astype(int)
        self.batch_size = batch_size
        # the points_num should be larger than or equal to the min_num of each cube
        self.points_num = points_num

    #问题，cude_idx对应的是什么，对应的是之前生成的cube的序号，一个points中有很多cube
    def init_data(self):
        # data :{pcd_idx: {'points': {cube_idx: ...}, 'meta_data':{'shift':..., 'min_points':..., 'max_points':...}}}
        self.patch_num = []
        for pcd_idx in self.data.keys():
            cur_patch_num = len(self.data[pcd_idx]['points'].keys())
            self.patch_num.append(cur_patch_num)
        # the last patch num of each full point cloud
        #看起来只是获取了所有的点云的cube个数，把每个cube当成一个patch
        self.pcd_last_patch_num = np.cumsum(self.patch_num)

    #这里还是挺困惑的，为什么能拿到idx
    def get_pcd_and_patch(self, idx):
        diff = idx + 1 - self.pcd_last_patch_num

        pcd_idx = np.where(diff <= 0)[0][0]
        if pcd_idx > 0:
            patch_idx = idx - self.pcd_last_patch_num[pcd_idx-1]
        else:
            # the first pcd
            patch_idx = idx

        return pcd_idx, patch_idx


    def __getitem__(self, idx):
        pcd_idx, patch_idx = self.get_pcd_and_patch(idx)
        cubes = list(self.data[pcd_idx]['points'].keys())
        # indicate which cube
        cube_x = cubes[patch_idx] // self.dim_cube_num ** 2
        cube_y  = (cubes[patch_idx] - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
        cube_z = cubes[patch_idx] % self.dim_cube_num
        # the coordinate of center point
        center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
        xyzs = self.data[pcd_idx]['points'][cubes[patch_idx]][:, :3]
        # 成功获取intensity
        intensity = self.data[pcd_idx]['points'][cubes[patch_idx]][:, 3:]
        # normalize to [-1, 1]
        xyzs = 2 * (xyzs - center) / self.cube_size
        xyzs = torch.tensor(xyzs).float()
        intensity = torch.tensor(intensity).float()
        input_dict = {}
        # 这边sample没看懂
        if self.batch_size == 1:
            input_dict['xyzs'] = xyzs
            input_dict['normals'] = intensity
        else:
            sample_idx = random.sample(range(xyzs.shape[0]), self.points_num)
            sample_idx = torch.tensor(sample_idx).long()
            input_dict['xyzs'] = xyzs[sample_idx, :]
            input_dict['normals'] = intensity[sample_idx, :]

        return input_dict


    def __len__(self):
        return sum(self.patch_num)


    # scale to original size
    # TODO可以在此处加对intensity的操作
    def scale_to_origin(self, xyzs, idx):
        pcd_idx, patch_idx = self.get_pcd_and_patch(idx)
        cubes = list(self.data[pcd_idx]['points'].keys())
        # indicate which cube
        cube_x = cubes[patch_idx] // self.dim_cube_num ** 2
        cube_y  = (cubes[patch_idx] - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
        cube_z = cubes[patch_idx] % self.dim_cube_num
        # the coordinate of center point
        center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
        center = torch.tensor(center).float().to(xyzs.device)
        # scale to the cube coordinate
        xyzs = xyzs * self.cube_size / 2 + center
        # scale to the original coordinate
        xyzs = xyzs / 100
        meta_data = self.data[pcd_idx]['meta_data']
        shift, max_coord, min_coord = meta_data['shift'], meta_data['max_coord'], meta_data['min_coord']
        xyzs = xyzs * (max_coord - min_coord) + min_coord + shift

        return xyzs
    
    def attr_to_origin(self, attribute, idx):
        pcd_idx, _ = self.get_pcd_and_patch(idx)
        meta_data = self.data[pcd_idx]['meta_data']
        max_attr, min_attr = meta_data["max_attr"],meta_data["min_attr"]
        attribute = attribute*(max_attr-min_attr) + min_attr
        attribute = torch.exp(attribute)
        
        return attribute

