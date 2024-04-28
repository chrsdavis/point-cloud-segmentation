import numpy as np
import os
from util.pointcloud import PointCloud
import torch
from torch.utils.data import Dataset

'''
element vertex
property float x
property float y
property float z
property float intensity
property uint t
property ushort reflectivity
property uchar ring
property ushort noise
property uint range
property uchar label
property uchar red
property uchar green
property uchar blue
'''

def get_filenames(root):
    pcl = 'os1_cloud_node_kitti_bin'
    lab = 'os1_cloud_node_semantickitti_label_id'
    pcl_files = []
    label_files = []
    for obj in os.listdir(root):
        scene = os.path.join(root, obj)
        if os.path.isdir(scene): # scene N
            lab_dir = os.path.join(scene, lab)
            pcl_dir = os.path.join(scene, pcl)
            for l in os.listdir(lab_dir):
                label_files.append(os.path.join(lab_dir, l))
            for p in os.listdir(pcl_dir):
                pcl_files.append(os.path.join(pcl_dir, p))

    return pcl_files, label_files
            
            

class PCLDataSet(Dataset):
    def __init__(self, 
                 data_dir, 
                 indices: np.ndarray, # 1d array of section indices
                 label_map: dict, # class -> label
                 learning_map: dict, # combine similar classes
                 learning_map_inv: dict, 
                 sensor_config):

        self.data_dir = data_dir
        self.indices = indices
        self.label_map = label_map
        self.lmap = learning_map
        self.lmap_inv = learning_map_inv
        self.sensor_cfg = sensor_config
        self.H = sensor_config['height']
        self.W = sensor_config['width']
        self.fov_up = sensor_config['fov_up']
        self.fov_down = sensor_config['fov_down']
        self.nclasses = len(self.lmap_inv) # 20
        
        self.pcl_files, self.label_files = get_filenames(self.data_dir)
        self.pcl_files.sort()
        self.label_files.sort()


    def __len__(self):
        return len(self.pcl_files)
    
    def __getitem__(self, idx):
        pcl_file = self.pcl_files[idx]
        lab_file = self.lab_files[idx]
        
        pcl = PointCloud(H=self.H,
                         W=self.W,
                         fov_up=self.fov_up,
                         fov_down=self.fov_down)

        pcl.load(pcl_file)
        pcl.load_labels(lab_file)
        pcl.make_panorama()
        
        u = pcl.u
        v = pcl.v
        
        proj_labels = torch.from_numpy(pcl.proj_labels) # H x W
        
        proj_xyz = torch.from_numpy(pcl.proj_xyz).clone() # H x W x 3
        proj_rad = torch.from_numpy(pcl.proj_rad).clone() # H x W
        proj_ref = torch.from_numpy(pcl.proj_ref).clone() # H x W
        
        # want a 5xHxW tensor, so need to reshape
        proj_xyz_adj = proj_xyz.permute(2,0,1) # 3 x H x W
        proj_rad_adj = proj_rad.unsqueeze(0)   # 1 x H x W
        proj_ref_adj = proj_ref.unsqueeze(0)   # 1 x H x W
        
        img = torch.cat([proj_rad_adj, proj_xyz_adj, proj_ref_adj])
        
        # TODO: normalize img; img /= stdev, etc.
        
        return img, proj_labels, u, v
        
        