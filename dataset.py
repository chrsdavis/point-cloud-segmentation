import torch
from torch.utils.data import Dataset
from plyfile import PlyData
import numpy as np
import os

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

class PCloudDataSet(Dataset):
    def __init__(self, data_dir):
        self.data = []
        
        for file in in os.listdir(data_dir):
            if file.endswith('.ply'):
                file_path = os.path.join(data_dir, file)
                ply_data = PlyData.read(file_path)
                vtcs = ply_data['vertex']
                data = np.array(list(zip(vtcs['x'], vtcs['y'], vtcs['z'],
                                         vtcs['intensity'], vtcs['t'],
                                         vtcs['reflectivity'], vtcs['ring'],
                                         vtcs['noise'], vtcs['range'],
                                         vtcs['label'], vtcs['red'],
                                         vtcs['green'], vtcs['blue'])),
                                dtype=[
                                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('intensity', 'f4'), ('t', 'u4'),
                                    ('reflectivity', 'u2'), ('ring', 'u1'),
                                    ('noise', 'u2'), ('range', 'u4'),
                                    ('label', 'u1'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                                ])
                self.data.append(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item[['x', 'y', 'z', 'intensity', 'reflectivity', 
                                      'ring', 'noise', 'range', 'red', 'green', 'blue']].tolist(),
                                dtype=torch.float32)
        targets = torch.tensor(item['label'], dtype=torch.long)
        return features, targets
                