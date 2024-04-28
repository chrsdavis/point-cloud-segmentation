import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloud:
    def __init__(self, H=64, W=1024, fov_up=22.5, fov_down=-22.5):
        self.H = H
        self.W = W
        self.proj_ref = np.full((H, W), -1, dtype=np.float32)
        self.proj_rad = np.full((H, W), -1, dtype=np.float32)
        self.proj_xyz = np.full((H, W, 3), -1, dtype=np.float32)

        self.fov_up = fov_up
        self.fov_down = fov_down
        self.fov = np.abs(fov_up) + np.abs(fov_down)
        self.cloud = None
        self.labels = None
        self.proj_labels = np.zeros((H, W), dtype=np.int32)
    
    def __len__(self):
        return self.points.shape[0]
    
    def load(self, file):
        self.cloud = np.fromfile(file, dtype=np.float32)
        self.cloud = self.cloud.reshape((-1, 4))
        self.xyz = self.cloud[:, :3]
        self.x = self.cloud[:, 0]
        self.y = self.cloud[:, 1]
        self.z = self.cloud[:, 2]
        self.ref = self.cloud[:, 3:]
        # TODO: drop and randomly transform points

    def load_labels(self, file):
        labels = np.fromfile(file, dtype=np.int32)
        labels = labels.reshape((-1))
        sem_labels = labels & 0xFFFF # extract sem label (other part is instance id)
        self.labels = sem_labels
    
    def make_panorama(self):
        if self.cloud is None:
            print('Missing point cloud')
            return
        
        # convert to spherical coords
        self.rad = np.linalg.norm(self.xyz, 2, axis=1) # radius of points from origin
        self.rad[self.rad == 0] = 1e-4 # adjust origin to make warping work

        yaw = -np.arctan2(self.y, self.x) # phi
        pitch = np.arcsin(self.z / self.rad) # theta

        # (rad, yaw, pitch)
        # now map to 2d image of yaw x pitch
        # yaw goes [-pi, pi] and pitch goes [fov_down, fov_up]
        proj_x_unit = 0.5 * (yaw / np.pi + 1.0) # scaled to [0,1]
        proj_y_unit = 1.0 - (pitch + np.abs(self.fov_down)) / self.fov # scaled to [0,1]
        proj_x = proj_x_unit * self.W # [0, W]
        proj_y = proj_y_unit * self.H # [0, H]

        # convert to indices to fit in panoramic view
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)
        
        self.u = proj_x.copy()
        self.v = proj_y.copy()

        self.proj_xyz[self.v, self.u] = self.xyz
        self.proj_rad[self.v, self.u] = self.rad
        self.proj_ref[self.v, self.u] = self.ref
        
        if self.labels is not None:
            self.proj_labels[self.v, self.u] = self.labels
        
        
    
    