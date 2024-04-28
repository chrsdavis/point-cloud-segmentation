import os
import numpy as np

root = 'Rellis-3D'
pcl_dir = 'os1_cloud_node_kitti_bin'
lab_dir = 'os1_cloud_node_semantickitti_label_id'
pcl_files = []
lab_files = []

for obj in os.listdir(root):
    scene = os.path.join(root, obj)
    print(f'Parsing dir: {scene}')
    if os.path.isdir(scene): # scene N
        for dir in os.listdir(scene):
            
            if dir == pcl_dir:
                path = os.path.join(scene, dir)
                for file in os.listdir(path):
                    if file.endswith('.bin'):
                        pcl_files.append(os.path.join(path, file))
            elif dir == lab_dir:
                path = os.path.join(scene, dir)
                for file in os.listdir(path):
                    if file.endswith('.label'):
                        lab_files.append(os.path.join(path, file))

pcl_files.sort()
lab_files.sort()

print('Writing to config/data/lab.lst')
with open('config/data/lab.lst', 'w') as file:
    for f in lab_files:
        file.write(f + '\n')

print('Writing to config/data/pcl.lst')
with open('config/data/pcl.lst', 'w') as file:
    for f in pcl_files:
        file.write(f + '\n')

n = len(pcl_files)

indices = np.arange(n)
np.random.shuffle(indices)

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

train_end = int(train_ratio * n)
validation_end = train_end + int(validation_ratio * n)

train_indices = indices[:train_end]
val_indices = indices[train_end:validation_end]
test_indices = indices[validation_end:]

rel_pcl_files = [os.path.relpath(f, root) for f in pcl_files]
rel_lab_files = [os.path.relpath(f, root) for f in lab_files]

print('Writing to config/train.lst')
with open('config/train.lst', 'w') as file:
    for i in train_indices:
        file.write(f'{rel_pcl_files[i]}\t{rel_lab_files[i]}\n')

print('Writing to config/val.lst')
with open('config/val.lst', 'w') as file:
    for i in val_indices:
        file.write(f'{rel_pcl_files[i]}\t{rel_lab_files[i]}\n')

print('Writing to config/test.lst')
with open('config/test.lst', 'w') as file:
    for i in test_indices:
        file.write(f'{rel_pcl_files[i]}\t{rel_lab_files[i]}\n')
