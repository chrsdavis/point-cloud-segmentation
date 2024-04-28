import os

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