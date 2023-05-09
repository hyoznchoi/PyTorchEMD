import open3d as o3d
import torch, gc
import numpy as np
import time
from emd import earth_mover_distance


# load sample ply
original = o3d.io.read_point_cloud("/home/hyojinchoi/plys/color2depth_raw.ply")
distorted = o3d.io.read_point_cloud("/home/hyojinchoi/plys/color2depth_cl10.ply")

# convert Open3D.o3d.geometry.PointCloud to numpy array
# original_np = np.array(original.points, dtype=np.float32)  * (10^3)
# distorted_np = np.array(distorted.points, dtype=np.float32) * (10^3)

original_np = (np.array(original.points)).astype(np.float32)
distorted_np = (np.array(distorted.points)).astype(np.float32)


# print('original_np')
# print(original_np)
# print("distorted_np")
# print(distorted_np)

# print("\n")

# numpy to tensor
p1 = torch.from_numpy(original_np).cuda()
p2 = torch.from_numpy(distorted_np).cuda()
p1.requires_grad = True
p2.requires_grad = True

# print("<<<<p1>>>>")
# print(p1)
# print("<<<<p2>>>>")
# print(p2)

# print("\n")

# divide into 20 chunks
p1_chunks = p1.chunk(20, dim=0)
p2_chunks = p2.chunk(20, dim=0)
loss = []


# emd
with torch.no_grad():
    for i, j in zip(p1_chunks, p2_chunks):
        i = i.repeat(3, 1, 1)
        j = j.repeat(3, 1, 1)
        i.requires_grad = True
        j.requires_grad = True
        d = earth_mover_distance(i, j, transpose=False)
        print(d)
        loss.append(d[0]/10 + d[1]/10 + d[2]/10)
    print("-----------------loss-----------------")
    print(sum(loss))
