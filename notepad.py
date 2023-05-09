import open3d as o3d
import torch, gc
import numpy as np
import time
# from emd import earth_mover_distance

gc.collect()
torch.cuda.empty_cache()


# load sample ply
original = o3d.io.read_point_cloud("/home/hyojinchoi/sample_ply/color2depth_raw.ply")
distorted = o3d.io.read_point_cloud("/home/hyojinchoi/sample_ply/color2depth_cl10.ply")

# convert Open3D.o3d.geometry.PointCloud to numpy array
original_np = np.asarray(original.points)
distorted_np = np.asarray(distorted.points)
print('original_np')
print(original_np)
print("distorted_np")
print(distorted_np)

# import pdb; pdb.set_trace()

# gt
p1 = torch.from_numpy(original_np).cuda()
p2 = torch.from_numpy(distorted_np).cuda()

# p1 = torch.from_numpy(original_np)
# p2 = torch.from_numpy(distorted_np)

p1.requires_grad = True
p2.requires_grad = True

import pdb; pdb.set_trace()