import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot
from default_program.class_method import create_gif

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

mpaping_2d = "/Users/kai/大学/小川研/LiDAR_step_length/20241113/pcd_01s/2d/mapping_1113"
mapping_3d = "/Users/kai/大学/小川研/LiDAR_step_length/20241113/pcd_01s/3d/mapping_1113"

pcd_information_2d = get_pcd_information.get_pcd_information()
pcd_information_2d.load_pcd_dir(mpaping_2d)
pcd_information_3d = get_pcd_information.get_pcd_information()
pcd_information_3d.load_pcd_dir(mapping_3d)

all_points_2d = None
for cloud in pcd_information_2d.cloud_list:
    if all_points_2d is None:
        all_points_2d = np.array(cloud)
    else:
        all_points_2d = np.vstack((all_points_2d, np.array(cloud)))

all_points_3d = None
for cloud in pcd_information_3d.cloud_list:
    if all_points_3d is None:
        all_points_3d = np.array(cloud)
    else:
        all_points_3d = np.vstack((all_points_3d, np.array(cloud)))

set_ax = plot.set_plot()
set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(all_points_2d[:, 0].min(), all_points_2d[:, 0].max()), ylim=(all_points_2d[:, 1].min(), all_points_2d[:, 1].max()), zlim=(all_points_2d[:, 2].min(), all_points_2d[:, 2].max()), azim=150)

# 補正値
x = 110
y = 100

plt.figure()
ax = plt.subplot(211)
ax.scatter(all_points_2d[:, 0]+x, all_points_2d[:, 1]+y, c="b", s=1)
all_points_3d_2 = all_points_3d[(all_points_3d[:, 2] > 42) & (all_points_3d[:, 2] < 48)]
ax.scatter(all_points_3d_2[:, 0], all_points_3d_2[:, 1], c="r", s=1)

ax = set_ax.set_ax(ax, xlim=[3000, 10000])

plt.show()
plt.close()
