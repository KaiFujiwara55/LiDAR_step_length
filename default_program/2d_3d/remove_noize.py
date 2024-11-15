import os
import numpy as np
import pcl
import glob
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/kai/大学/小川研/LiDAR_step_length/default_program/3dlidar')
import plot
import default_method
import original_method
import get_pcd_information
import create_gif


pcd_info_list = get_pcd_information.get_pcd_information()

# nothingと比較して、ごみ取りを行う
nothing_folder = "/Users/kai/大学/小川研/LiDAR_step_length/20241113/pcd_01s/2d/mapping/"

pcd_info_list.load_pcd_dir(nothing_folder)

noize_points = None
for cloud in pcd_info_list.cloud_list:
    if noize_points is None:
        noize_points = np.array(cloud)
    else:
        noize_points = np.vstack((noize_points, np.array(cloud)))

noize_cloud = default_method.cloud_method().get_cloud(noize_points)
noize_cloud = default_method.cloud_method().statistical_outlier_removal(noize_cloud)
noize_cloud = default_method.cloud_method().voxel_grid_filter(noize_cloud, leaf_size=(5, 5, 5))

noize_points = np.array(noize_cloud)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(noize_points[:, 0], noize_points[:, 1], s=1)
plt.show()
plt.close()

# ごみ取りを行う
active_folder = "/Users/kai/大学/小川研/LiDAR_step_length/20241113/pcd_01s/2d/fujiwara_front/"
pcd_info_list_active = get_pcd_information.get_pcd_information()
pcd_info_list_active.load_pcd_dir(active_folder)

ax_set = plot.set_plot()
ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list_active.get_all_min()[0], pcd_info_list_active.get_all_max()[0]), ylim=(pcd_info_list_active.get_all_min()[1], pcd_info_list_active.get_all_max()[1]), zlim=(pcd_info_list_active.get_all_min()[2], pcd_info_list_active.get_all_max()[2]), azim=150)

for acteive_cloud in pcd_info_list_active.cloud_list:
    active_points = np.array(acteive_cloud)
    
    new_points = None
    for point in active_points:
        # noize_pointsとの距離を計算
        distances = np.linalg.norm(noize_points - point, axis=1)
        if np.min(distances) > 50:
            if new_points is None:
                new_points = point
            else:
                new_points = np.vstack((new_points, point))


    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(new_points[:, 0], new_points[:, 1], s=1)
    ax = ax_set.set_ax(ax, title="before")

    new_cloud = default_method.cloud_method().get_cloud(new_points)
    new_cloud = default_method.cloud_method().statistical_outlier_removal(new_cloud)
    new_points = np.array(new_cloud)
    ax2 = fig.add_subplot(212)
    ax2.scatter(new_points[:, 0], new_points[:, 1], s=1)
    ax2 = ax_set.set_ax(ax2, title="after_outlier_removal")

    plt.show()
    plt.close()




