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

# nothingと比較して、ごみ取りを行う
def get_noize_points(noize_folder):
    pcd_info_list = get_pcd_information.get_pcd_information()
    pcd_info_list.load_pcd_dir(noize_folder)

    noize_points = None
    for cloud in pcd_info_list.cloud_list:
        if noize_points is None:
            noize_points = np.array(cloud)
        else:
            noize_points = np.vstack((noize_points, np.array(cloud)))

    noize_cloud = def_method.get_cloud(noize_points)
    noize_cloud = def_method.statistical_outlier_removal(noize_cloud)
    noize_cloud = def_method.voxel_grid_filter(noize_cloud, leaf_size=(5, 5, 5))

    noize_points = def_method.get_points(noize_cloud)
    
    return noize_cloud, noize_points

# noize_pointsとの距離がthreshold mm以上の点を残す
def remove_noize(cloud, noize_points, threshold=300):
    points = np.array(cloud)

    new_points = None
    for point in points:
        # noize_pointsとの距離を計算
        distances = np.linalg.norm(noize_points - point, axis=1)
        if np.min(distances) > threshold:
            if new_points is None:
                new_points = point
            else:
                new_points = np.vstack((new_points, point))

    if new_points is None:
        return None, None
    else:
        new_cloud = def_method.get_cloud(new_points)
        return new_cloud, new_points

# ごみ取りを行う
sec_list = ["01", "005", "0025"]
sec_list = ["0025"]
noize_folder = "/Users/kai/大学/小川研/LiDAR_step_length/20241120/pcd_01s/2d/nothing_1120"

noize_cloud, noize_points = get_noize_points(noize_folder)

# pcd_info_list = get_pcd_information.get_pcd_information()
# pcd_info_list.load_pcd_dir(noize_folder)
# set_ax = plot.set_plot()
# set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax = set_ax.set_ax(ax, title="noize")
# ax.scatter(np.array(noize_cloud)[:, 0], np.array(noize_cloud)[:, 1], s=1)
# plt.show()
# plt.close()

for sec in sec_list:
    active_folder_list = glob.glob(f"/Users/kai/大学/小川研/LiDAR_step_length/20241120/pcd_{sec}s/2d/*")
    for active_folder in active_folder_list:
        if noize_folder.split("/")[-1] in active_folder:
            continue
        if "far" in active_folder:
            continue
        print(active_folder)
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(active_folder)

        set_ax = plot.set_plot()
        set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        time_area_points_list = []
        time_area_center_point_list = []
        for cloud in tqdm(pcd_info_list.cloud_list):
            new_cloud, new_points = remove_noize(cloud, noize_points)
            new_cloud = def_method.statistical_outlier_removal(new_cloud, mean_k=3, std_dev_mul_thresh=1.0)

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1 = set_ax.set_ax(ax1, title="before")
            ax2 = set_ax.set_ax(ax2, title="after")

            ax1.scatter(np.array(cloud)[:, 0], np.array(cloud)[:, 1], s=1)
            ax1.scatter(noize_points[:, 0], noize_points[:, 1], s=1, c="r")
            ax2.scatter(np.array(new_cloud)[:, 0], np.array(new_cloud)[:, 1], s=1)
            
            plt.show()
            plt.close()
            continue

            if new_cloud is None:
                time_area_points_list.append([])
                time_area_center_point_list.append([])
            else:
                new_cloud = def_method.get_cloud(new_points)
                new_cloud = def_method.statistical_outlier_removal(new_cloud)

                area_points_list, area_center_point_list = ori_method.get_neighborhood_points(new_cloud, radius=250, count_threshold=10)
                
                time_area_points_list.append(area_points_list)
                time_area_center_point_list.append(area_center_point_list)

        # 処理結果を保存
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_list.dir_name}"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_list.dir_name}"
        ori_method.save_original_data(time_area_points_list, time_area_center_point_list, area_path, center_path)
