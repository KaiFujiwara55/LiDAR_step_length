import os
import sys
import numpy as np
import pcl
import glob
import time
import matplotlib.pyplot as plt

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import get_pcd_information
from default_program.class_method import plot
from default_program.class_method import create_gif

sec_list = ["0025"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241120/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/2d/*")
    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if "nothing" in dir or "mapping" in dir:
            continue

        if "far" in dir:
            continue

        if "tajima_back_1113" in dir:
            continue

        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")


        for time_idx, cloud in enumerate(pcd_info_list.cloud_list):
            points = np.array(cloud)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
            ax1 = ax_set.set_ax(ax1, title=title)
            ax1.scatter(points[:, 0], points[:, 1], s=1, c="b")
            plt.show()
            plt.close()

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

        # noize_pointsとの距離が50mm以上の点を残す
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

        noize_cloud, noize_points = get_noize_points("/Users/kai/大学/小川研/LiDAR_step_length/20241120/pcd_01s/2d/nothing_1120")
        for time_idx, cloud in enumerate(pcd_info_list.cloud_list):
            # points = np.array(cloud)
            # fig = plt.figure()
            # ax1 = fig.add_subplot(111)
            # title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
            # ax1 = ax_set.set_ax(ax1, title=title)
            # ax1.scatter(points[:, 0], points[:, 1], s=1, c="b")
            # ax1.scatter(noize_points[:, 0], noize_points[:, 1], s=1, c="r")
            # plt.show()
            # plt.close()

            new_cloud, new_points = remove_noize(cloud, noize_points)

            # fig = plt.figure()
            # ax1 = fig.add_subplot(111)
            # title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
            # ax1 = ax_set.set_ax(ax1, title=title)
            # ax1.scatter(new_points[:, 0], new_points[:, 1], s=1, c="b")
            # plt.show()
            # plt.close()

            new_cloud = def_method.get_cloud(new_points)
            new_cloud = def_method.statistical_outlier_removal(new_cloud)

            area_points_list, area_center_point_list = ori_method.get_neighborhood_points(new_cloud, radius=1000, count_threshold=5)
            
            color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
            ax = ax_set.set_ax(ax, title=title)
            for group_idx in range(len(area_center_point_list)):
                ax.scatter(area_points_list[group_idx][:, 0], area_points_list[group_idx][:, 1], s=1, c=color_list[group_idx])
            plt.show()
            plt.close()

