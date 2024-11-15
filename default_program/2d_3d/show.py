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

sec_list = ["01"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241113/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/2d/*")
    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()

    for dir in dirs:
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")

        if False:
            time_area_points_list = []
            time_area_center_point_list = []
            for i, cloud in enumerate(pcd_info_list.cloud_list):
                points = np.array(cloud)
                area_points_list, area_center_point_list = ori_method.get_neighborhood_points(cloud, radius=250, count_threshold=10)

                # 時系列の点群を保存
                time_area_points_list.append(area_points_list)
                time_area_center_point_list.append(area_center_point_list)

            # 処理結果を保存
            area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_points_list/2d/{pcd_info_list.dir_name}_{sec}s"
            center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_center_point_list/2d/{pcd_info_list.dir_name}_{sec}s"
            ori_method.save_original_data(time_area_points_list, time_area_center_point_list, area_path, center_path)

        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_points_list/2d/{pcd_info_list.dir_name}_{sec}s"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_center_point_list/2d/{pcd_info_list.dir_name}_{sec}s"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5)

        gif = create_gif.create_gif()
        color_list = ["b", "g", "c", "m", "y", "k"]*100
        for time_idx in range(len(time_area_points_list)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
            ax = ax_set.set_ax(ax, title=title)
            for group_idx in range(len(time_area_points_list[time_idx])):
                points = time_area_points_list[time_idx][group_idx]
                if len(points) == 0:
                    continue
                points = np.array(points)

                ax.scatter(points[:, 0], points[:, 1], c=color_list[group_idx], s=1)
            plt.show()
            gif.save_fig(fig)
            plt.close()
        gif.create_gif(f"/Users/kai/大学/小川研/LiDAR_step_length/gif/2d/{pcd_info_list.dir_name}_{sec}s.gif")
