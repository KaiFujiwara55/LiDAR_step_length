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

sec_list = ["01"]
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

        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")
        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_list.dir_name}"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_list.dir_name}"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)
        # 点群をグループ化
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=3)
        
        move_flg_list = ori_method.judge_move(integraded_area_center_point_list)
        for group_idx in range(len(integraded_area_points_list)):
            if move_flg_list[group_idx]:
                for time_idx in range(len(integraded_area_points_list[group_idx])):
                    points = integraded_area_points_list[group_idx][time_idx]
                    center_point = integraded_area_center_point_list[group_idx][time_idx]
                    if len(center_point)==0:
                        continue


                    fig = plt.figure()
                    ax1 = fig.add_subplot(111, projection="3d")
                    title = f"{pcd_info_list.dir_name}_{group_idx}_{time_idx}"
                    ax1 = ax_set.set_ax(ax1, title=title)
                    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="b")
                    plt.show()
                    plt.close()



        # # 中心点の軌跡から新たにグループを作成
        sec_2 = 0.025
        cloud_folder_path = "/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_"+str(sec_2).replace(".", "")+"s/2d/"+pcd_info_list.dir_name
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, judge_move_threshold=500, is_incline=False)

        vectros_list = ori_method.get_vector(integraded_area_center_point_list)
        move_flg_list = ori_method.judge_move(vectros_list, threshold=0)

        for group_idx in range(len(integraded_area_points_list)):
            if move_flg_list[group_idx]:
                for time_idx in range(len(integraded_area_points_list[0])):
                    points = integraded_area_points_list[group_idx][time_idx]
                    center_point = integraded_area_center_point_list[group_idx][time_idx]

                    fig = plt.figure()
                    ax1 = fig.add_subplot(111, projection="3d")
                    title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
                    ax1 = ax_set.set_ax(ax1, title=title)
                    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="b")
                    plt.show()
                    plt.close()
