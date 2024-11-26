import os
import numpy as np
import pcl
import plot
import glob
import time
import matplotlib.pyplot as plt
import default_method
import original_method
import get_pcd_information
import create_gif

sec_list = ["01"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241028/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/*")
    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if "repeat" not in dir or "low" not in dir:
            continue

        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")
        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_points_list/{pcd_info_list.dir_name}_{sec}s"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_center_point_list/{pcd_info_list.dir_name}_{sec}s"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

        # 点群をグループ化
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5)

        # 中心点の軌跡から新たにグループを作成
        sec_2 = 0.025
        cloud_folder_path = dir_path+"pcd_"+str(sec_2).replace(".", "")+"s/"+pcd_info_list.dir_name
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, is_incline=False)

        vectros_list = ori_method.get_vector(integraded_area_center_point_list)
        move_flg_list = ori_method.judge_move(vectros_list)


        # 各時刻の点群を表示、身長データをまとめる
        color_list = ["b", "g", "c", "m", "y", "k"]*10
        heights_lists = {}
        centers = {}
        gif = create_gif.create_gif(create_flg=True)
        for group_idx in range(len(integraded_area_points_list)):
            if move_flg_list[group_idx]:
                for time_idx in range(len(integraded_area_points_list[0])):
                    group_point = integraded_area_points_list[group_idx][time_idx]
                    group_center_point = integraded_area_center_point_list[group_idx][time_idx]
                    new_group_point = integraded_area_points_list[0][time_idx]
                    new_group_center_point = integraded_area_center_point_list[0][time_idx]

                    if len(group_center_point)>0:
                        tmp_point = group_center_point.copy()
                        tmp_point[2] = 0
                        normalized_points = group_point - tmp_point
                        tmp_point = group_point.copy()
                        tmp_point = tmp_point[(tmp_point[:, 2] > 1200) & (tmp_point[:, 2] < 1400)]
                        height = ori_method.get_height(group_point, 40)
                        height = ori_method.get_bentchmark(tmp_point, percent=[0, 100])[2]

                        new_tmp_point = new_group_center_point.copy()
                        new_tmp_point[2] = 0
                        new_normalized_points = new_group_point - new_tmp_point
                        new_tmp_point = new_group_point.copy()
                        new_tmp_point = new_tmp_point[(new_tmp_point[:, 2] > 1200) & (new_tmp_point[:, 2] < 1400)]
                        new_height = ori_method.get_bentchmark(new_tmp_point, percent=[0, 100])[2]

                        fig = plt.figure(figsize=(10, 10))
                        ax0 = fig.add_subplot(121, projection='3d')
                        ax1 = fig.add_subplot(122, projection='3d')
                        # ax2 = fig.add_subplot(224, projection='3d')
                        ax0 = ax_set.set_ax(ax0, title=pcd_info_list.dir_name+" time_idx:"+str(time_idx), xlim=[0, 15000], zlim=[-100, 1900], azim=240, elev=90)
                        ax1 = ax_set.set_ax(ax1, title="point_num:"+str(len(normalized_points)), xlim=[-250, 250], ylim=[-250, 250], zlim=[0, 2000])
                        # ax2 = ax_set.set_ax(ax2, title="point_num:"+str(len(new_normalized_points)), xlim=[-250, 250], ylim=[-250, 250], zlim=[0, 2000])
                        # 10%毎に高さの平均を取得
                        heights = []
                        new_heights = []
                        for i in range(0, 100, 10):
                            heights.append(ori_method.get_bentchmark(group_point, percent=[i, i+10])[2])
                            new_heights.append(ori_method.get_bentchmark(new_group_point, percent=[i, i+10])[2])

                        ax0.scatter(np.array(group_point)[:, 0], np.array(group_point)[:, 1], np.array(group_point)[:, 2], s=1, c="r")
                        
                        if group_idx not in centers.keys():
                            centers[group_idx] = [group_center_point]
                        else:
                            centers[group_idx].append(group_center_point)
                        for key, value in centers.items():
                            ax0.scatter(np.array(value)[:, 0], np.array(value)[:, 1], np.array(value)[:, 2], s=10, c=color_list[key])

                        ax1.scatter(np.array(normalized_points)[:, 0], np.array(normalized_points)[:, 1], np.array(normalized_points)[:, 2], s=1, c=color_list[group_idx])
                        ax1.scatter(np.zeros(10), np.zeros(10), np.array(heights), s=10, c="r")

                        gif.save_fig(fig)
                        plt.show()
                    plt.close()
        gif.create_gif(f"/Users/kai/大学/小川研/LIDAR_step_length/gif/mid/{pcd_info_list.dir_name}_{sec}s_samplig{sec_2}.gif", 0.025)
        