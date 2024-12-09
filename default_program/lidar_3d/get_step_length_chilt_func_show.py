import os
import sys
import numpy as np
import pcl
import glob
import time
import scipy.stats
import matplotlib.pyplot as plt

sys.path.append("/Users/kai/大学/小川研/LiDAR_step_length/")
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import get_pcd_information
from default_program.class_method import plot
from default_program.class_method import create_gif

# ノイズ除去のクラスをインスタンス化
def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

# plotの設定
set_ax = plot.set_plot()

def get_step(sec_1, sec_2, dir, plt_flg=False):
    pcd_info_list = get_pcd_information.get_pcd_information()
    pcd_info_list.load_pcd_dir(dir)

    set_ax.set_ax_info(pcd_info_list.dir_name, xlabel="X(mm)", ylabel="Y(mm)", zlabel="Z(mm)", xlim=[pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]], ylim=[pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]], zlim=[pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]])

    # 処理結果を読み込み
    area_path = "/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/"+str(sec_1).replace(".", "")+"s/"+pcd_info_list.dir_name
    center_path = "/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/"+str(sec_1).replace(".", "")+"s/"+pcd_info_list.dir_name
    time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

    # 点群をグループ化
    integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5, distance_threshold=500)
    # 中心点の軌跡から新たにグループを作成
    cloud_folder_path = dir.replace(str(sec_1).replace(".", "")+"s", str(sec_2).replace(".", "")+"s")
    integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, is_incline=False)
    height_list = ori_method.get_height_all(integraded_area_points_list, top_percent=0.1)
    collect_theta_z_list = ori_method.get_collect_theta_z(integraded_area_center_point_list)

    peak_list_list = []
    step_length_list_list = []
    move_flg_list = ori_method.judge_move(integraded_area_center_point_list)
    for group_idx in range(len(move_flg_list)):
        height = height_list[group_idx]
        if move_flg_list[group_idx]:
            chilt_list = []
            for time_idx in range(len(integraded_area_points_list[group_idx])):
                points = integraded_area_points_list[group_idx][time_idx]
                center_point = integraded_area_center_point_list[group_idx][time_idx]
                if len(center_point)==0:
                    continue

                points = def_method.rotate_points(points, theta_z=collect_theta_z_list[group_idx])
                center_point = def_method.rotate_points(center_point, theta_z=collect_theta_z_list[group_idx])

                bench_points = points[(points[:, 2]>height*0.5) & (points[:, 2]<height-30)]
                normarized_points = ori_method.normalization_points(bench_points, center_point)
                
                if len(normarized_points)==0:
                    continue

                # 体の軸の近似直線を取得
                cofficient = np.polyfit(normarized_points[:, 0], normarized_points[:, 1], 1)
                x = np.linspace(-250, 250, 100)
                y = cofficient[0]*x + cofficient[1]
                chilt_list.append([time_idx, cofficient[0]])

            # chilt_listを平均0で正規化
            standard_chilt_list = np.array(chilt_list)
            standard_chilt_list[:, 1] = scipy.stats.zscore(standard_chilt_list[:, 1])

            # 傾きが0で切り替わるタイミングをピークとする
            peak_list = []
            for idx in range(1, len(standard_chilt_list)):
                before_time_idx = int(standard_chilt_list[idx-1, 0])
                before_chilt = standard_chilt_list[idx-1, 1]
                after_time_idx = int(standard_chilt_list[idx, 0])
                after_chilt = standard_chilt_list[idx, 1]

                if before_chilt*after_chilt < 0:
                    # 細かいtime_idxを取得
                    peak_time_idx = -1*((after_time_idx*before_chilt - before_time_idx*after_chilt)/(after_chilt - before_chilt))

                    if len(peak_list)==0:
                        before_point = integraded_area_center_point_list[group_idx][before_time_idx]
                        after_point = integraded_area_center_point_list[group_idx][after_time_idx]
                        point = before_point + (after_point-before_point)/((peak_time_idx-before_time_idx)/(after_time_idx-before_time_idx))
                        peak_list.append([peak_time_idx, point])
                    else:
                        before_point = integraded_area_center_point_list[group_idx][before_time_idx]
                        after_point = integraded_area_center_point_list[group_idx][after_time_idx]
                        point = before_point + (after_point-before_point)*((peak_time_idx-before_time_idx)/(after_time_idx-before_time_idx))
                        distance = ori_method.calc_points_distance(peak_list[-1][1], point)
                        if distance > 400:
                            peak_list.append([peak_time_idx, point])
            
            if plt_flg:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax = set_ax.set_ax(ax, title=pcd_info_list.dir_name+"_chilt",xlabel="time (ms)", ylabel="chilt", xlim=[0, np.max(standard_chilt_list)*25], ylim=[np.min(standard_chilt_list[:, 1]), np.max(standard_chilt_list[:, 1])], is_box_aspect=False)
                ax.plot(standard_chilt_list[:, 0]*25, standard_chilt_list[:, 1])
                for peak in peak_list:
                    ax.scatter(peak[0]*25, 0, c="r", s=5)
                plt.show()
                plt.close()

            # 歩幅を取得
            step_length_list = []
            for idx in range(1, len(peak_list)):
                step_length = ori_method.calc_points_distance(peak_list[idx][1], peak_list[idx-1][1])
                step_length_list.append(step_length)

            peak_list_list.append(peak_list)
            step_length_list_list.append(step_length_list)

    return peak_list_list, step_length_list_list

