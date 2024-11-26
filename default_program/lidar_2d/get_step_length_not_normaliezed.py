import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from scipy.signal import find_peaks

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot
from default_program.class_method import create_gif

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()
set_ax = plot.set_plot()

corect_x = 110
corect_y = 100

sec_list = ["01"]
for sec in sec_list:
    # データの読み込み
    dir_list = glob.glob(f"/Users/kai/大学/小川研/Lidar_step_length/20241113/pcd_{sec}s/2d/*")
    for dir in dir_list:
        if "nothing" in dir or "mapping" in dir:
            continue

        if "far" in dir:
            continue
        
        print(dir)
        pcd_info_2d = get_pcd_information.get_pcd_information()
        pcd_info_2d.load_pcd_dir(dir)
        pcd_info_3d = get_pcd_information.get_pcd_information()
        pcd_info_3d.load_pcd_dir(dir.replace("2d", "3d"))
        set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_3d.get_all_min()[0], pcd_info_3d.get_all_max()[0]), ylim=(pcd_info_3d.get_all_min()[1], pcd_info_3d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)

        area_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_2d.dir_name}"
        center_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_2d.dir_name}"
        time_area_points_list_2d, time_area_center_point_list_2d = ori_method.load_original_data(area_path_2d, center_path_2d)
        integraded_area_points_list_2d, integraded_area_center_point_list_2d = ori_method.grouping_points_list(time_area_points_list_2d, time_area_center_point_list_2d, integrade_threshold=3)
        sec_2 = 0.025
        cloud_folder_path = "/Users/kai/大学/小川研/Lidar_step_length/20241113/pcd_"+str(sec_2).replace(".", "")+"s/2d/"+pcd_info_2d.dir_name
        integraded_area_points_list_2d, integraded_area_center_point_list_2d = ori_method.grouping_points_list_2(integraded_area_points_list_2d, integraded_area_center_point_list_2d, cloud_folder_path, sec=sec_2, judge_move_threshold=500, is_incline=False)

        area_path_3d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/{sec}s/{pcd_info_3d.dir_name}"
        center_path_3d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/{sec}s/{pcd_info_3d.dir_name}"
        time_area_points_list_3d, time_area_center_point_list_3d = ori_method.load_original_data(area_path_3d, center_path_3d)
        integraded_area_points_list_3d, integraded_area_center_point_list_3d = ori_method.grouping_points_list(time_area_points_list_3d, time_area_center_point_list_3d, integrade_threshold=5)
        sec_2 = 0.1
        cloud_folder_path = "/Users/kai/大学/小川研/Lidar_step_length/20241113/pcd_"+str(sec_2).replace(".", "")+"s/3d/"+pcd_info_2d.dir_name
        integraded_area_points_list_3d, integraded_area_center_point_list_3d = ori_method.grouping_points_list_2(integraded_area_points_list_3d, integraded_area_center_point_list_3d, cloud_folder_path, sec=sec_2, is_incline=True)
        height = ori_method.get_height_all(integraded_area_center_point_list_3d)
        theta_z_list = ori_method.get_collect_theta_z(integraded_area_center_point_list_3d)
        
        for group_idx in range(len(integraded_area_points_list_2d)):
            all_points = None
            for time_idx in range(len(integraded_area_points_list_2d[group_idx])):
                points = integraded_area_points_list_2d[group_idx][time_idx]
                if len(points) == 0:
                    continue
                if all_points is None:
                    all_points = points
                else:
                    all_points = np.vstack([all_points, points])
            
            # 進行方向の直線を求める
            coefficent = np.polyfit(all_points[:, 0], all_points[:, 1], 1)
            x = np.linspace(0, 20000, 100)
            y = coefficent[0]*x + coefficent[1]

        
        left_leg_list = {}
        right_leg_list = {}
        for group_idx_2d in range(len(integraded_area_points_list_2d)):
            gif = create_gif.create_gif(False)
            if ori_method.judge_move(ori_method.get_vector(integraded_area_center_point_list_2d), threshold=1000)[group_idx_2d]:
                for time_idx in range(len(integraded_area_points_list_2d[group_idx_2d])):
                    points = np.array(integraded_area_points_list_2d[group_idx_2d][time_idx])
                    center_point = np.array(integraded_area_center_point_list_2d[group_idx_2d][time_idx])
                    if len(points) == 0:
                        continue
                    
                    # coefficientを使って領域を分ける
                    left_leg = points[points[:, 1] <= coefficent[0]*points[:, 0] + coefficent[1]]
                    right_leg = points[points[:, 1] >= coefficent[0]*points[:, 0] + coefficent[1]]

                    # fig = plt.figure()
                    # ax1 = fig.add_subplot(121)
                    # title = f"{pcd_info_2d.dir_name}_{sec}s_{group_idx_2d}_{time_idx}"
                    # ax1 = set_ax.set_ax(ax1, title=title)
                    # ax2 = fig.add_subplot(122)
                    # # ax2 = set_ax.set_ax(ax2, title=title)

                    # ax1.scatter(left_leg[:, 0]+corect_x, left_leg[:, 1]+corect_y, c="b", s=1)
                    # ax1.scatter(right_leg[:, 0]+corect_x, right_leg[:, 1]+corect_y, c="r", s=1)
                    # ax2.scatter(left_leg[:, 0], left_leg[:, 1], c="b", s=1)
                    # ax2.scatter(right_leg[:, 0], right_leg[:, 1], c="r", s=1)
                    # ax2.plot(x, y, c="g")

                    # plt.show()
                    # gif.save_fig(fig)
                    # plt.close()

                    if len(left_leg)>0:
                        if group_idx_2d not in left_leg_list.keys():
                            left_leg_list[group_idx_2d] = []
                        left_leg_list[group_idx_2d].append([time_idx, np.mean(left_leg, axis=0)])
                    if len(right_leg)>0:
                        if group_idx_2d not in right_leg_list.keys():
                            right_leg_list[group_idx_2d] = []
                        right_leg_list[group_idx_2d].append([time_idx, np.mean(right_leg, axis=0)])
                gif.create_gif(f"/Users/kai/大学/小川研/LiDAR_step_length/gif/2d/{pcd_info_2d.dir_name}_step.gif", duration=0.025)

        for group_idx in left_leg_list.keys():
            left_speed_list = []
            for i in range(1, len(left_leg_list[group_idx])):
                if i == 0:
                    continue
                left_step = left_leg_list[group_idx][i][0] - left_leg_list[group_idx][i-1][0]
                left_speed = (ori_method.calc_points_distance(left_leg_list[group_idx][i][1], left_leg_list[group_idx][i-1][1])) / (sec_2*left_step)
                left_speed_list.append([left_leg_list[group_idx][i][0], left_speed])
            
            right_speed_list = []
            for i in range(1, len(right_leg_list[group_idx])):
                if i == 0:
                    continue
                right_step = right_leg_list[group_idx][i][0] - right_leg_list[group_idx][i-1][0]
                right_speed = (ori_method.calc_points_distance(right_leg_list[group_idx][i][1], right_leg_list[group_idx][i-1][1])) / (sec_2*right_step)
                right_speed_list.append([right_leg_list[group_idx][i][0], right_speed])

            left_speed_list = np.array(left_speed_list)
            right_speed_list = np.array(right_speed_list)

            # 移動平均を取る
            window = 10
            left_speed_list[:, 1] = np.convolve(left_speed_list[:, 1], np.ones(window)/window, mode='same')
            right_speed_list[:, 1] = np.convolve(right_speed_list[:, 1], np.ones(window)/window, mode='same')

            # ピークの取得
            ######################################################################
            left_speed_mean = np.mean(left_speed_list[:, 1])
            right_speed_mean = np.mean(right_speed_list[:, 1])

            # 平均速度で反転させて、scipyを使ってピークを取得
            inverted_left_speed_list = left_speed_mean + (left_speed_mean - left_speed_list[:, 1])
            inverted_right_speed_list = right_speed_mean + (right_speed_mean - right_speed_list[:, 1])

            left_peaks, _ = find_peaks(inverted_left_speed_list, height=left_speed_mean, prominence=100)
            right_peaks, _ = find_peaks(inverted_right_speed_list, height=right_speed_mean, prominence=100)
            ######################################################################
            
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(left_speed_list[:, 0], left_speed_list[:, 1], label="left")
            ax.scatter(left_speed_list[:, 0][left_peaks], left_speed_list[:, 1][left_peaks], c="green", label="left_peak", s=10)

            ax.plot(right_speed_list[:, 0], right_speed_list[:, 1], label="right")
            ax.scatter(right_speed_list[:, 0][right_peaks], right_speed_list[:, 1][right_peaks], c="red", label="right_peak", s=10)

            plt.legend()
            plt.suptitle(f"{pcd_info_2d.dir_name}")
            plt.show()
            plt.close()


            # 歩幅の取得
            step_length_list = []

            left_time_idx = left_speed_list[:, 0][left_peaks]
            right_time_idx = right_speed_list[:, 0][right_peaks]
            peak_time_idx_list = np.sort(np.concatenate([left_time_idx, right_time_idx]))
            for i in range(1, len(peak_time_idx_list)):
                before_time_idx = int(peak_time_idx_list[i-1])
                after_time_idx = int(peak_time_idx_list[i])

                before_point = integraded_area_center_point_list_2d[0][before_time_idx]
                after_point = integraded_area_center_point_list_2d[0][after_time_idx]

                step_length = ori_method.calc_points_distance(before_point, after_point)
                
                step_length_list.append(step_length)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(step_length_list, bins=55, range=(0, 1100))
            title = f"step_hist"
            ax.set_title(title)

            fig.suptitle(f"{pcd_info_2d.dir_name}_{sec}s sampling={sec_2}s, window={window}", y=0)
            plt.show()
            plt.close()


            for time_idx in range(len(integraded_area_points_list_2d[group_idx])):
                points = integraded_area_points_list_2d[group_idx][time_idx]
                if len(points) == 0:
                    continue
                points = np.array(points)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                title = f"{pcd_info_2d.dir_name}_{sec}s_{group_idx}_{time_idx}"
                ax = set_ax.set_ax(ax, title=title)
                ax.scatter(points[:, 0], points[:, 1], s=1)
                
                for peak_time_idx in peak_time_idx_list:
                    if peak_time_idx <= time_idx:
                        ax.scatter(integraded_area_center_point_list_2d[group_idx][int(peak_time_idx)][0], integraded_area_center_point_list_2d[group_idx][int(peak_time_idx)][1], c="red", s=10)
                
                plt.show()
                plt.close()



