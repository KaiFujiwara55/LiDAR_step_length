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

sec_list = ["0025"]
for sec in sec_list:
    sec_2 = 0.1
    # データの読み込み
    dir_list = glob.glob(f"/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_{sec}s/2d/*")
    for dir in dir_list:
        if "far" in dir or "nothing" in dir or "mapping" in dir:
            continue
        print(dir)
        pcd_info_2d = get_pcd_information.get_pcd_information()
        pcd_info_2d.load_pcd_dir(dir)
        # pcd_info_3d = get_pcd_information.get_pcd_information()
        # pcd_info_3d.load_pcd_dir(dir.replace("2d", "3d"))
        set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_2d.get_all_min()[0], pcd_info_2d.get_all_max()[0]), ylim=(pcd_info_2d.get_all_min()[1], pcd_info_2d.get_all_max()[1]), zlim=(pcd_info_2d.get_all_min()[2], pcd_info_2d.get_all_max()[2]), azim=150)

        area_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_2d.dir_name}"
        center_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_2d.dir_name}"
        time_area_points_list_2d, time_area_center_point_list_2d = ori_method.load_original_data(area_path_2d, center_path_2d)

        left_leg_list = []
        right_leg_list = []
        time_points_list = []
        for time_idx in range(len(time_area_points_list_2d)):
            time_points = None
            for group_idx in range(len(time_area_points_list_2d[time_idx])):
                if time_points is None:
                    time_points = time_area_points_list_2d[time_idx][group_idx]
                else:
                    time_points = np.concatenate([time_points, time_area_points_list_2d[time_idx][group_idx]], axis=0)
            
            time_points_list.append(time_points)
            if time_points is None:
                continue
            center_point_2 = np.mean(time_points, axis=0)

            left_leg = time_points[time_points[:, 1]>center_point_2[1]]
            right_leg = time_points[time_points[:, 1]<center_point_2[1]]

            if len(left_leg)>0:
                left_leg_list.append([time_idx, np.mean(left_leg, axis=0)])
            if len(right_leg)>0:
                right_leg_list.append([time_idx, np.mean(right_leg, axis=0)])

        left_speed_list = []
        for i in range(1, len(left_leg_list)):
            if i == 0:
                continue
            left_step = left_leg_list[i][0] - left_leg_list[i-1][0]
            left_speed = (ori_method.calc_points_distance(left_leg_list[i][1], left_leg_list[i-1][1])) / (sec_2*left_step)
            left_speed_list.append([left_leg_list[i][0], left_speed])
        
        right_speed_list = []
        for i in range(1, len(right_leg_list)):
            if i == 0:
                continue
            right_step = right_leg_list[i][0] - right_leg_list[i-1][0]
            right_speed = (ori_method.calc_points_distance(right_leg_list[i][1], right_leg_list[i-1][1])) / (sec_2*right_step)
            right_speed_list.append([right_leg_list[i][0], right_speed])

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

        # ピークの取得2 交点間で一つのピークしか持たないとする
        ######################################################################        
        # left_legとright_legの交点を取得
        cross_points = ori_method.get_cross_points(left_speed_list[:, 0], left_speed_list[:, 1], right_speed_list[:, 0], right_speed_list[:, 1])

        left_peaks = []
        right_peaks = []
        for i in range(len(cross_points)-1):
            pick_left = left_speed_list[(left_speed_list[:, 0] >= cross_points[i][0]) & (left_speed_list[:, 0] <= cross_points[i+1][0])]
            pick_left_mean = np.mean(pick_left[:, 1])
            pick_right = right_speed_list[(right_speed_list[:, 0] >= cross_points[i][0]) & (right_speed_list[:, 0] <= cross_points[i+1][0])]
            pick_right_mean = np.mean(pick_right[:, 1])

            if pick_left_mean > pick_right_mean:
                min_time_idx = int(pick_right[np.argmin(pick_right[:, 1])][0])
                min_idx = np.argwhere(right_speed_list[:, 0] == min_time_idx)[0][0]
                right_peaks.append(min_idx)
            else:
                min_time_idx = int(pick_left[np.argmin(pick_left[:, 1])][0])
                min_idx = np.argwhere(left_speed_list[:, 0] == min_time_idx)[0][0]
                left_peaks.append(min_idx)

        ######################################################################
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(left_speed_list[:, 0], left_speed_list[:, 1], label="left")
        ax.scatter(left_speed_list[:, 0][left_peaks], left_speed_list[:, 1][left_peaks], c="green", label="left_peak", s=10)

        ax.plot(right_speed_list[:, 0], right_speed_list[:, 1], label="right")
        ax.scatter(right_speed_list[:, 0][right_peaks], right_speed_list[:, 1][right_peaks], c="red", label="right_peak", s=10)

        ax.scatter(cross_points[:, 0], cross_points[:, 1], c="blue", label="cross_point", s=10)

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

            before_point = np.mean(time_points_list[before_time_idx], axis=0)
            after_point = np.mean(time_points_list[after_time_idx], axis=0)

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

        continue
        for time_idx in range(len(time_area_points_list_2d)):
            points = time_points_list[time_idx]
            if len(points) is None:
                continue
            points = np.array(points)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            title = f"{pcd_info_2d.dir_name}_{sec}s_{group_idx}_{time_idx}"
            ax = set_ax.set_ax(ax, title=title)
            ax.scatter(points[:, 0], points[:, 1], s=1)
            
            for peak_time_idx in peak_time_idx_list:
                if peak_time_idx <= time_idx:
                    ax.scatter(np.mean(time_points_list[int(peak_time_idx)], axis=0)[0], np.mean(time_points_list[int(peak_time_idx)], axis=0)[1], c="red", s=10)
            
            plt.show()
            plt.close()




