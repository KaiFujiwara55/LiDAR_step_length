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

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()


def get_step(sec, dir):
    pcd_info_list = get_pcd_information.get_pcd_information()
    pcd_info_list.load_pcd_dir(dir)

    area_path = "/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/"+str(sec).replace(".", "")+"s/"+pcd_info_list.dir_name
    center_path = "/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/"+str(sec).replace(".", "")+"s/"+pcd_info_list.dir_name
    time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

    left_leg_list = []
    right_leg_list = []
    time_points_list = []
    for time_idx in range(len(time_area_points_list)):
        time_points = None
        for group_idx in range(len(time_area_points_list[time_idx])):
            if time_points is None:
                time_points = time_area_points_list[time_idx][group_idx]
            else:
                time_points = np.concatenate([time_points, time_area_points_list[time_idx][group_idx]], axis=0)
        
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
        left_speed = (ori_method.calc_points_distance(left_leg_list[i][1], left_leg_list[i-1][1])) / (sec*left_step)
        left_speed_list.append([left_leg_list[i][0], left_speed])
    
    right_speed_list = []
    for i in range(1, len(right_leg_list)):
        if i == 0:
            continue
        right_step = right_leg_list[i][0] - right_leg_list[i-1][0]
        right_speed = (ori_method.calc_points_distance(right_leg_list[i][1], right_leg_list[i-1][1])) / (sec*right_step)
        right_speed_list.append([right_leg_list[i][0], right_speed])

    left_speed_list = np.array(left_speed_list)
    right_speed_list = np.array(right_speed_list)

    # 移動平均を取る
    window = 10
    left_speed_list[:, 1] = np.convolve(left_speed_list[:, 1], np.ones(window)/window, mode='same')
    right_speed_list[:, 1] = np.convolve(right_speed_list[:, 1], np.ones(window)/window, mode='same')

    # ピークの取得
    cross_points = ori_method.get_cross_points(left_speed_list[:, 0], left_speed_list[:, 1], right_speed_list[:, 0], right_speed_list[:, 1])

    left_peak_list = []
    right_peak_list = []
    for i in range(len(cross_points)-1):
        pick_left = left_speed_list[(left_speed_list[:, 0] >= cross_points[i][0]) & (left_speed_list[:, 0] <= cross_points[i+1][0])]
        pick_left_mean = np.mean(pick_left[:, 1])
        pick_right = right_speed_list[(right_speed_list[:, 0] >= cross_points[i][0]) & (right_speed_list[:, 0] <= cross_points[i+1][0])]
        pick_right_mean = np.mean(pick_right[:, 1])

        if pick_left_mean > pick_right_mean:
            min_time_idx = int(pick_right[np.argmin(pick_right[:, 1])][0])
            min_idx = np.argwhere(right_speed_list[:, 0] == min_time_idx)[0][0]
            
            right_peak_list.append([min_time_idx, right_leg_list[min_idx][1]])
        else:
            min_time_idx = int(pick_left[np.argmin(pick_left[:, 1])][0])
            min_idx = np.argwhere(left_speed_list[:, 0] == min_time_idx)[0][0]
            
            left_peak_list.append([min_time_idx, left_leg_list[min_idx][1]])

    # 歩幅の取得
    step_length_list = []
    sort_time_idx = np.argsort(np.concatenate([np.array([x[0] for x in left_peak_list]), np.array([x[0] for x in right_peak_list])]))
    peak_point_list = np.concatenate([np.array([x[1] for x in left_peak_list]), np.array([x[1] for x in right_peak_list])])[sort_time_idx]
    for i in range(1, len(peak_point_list)):
        before_point = peak_point_list[i-1]
        after_point = peak_point_list[i]

        step_length = ori_method.calc_points_distance(before_point, after_point)
        
        step_length_list.append(step_length)
    
    return left_peak_list, right_peak_list, step_length_list
