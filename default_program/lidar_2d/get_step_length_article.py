import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()
set_ax = plot.set_plot()

dir = "/Users/kai/大学/小川研/LiDAR_step_length/20241120/pcd_01s/2d/fujiwara_front_1120"

# 最小二乗法で円の中心を求める関数
def fit_circle_fixed_radius(points, radius):
    A = np.array([points[:, 0], points[:, 1], np.ones(len(points))]).T
    b = -1*(points[:, 0]**2 + points[:, 1]**2) - radius**2
    u = np.linalg.lstsq(A, b, rcond=None)[0]
    cx = -u[0] / 2
    cy = -u[1] / 2
    return cx, cy

def get_step(sec, dir):
    pcd_info_list = get_pcd_information.get_pcd_information()
    pcd_info_list.load_pcd_dir(dir)

    set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

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

        time_points = time_points[np.argsort(time_points[:, 1])]

        center_idx = len(time_points)//2

        left_leg = time_points[:center_idx]
        right_leg = time_points[center_idx:]

        if len(left_leg)==0 or len(right_leg)==0:
            continue
        # 外れ値除外
        left_median_point = left_leg[len(left_leg)//2]
        tmp_points = []
        for left_point in left_leg:
            if abs(left_point[0]-left_median_point[0])<100:
                tmp_points.append(left_point)
        
        left_leg_2 = np.array(tmp_points)

        right_median_point = right_leg[len(right_leg)//2]
        tmp_points = []
        for right_point in right_leg:
            if abs(right_point[0]-right_median_point[0])<100:
                tmp_points.append(right_point)
        
        right_leg_2 = np.array(tmp_points)

        # 円形でフィッティングして中心を求めるコードを書かなければいけない

        # 平均値で基準を取得
        if len(left_leg)>0:
            left_leg_list.append([time_idx, np.mean(left_leg_2, axis=0)])
        if len(right_leg)>0:
            right_leg_list.append([time_idx, np.mean(right_leg_2, axis=0)])

        if False:
            center_point = np.mean(time_points, axis=0)
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1.set_title(pcd_info_list.dir_name + "_" + str(time_idx))
            ax2 = fig.add_subplot(132)
            ax2 = set_ax.set_ax(ax2, title=pcd_info_list.dir_name + "_" + str(time_idx), xlim=[center_point[0]-500, center_point[0]+500], ylim=[center_point[1]-500, center_point[1]+500])
            ax3 = fig.add_subplot(133)
            ax3 = set_ax.set_ax(ax3, title=pcd_info_list.dir_name + "_" + str(time_idx), xlim=[center_point[0]-500, center_point[0]+500], ylim=[center_point[1]-500, center_point[1]+500])

            ax1.scatter(time_points[:, 0], time_points[:, 1], s=1)
            ax2.scatter(left_leg[:, 0], left_leg[:, 1], s=1, color="red")
            ax2.scatter(right_leg[:, 0], right_leg[:, 1], s=1, color="blue")
            ax3.scatter(left_leg[:, 0], left_leg[:, 1], s=1, color="red")
            ax3.scatter(right_leg_2[:, 0], right_leg_2[:, 1], s=1, color="blue")
            ax3.scatter(np.mean(left_leg_2, axis=0)[0], np.mean(left_leg_2, axis=0)[1], s=10, color="red")
            ax3.scatter(np.mean(right_leg_2, axis=0)[0], np.mean(right_leg_2, axis=0)[1], s=10, color="blue")

            plt.show()
            plt.close()

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

    threshold = 50
    is_left_stand_list = []
    for i in range(1, len(left_leg_list)-1):
        before_distance = abs(left_leg_list[i-1][1][0]-left_leg_list[i][1][0])
        after_distance = abs(left_leg_list[i][1][0]-left_leg_list[i+1][1][0])
        
        if before_distance < threshold and after_distance < threshold:
            is_left_stand_list.append(True)
        else:
            is_left_stand_list.append(False)
    
    is_right_stand_list = []
    for i in range(1, len(right_leg_list)-1):
        before_distance = abs(right_leg_list[i-1][1][0]-right_leg_list[i][1][0])
        after_distance = abs(right_leg_list[i][1][0]-right_leg_list[i+1][1][0])
        
        if before_distance < threshold and after_distance < threshold:
            is_right_stand_list.append(True)
        else:
            is_right_stand_list.append(False)
    
    left_stand_phase_list = []
    for idx in range(len(is_left_stand_list)-2):
        is_stand_0 = is_left_stand_list[idx]
        is_stand_1 = is_left_stand_list[idx+1]
        is_stand_2 = is_left_stand_list[idx+2]
        if is_stand_0 and is_stand_1 and is_stand_2:
            left_stand_phase_list += [idx, idx+1, idx+2]
    left_stand_phase_list = list(set(left_stand_phase_list))

    right_stand_phase_list = []
    for idx in range(len(is_right_stand_list)-2):
        is_stand_0 = is_right_stand_list[idx]
        is_stand_1 = is_right_stand_list[idx+1]
        is_stand_2 = is_right_stand_list[idx+2]
        if is_stand_0 and is_stand_1 and is_stand_2:
            right_stand_phase_list += [idx, idx+1, idx+2]
    right_stand_phase_list = list(set(right_stand_phase_list))
    
    # 連番ごとに切り分ける
    left_stand_phase_list = np.array(left_stand_phase_list)
    left_stand_phase_list = np.sort(left_stand_phase_list)
    left_stand_phase_list = np.split(left_stand_phase_list, np.where(np.diff(left_stand_phase_list) != 1)[0]+1)
    left_stand_phase_list = [x for x in left_stand_phase_list if len(x) > 1]
    
    right_stand_points_list = np.array(right_stand_phase_list)
    right_stand_points_list = np.sort(right_stand_points_list)
    right_stand_points_list = np.split(right_stand_points_list, np.where(np.diff(right_stand_points_list) != 1)[0]+1)
    right_stand_points_list = [x for x in right_stand_points_list if len(x) > 1]

    # 左右足の距離を表示
    left_stand_points = []
    left_swing_points = []
    for idx, is_left_stand in enumerate(is_left_stand_list):
        if is_left_stand:
            left_stand_points.append([left_leg_list[idx][0], left_leg_list[idx][1][0]])
        else:
            left_swing_points.append([left_leg_list[idx][0], left_leg_list[idx][1][0]])

    right_stand_points = []
    right_swing_points = []
    for idx, is_right_stand in enumerate(is_right_stand_list):
        if is_right_stand:
            right_stand_points.append([right_leg_list[idx][0], right_leg_list[idx][1][0]])
        else:
            right_swing_points.append([right_leg_list[idx][0], right_leg_list[idx][1][0]])

    
    left_x = None
    left_y = None
    for idx in range(len(left_stand_phase_list)):
        if idx==0 and left_stand_phase_list[idx][0]!=0:
            swing_time_idx_list = np.array(range(0, left_stand_phase_list[idx][0]))
            swing_point_list = np.array([x[1][0] for x in left_leg_list[0:left_stand_phase_list[idx][0]]])
            cofficient = np.polyfit(swing_time_idx_list, swing_point_list, 1)
            swing_distance_list = swing_time_idx_list*cofficient[0] + cofficient[1]

            if left_x is None:
                left_x = swing_time_idx_list
                left_y = swing_distance_list
        
        left_stand_phase = left_stand_phase_list[idx]
        stand_time_idx_list = np.array(range(left_stand_phase[0], left_stand_phase[-1]+1))
        stand_distance_mean = np.mean(np.array([x[1][0] for x in left_leg_list[left_stand_phase[0]:left_stand_phase[-1]+1]]))
        stand_distance_list = np.array([stand_distance_mean for _ in range(len(stand_time_idx_list))])

        if left_x is None:
            left_x = stand_time_idx_list
            left_y = stand_distance_list
        else:
            left_x = np.concatenate([left_x, stand_time_idx_list])
            left_y = np.concatenate([left_y, stand_distance_list])
        
        if idx != len(left_stand_phase_list)-1:
            swing_time_idx_list = np.array(range(left_stand_phase[-1]+1, left_stand_phase_list[idx+1][0]))
            swing_point_list = np.array([x[1][0] for x in left_leg_list[left_stand_phase[-1]+1:left_stand_phase_list[idx+1][0]]])
        else:
            swing_time_idx_list = np.array(range(left_stand_phase[-1]+1, len(left_leg_list)))
            swing_point_list = np.array([x[1][0] for x in left_leg_list[left_stand_phase[-1]+1:]])
        cofficient = np.polyfit(swing_time_idx_list, swing_point_list, 1)
        swing_distance_list = swing_time_idx_list*cofficient[0] + cofficient[1]

        left_x = np.concatenate([left_x, swing_time_idx_list])
        left_y = np.concatenate([left_y, swing_distance_list])

    right_x = None
    right_y = None
    for idx in range(len(right_stand_points_list)):
        if idx==0 and right_stand_points_list[idx][0]!=0:
            swing_time_idx_list = np.array(range(0, right_stand_points_list[idx][0]))
            swing_point_list = np.array([x[1][0] for x in right_leg_list[0:right_stand_points_list[idx][0]]])
            cofficient = np.polyfit(swing_time_idx_list, swing_point_list, 1)
            swing_distance_list = swing_time_idx_list*cofficient[0] + cofficient[1]

            if right_x is None:
                right_x = swing_time_idx_list
                right_y = swing_distance_list
        
        right_stand_phase = right_stand_points_list[idx]
        stand_time_idx_list = np.array(range(right_stand_phase[0], right_stand_phase[-1]+1))
        stand_distance_mean = np.mean(np.array([x[1][0] for x in right_leg_list[right_stand_phase[0]:right_stand_phase[-1]+1]]))
        stand_distance_list = np.array([stand_distance_mean for _ in range(len(stand_time_idx_list))])

        if right_x is None:
            right_x = stand_time_idx_list
            right_y = stand_distance_list
        else:
            right_x = np.concatenate([right_x, stand_time_idx_list])
            right_y = np.concatenate([right_y, stand_distance_list])
        
        if idx != len(right_stand_points_list)-1:
            swing_time_idx_list = np.array(range(right_stand_phase[-1]+1, right_stand_points_list[idx+1][0]))
            swing_point_list = np.array([x[1][0] for x in right_leg_list[right_stand_phase[-1]+1:right_stand_points_list[idx+1][0]]])
        else:
            swing_time_idx_list = np.array(range(right_stand_phase[-1]+1, len(right_leg_list)))
            swing_point_list = np.array([x[1][0] for x in right_leg_list[right_stand_phase[-1]+1:]])
        cofficient = np.polyfit(swing_time_idx_list, swing_point_list, 1)
        swing_distance_list = swing_time_idx_list*cofficient[0] + cofficient[1]

        right_x = np.concatenate([right_x, swing_time_idx_list])
        right_y = np.concatenate([right_y, swing_distance_list])

    cross_points = ori_method.get_cross_points(left_x, left_y, right_x, right_y)

    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.scatter(np.array(left_stand_points)[:, 0]*0.025, np.array(left_stand_points)[:, 1], color="orange", label="left_stand", s=1)
        ax1.scatter(np.array(left_swing_points)[:, 0]*0.025, np.array(left_swing_points)[:, 1], color="red", label="left_swing", s=1)
        ax1.scatter(np.array(right_stand_points)[:, 0]*0.025, np.array(right_stand_points)[:, 1], color="skyblue", label="right_stand", s=1)
        ax1.scatter(np.array(right_swing_points)[:, 0]*0.025, np.array(right_swing_points)[:, 1], color="blue", label="right_swing", s=1)
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("distance (mm)")
        plt.suptitle(pcd_info_list.dir_name+"_threshold:"+str(threshold))
        plt.legend()
        plt.show()
        plt.close()

        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        ax2.plot(left_x*0.025, left_y, color="red", label="left")
        ax2.plot(right_x*0.025, right_y, color="blue", label="right")
        ax2.scatter(np.array(cross_points)[:, 0]*0.025, np.array(cross_points)[:, 1], color="black", label="cross_points", s=10)
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("distance (mm)")
        plt.suptitle(pcd_info_list.dir_name+"_threshold:"+str(threshold))
        plt.legend()
        plt.show()
        plt.close()

    return cross_points
