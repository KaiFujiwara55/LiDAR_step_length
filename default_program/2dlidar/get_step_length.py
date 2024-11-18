import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

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
        if "mapping" in dir:
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
        height = ori_method.get_height_all(pcd_info_3d.cloud_list)
        theta_z_list = ori_method.get_collect_theta_z(integraded_area_points_list_3d, integraded_area_center_point_list_3d)

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

                    # 2dの描画
                    # y軸方向に揃える
                    normalized_points = points - center_point
                    normalized_points = def_method.rotate_points(normalized_points, theta_x=0, theta_y=0, theta_z=theta_z_list[0])
                    points = def_method.rotate_points(points, theta_x=0, theta_y=0, theta_z=theta_z_list[0])

                    center_x = np.sort(points[:, 0])[len(points)//2]
                    left_leg = points[points[:, 0] < center_x]
                    right_leg = points[points[:, 0] > center_x]
                    if group_idx_2d not in left_leg_list.keys():
                        left_leg_list[group_idx_2d] = []
                        right_leg_list[group_idx_2d] = []
                    left_leg_list[group_idx_2d].append([time_idx, np.mean(left_leg, axis=0)])
                    right_leg_list[group_idx_2d].append([time_idx, np.mean(right_leg, axis=0)])

                    
                    normalized_left_leg = normalized_points[normalized_points[:, 0] < 0]
                    normalized_left_center = np.mean(normalized_left_leg, axis=0)
                    normalized_right_leg = normalized_points[normalized_points[:, 0] > 0]
                    normalized_right_center = np.mean(normalized_right_leg, axis=0)

                    if False:
                        fig = plt.figure()
                        ax1 = fig.add_subplot(121)
                        title = f"{pcd_info_2d.dir_name}_{sec}s_{group_idx_2d}_{time_idx}"
                        ax1 = set_ax.set_ax(ax1, title=title, xlim=[pcd_info_2d.get_all_min()[1], pcd_info_2d.get_all_max()[1]], ylim=[pcd_info_2d.get_all_min()[0], pcd_info_2d.get_all_max()[0]])
                        ax2 = fig.add_subplot(122)
                        ax2 = set_ax.set_ax(ax2, title=title, xlim=[-250, 250], ylim=[-250, 250])

                        ax1.scatter(left_leg[:, 0]+corect_x, left_leg[:, 1]+corect_y, c="b", s=1)
                        ax1.scatter(right_leg[:, 0]+corect_x, right_leg[:, 1]+corect_y, c="r", s=1)
                        ax2.scatter(normalized_left_leg[:, 0], normalized_left_leg[:, 1], c="b", s=1)
                        ax2.scatter(normalized_right_leg[:, 0], normalized_right_leg[:, 1], c="r", s=1)

                        # plt.show()
                        gif.save_fig(fig)
                        plt.close()
                output_path = f"/Users/kai/大学/小川研/LiDAR_step_length/gif/2d/{pcd_info_2d.dir_name}_0025s.gif"
                gif.create_gif(output_path, duration=0.025)

        left_speed_list = []
        right_speed_list = []
        for group_idx in left_leg_list.keys():
            for i in range(len(left_leg_list[group_idx_2d])):
                if i == 0:
                    continue
                left_speed = (ori_method.calc_points_distance(left_leg_list[group_idx_2d][i][1], left_leg_list[group_idx_2d][i-1][1])) / (sec_2*(left_leg_list[group_idx_2d][i][0] - left_leg_list[group_idx_2d][i-1][0]))
                right_speed = (ori_method.calc_points_distance(right_leg_list[group_idx_2d][i][1], right_leg_list[group_idx_2d][i-1][1])) / (sec_2*(right_leg_list[group_idx_2d][i][0] - right_leg_list[group_idx_2d][i-1][0]))

                left_speed_list.append(left_speed)
                right_speed_list.append(right_speed)
            
            # 移動平均を取る
            window = 10
            left_speed_list = np.convolve(left_speed_list, np.ones(window)/window, mode='valid')
            right_speed_list = np.convolve(right_speed_list, np.ones(window)/window, mode='valid')

            plt.figure()
            plt.plot(left_speed_list, label="left")
            plt.plot(right_speed_list, label="right")
            plt.suptitle(f"{pcd_info_2d.dir_name}")
            plt.xlabel("time")
            plt.ylabel("speed (mm/s)")
            plt.legend()
            plt.show()
            plt.close()