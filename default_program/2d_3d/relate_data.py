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

from default_program.lidar_3d import get_step_length_chilt_func
from default_program.lidar_2d import get_step_length_half_func
from default_program.lidar_2d import get_step_length_article

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

set_ax_2d = plot.set_plot()
set_ax_3d = plot.set_plot()

corect_x = -20
corect_y = -200

sec_2d = 0.025
sec_3d_1 = 0.1
sec_3d_2 = 0.1

# データの読み込み
dir_list = glob.glob("/Users/kai/大学/小川研/Lidar_step_length/20241204/pcd_"+str(sec_3d_1).replace(".", "")+"s/3d/*")
dir_list = sorted(dir_list)

for dir in dir_list:
    if not "cose_6" in dir:
        continue

    print(dir)

    try:
        dir_2d = dir.replace("3d", "2d").replace(str(sec_3d_1).replace(".", ""), str(sec_2d).replace(".", ""))
        dir_3d = dir.replace("2d", "3d").replace(str(sec_2d).replace(".", ""), str(sec_3d_1).replace(".", ""))


        pcd_info_2d = get_pcd_information.get_pcd_information()
        pcd_info_2d.load_pcd_dir(dir_2d)
        pcd_info_3d = get_pcd_information.get_pcd_information()
        pcd_info_3d.load_pcd_dir(dir_3d)

        set_ax_2d.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_2d.get_all_min()[0], pcd_info_2d.get_all_max()[0]), ylim=(pcd_info_2d.get_all_min()[1], pcd_info_2d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)
        set_ax_3d.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_3d.get_all_min()[0], pcd_info_3d.get_all_max()[0]), ylim=(pcd_info_3d.get_all_min()[1], pcd_info_3d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)

        left_peak_list, right_peak_list, step_length_list_2d = get_step_length_half_func.get_step(0.025, dir_2d)
        peak_list, step_length_list_3d = get_step_length_chilt_func.get_step(sec_3d_1, sec_3d_2, dir_3d)
        cross_points = get_step_length_article.get_step(0.025, dir_2d)

        # 2dLidarの設置場所を設定
        # x, y, z = (mm, mm, bool)
        if "cose_1" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = 200/np.sqrt(2), 200/np.sqrt(2), False
        elif "cose_2" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = 3500/np.sqrt(2), 3500/np.sqrt(2), False
        elif "cose_3" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = (7000-200)/np.sqrt(2), (7000-200)/np.sqrt(2), False
        elif "cose_4" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2)+200/np.sqrt(2), 7000/np.sqrt(2)-200/np.sqrt(2), True
        elif "cose_5" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2)+3500/np.sqrt(2), 7000/np.sqrt(2)-3500/np.sqrt(2), True
        elif "cose_6" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2)+7000/np.sqrt(2), 7000/np.sqrt(2)-7000/np.sqrt(2), True
        elif "cose_7" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2), 7000/np.sqrt(2), False
        elif "cose_8" in dir:
            x_2d_lidar, y_2d_lidar, is_inverse = -200, 0, False
        
        # 2dの歩幅を取得する
        peak_time_idx_2d = np.sort(np.array([x[0] for x in left_peak_list]+[x[0] for x in right_peak_list]))
        peak_points_2d = np.array([x[1] for x in left_peak_list]+[x[1] for x in right_peak_list])[np.argsort(np.array([x[0] for x in left_peak_list]+[x[0] for x in right_peak_list]))]
        cofficient = np.polyfit(peak_points_2d[:, 0], peak_points_2d[:, 1], 1)
        collect_theta_z = np.arctan(cofficient[0])
        rotated_peak_points_2d = def_method.rotate_points(peak_points_2d, theta_z=-collect_theta_z)
        
        # 3d
        peak_time_idx_3d = {}
        peak_points_3d = {}
        for group_idx, group_peak in enumerate(peak_list):
            peak_time_idx_3d[group_idx] = np.sort(np.array([x[0] for x in group_peak]))
            peak_points_3d[group_idx] = np.array([x[1] for x in group_peak])[np.argsort(np.array([x[0] for x in group_peak]))]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = set_ax_2d.set_ax(ax, title=pcd_info_2d.dir_name, xlim=[0, 10000], ylim=[-5000, 5000])
        # 補助線をひく
        square_x = np.linspace(0, 7000/np.sqrt(2), 100)
        y_square_down = np.tan(np.radians(-45))*square_x
        y_square_up = np.tan(np.radians(45))*square_x
        ax.plot(square_x, y_square_down, c="yellow", label="room_area")
        ax.plot(square_x, y_square_up, c="yellow")
        square_x = np.linspace(7000/np.sqrt(2), 7000/np.sqrt(2)*2, 100)
        y_square_down = np.tan(np.radians(-45))*square_x+7000/np.sqrt(2)*2
        y_square_up = np.tan(np.radians(45))*square_x-7000/np.sqrt(2)*2
        ax.plot(square_x, y_square_down, c="yellow")
        ax.plot(square_x, y_square_up, c="yellow")
        x = np.linspace(0, 12000, 100)
        y_down = np.tan(np.radians(-35.2))*x
        y_up = np.tan(np.radians(35.2))*x
        ax.plot(x, y_down, c="black", label="avia_range")
        ax.plot(x, y_up, c="black")

        # 3dのピーク点の線形近似を行う
        cofficient = np.polyfit(peak_points_3d[0][:, 0], peak_points_3d[0][:, 1], 1)
        x = np.linspace(0, 10000, 100)
        y = cofficient[0]*x + cofficient[1]
        collect_radian_2d = np.arctan(cofficient[0])

        # LiDARの設置場所より補正を行う
        collected_cross_points = np.array([[x[1], 0, 0] for x in cross_points])
        # 2dLidarの設置場所によって、反転させるかを分ける
        if is_inverse:
            collected_cross_points = def_method.rotate_points(collected_cross_points, theta_z=collect_radian_2d+np.pi)
        else:
            collected_cross_points = def_method.rotate_points(collected_cross_points, theta_z=collect_radian_2d)
        collected_cross_points += np.array([x_2d_lidar, y_2d_lidar, 0])
        ax.scatter(collected_cross_points[:, 0], collected_cross_points[:, 1], s=5, c="blue", label="2d")

        for group_idx in peak_points_3d.keys():
            ax.scatter(peak_points_3d[group_idx][:, 0], peak_points_3d[group_idx][:, 1], s=5, c="r", label="3d")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
        plt.close()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        continue


    if False:
        # ヒストグラム
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        ax1.hist(step_length_list_2d, bins=55, range=(0, 1100))
        ax2.hist(step_length_list_3d[0], bins=55, range=(0, 1100))
        
        ax1.set_title("2d")
        ax2.set_title("3d")
        
        plt.suptitle(pcd_info_2d.dir_name)
        plt.tight_layout()
        plt.show()
        plt.close()

        # 平均値の比較
        print("2d", np.mean(step_length_list_2d))
        print("3d", np.mean(step_length_list_3d[0]))
