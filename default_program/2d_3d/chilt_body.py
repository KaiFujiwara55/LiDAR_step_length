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
    dir_list = glob.glob(f"/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_{sec}s/2d/*")
    for dir in dir_list:
        if "nothing" in dir:
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

        area_path_3d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/{sec}s/{pcd_info_3d.dir_name}"
        center_path_3d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/{sec}s/{pcd_info_3d.dir_name}"
        time_area_points_list_3d, time_area_center_point_list_3d = ori_method.load_original_data(area_path_3d, center_path_3d)
        integraded_area_points_list_3d, integraded_area_center_point_list_3d = ori_method.grouping_points_list(time_area_points_list_3d, time_area_center_point_list_3d, integrade_threshold=5)
        sec_2 = 0.1
        cloud_folder_path = "/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_"+str(sec_2).replace(".", "")+"s/3d/"+pcd_info_3d.dir_name
        integraded_area_points_list_3d, integraded_area_center_point_list_3d = ori_method.grouping_points_list_2(integraded_area_points_list_3d, integraded_area_center_point_list_3d, cloud_folder_path, sec=sec_2, is_incline=False)
        height = ori_method.get_height_all(pcd_info_3d.cloud_list)
        theta_z_list = ori_method.get_collect_theta_z(integraded_area_center_point_list_3d)

        gif = create_gif.create_gif(False)
        for time_idx in range(min(len(time_area_points_list_2d), len(integraded_area_points_list_3d[0]))):
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection="3d")
            title = f"{pcd_info_2d.dir_name}_{sec}s_{time_idx}"
            ax1 = set_ax.set_ax(ax1, title=title)
            ax2 = fig.add_subplot(122)
            ax2 = set_ax.set_ax(ax2, title=title, xlim=[-250, 250], ylim=[-250, 250])

            # 2dの描画
            for group_idx_2d in range(len(time_area_points_list_2d[time_idx])):
                # y軸方向に揃える
                points = time_area_points_list_2d[time_idx][group_idx_2d]
                if len(points) == 0:
                    continue
                normalized_points = points - time_area_center_point_list_2d[time_idx][group_idx_2d]
                normalized_points = def_method.rotate_points(normalized_points, theta_x=0, theta_y=0, theta_z=theta_z_list[0])
                points = np.array(points)
                
                ax1.scatter(points[:, 0]+corect_x, points[:, 1]+corect_y, 450, c="b", s=1)
                ax2.scatter(normalized_points[:, 0], normalized_points[:, 1], c="b", s=1)

            # 3dの描画
            for group_idx_3d in range(len(integraded_area_points_list_3d)):
                points = integraded_area_points_list_3d[group_idx_3d][time_idx]
                if len(points) == 0:
                    continue
                normalized_points = points.copy()
                normalized_points[:, 0] = points[:, 0] - integraded_area_center_point_list_3d[group_idx_3d][time_idx][0]
                normalized_points[:, 1] = points[:, 1] - integraded_area_center_point_list_3d[group_idx_3d][time_idx][1]
                normalized_points = normalized_points[(normalized_points[:, 2] > height*0.5) & (normalized_points[:, 2] < height-30)]
                normalized_points = def_method.rotate_points(normalized_points, theta_x=0, theta_y=0, theta_z=-theta_z_list[group_idx_3d])
                points = np.array(points)
                ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", s=1)
                ax2.scatter(normalized_points[:, 0], normalized_points[:, 1], c="r", s=1)

            plt.show()
            gif.save_fig(fig)
            plt.close()
        output_path = f"/Users/kai/大学/小川研/LiDAR_step_length/gif/2d_3d/{pcd_info_2d.dir_name}_{sec}s.gif"
        gif.create_gif(output_path)
