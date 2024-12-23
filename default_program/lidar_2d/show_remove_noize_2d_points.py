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

# ノイズ除去のクラスをインスタンス化
def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

sec_list = ["0025"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241218/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/2d/*")
    
    for dir in dirs:
        # # スキップしない対象
        if "cose_7_1_f" not in dir:
            continue

        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        
        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")        
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        # 処理結果を読み込み
        area_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_list.dir_name}"
        center_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_list.dir_name}"
        time_area_points_list_2d, time_area_center_point_list_2d = ori_method.load_original_data(area_path_2d, center_path_2d)

        gif = create_gif.create_gif(False)
        for time_idx in range(len(time_area_points_list_2d)):
            cloud = pcd_info_list.cloud_list[time_idx]
            points = np.array(cloud)

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1 = ax_set.set_ax(ax1, title="raw_points")
            ax1.scatter(points[:, 0], points[:, 1], s=1, c="b")
            ax2 = fig.add_subplot(122)
            ax2 = ax_set.set_ax(ax2, title="remove_noize_points")
            for group_idx in range(len(time_area_points_list_2d[time_idx])):
                remove_noize_points = time_area_points_list_2d[time_idx][group_idx]                
                
                ax2.scatter(remove_noize_points[:, 0], remove_noize_points[:, 1], s=1, c="b")

            plt.suptitle(f"{pcd_info_list.dir_name}_timeidx:{time_idx}")
            plt.show()
            gif.save_fig(fig)
            plt.close()
        gif.create_gif("/Users/kai/大学/小川研/LiDAR_step_length/gif/remove_noize/"+pcd_info_list.dir_name+".gif")

