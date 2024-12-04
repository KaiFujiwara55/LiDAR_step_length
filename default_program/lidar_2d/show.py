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

sec_list = ["0025"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241204/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/2d/*")
    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if "nothing" in dir or "mapping" in dir:
            continue

        if "far" in dir:
            continue

        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        
        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")
        if "cose_8_1_t_1204" in dir or "cose_1_0_y_1204" in dir:
            continue

        # エラー吐いた奴ら
        if "cose_2_1_t_1204" in dir:
            continue
        
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        # 処理結果を読み込み
        area_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_list.dir_name}"
        center_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_list.dir_name}"
        time_area_points_list_2d, time_area_center_point_list_2d = ori_method.load_original_data(area_path_2d, center_path_2d)

        gif = create_gif.create_gif(False)
        for time_idx in range(len(time_area_points_list_2d)):
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1 = ax_set.set_ax(ax1, title=f"{pcd_info_list.dir_name}_timeidx:{time_idx}", xlim=[0, 10000], ylim=[-500, 500])
            ax2 = fig.add_subplot(212)
            for group_idx in range(len(time_area_points_list_2d[time_idx])):
                points = time_area_points_list_2d[time_idx][group_idx]
                center_point = time_area_center_point_list_2d[time_idx][group_idx]

                if len(center_point) == 0:
                    continue
                
                
                center_point_2 = np.mean(points, axis=0)
                left_leg = points[points[:, 1]>center_point_2[1]]
                right_leg = points[points[:, 1]<center_point_2[1]]
                
                ax1.scatter(left_leg[:, 0], left_leg[:, 1], s=1, c="g")
                ax1.scatter(right_leg[:, 0], right_leg[:, 1], s=1, c="r")
                
                ax2 = ax_set.set_ax(ax2, xlim=[center_point_2[0]-500, center_point_2[0]+500], ylim=[center_point_2[1]-500, center_point_2[1]+500])
                
                ax2.scatter(left_leg[:, 0], left_leg[:, 1], s=1, c="g")
                ax2.scatter(np.mean(left_leg, axis=0)[0], np.mean(left_leg, axis=0)[1], s=10, c="g")
                ax2.scatter(right_leg[:, 0], right_leg[:, 1], s=1, c="r")
                ax2.scatter(np.mean(right_leg, axis=0)[0], np.mean(right_leg, axis=0)[1], s=10, c="r")


            plt.show()
            gif.save_fig(fig)
            plt.close()
        gif.create_gif("/Users/kai/大学/小川研/LiDAR_step_length/gif/separete/"+pcd_info_list.dir_name+"_center.gif")

