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

def extract_points(cloud, x_min, x_max, y_min, y_max):
    points = def_method.get_points(cloud)
    
    new_points = points[(points[:, 0]>x_min) & (points[:, 0]<x_max) & (points[:, 1]>y_min) & (points[:, 1]<y_max)]
    new_cloud = def_method.get_cloud(new_points)
    
    return new_cloud, new_points

sec_list = ["0025"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241204/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/2d/*")
    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        print(dir)
        try:
            # pcdファイルの情報を取得
            pcd_info_list = get_pcd_information.get_pcd_information()
            pcd_info_list.load_pcd_dir(dir)
            # plot用のaxのdefault設定
            ax_set = plot.set_plot()
            ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

            gif = create_gif.create_gif(False)
            for time_idx, cloud in enumerate(pcd_info_list.cloud_list):
                points = np.array(cloud)
                fig = plt.figure()
                ax = fig.add_subplot(121)
                title = f"{pcd_info_list.dir_name}_{sec}s_{time_idx}"
                ax = ax_set.set_ax(ax, title=title)
                ax.scatter(points[:, 0], points[:, 1], s=1, c="b")

                plt.show()
                gif.save_fig(fig)
                plt.close()

            output_path = "/Users/kai/大学/小川研/LiDAR_step_length/gif/2d_raw_cloud/"+pcd_info_list.dir_name+".gif"
            gif.create_gif(output_path, duration=0.025)
        except Exception as e:
            print(e)
            continue            
