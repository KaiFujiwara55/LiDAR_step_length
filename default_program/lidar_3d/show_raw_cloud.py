import os
import numpy as np
import pcl
import glob
import time
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/kai/大学/小川研/LiDAR_step_length/")
from default_program.class_method import plot
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import get_pcd_information
from default_program.class_method import create_gif

sec_list = ["01"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241204/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/3d/*")

    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(0, pcd_info_list.get_all_max()[2]), azim=150)

        if ("cose_3_1_t" not in dir and "cose_4_1_f" not in dir and "cose_6_0_f" not in dir and "cose_6_0_y" not in dir):
            continue
        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")

        
        for cloud in pcd_info_list.cloud_list:
            down_cloud = def_method.voxel_grid_filter(cloud)

            ax = plt.subplot(111, projection='3d')
            ax = ax_set.set_ax(ax, xlim=[0, 10000], ylim=[-5000, 5000], azim=30, elev=0)
            points = np.array(down_cloud)
            points = points[points[:, 2] < 16000]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue")
            plt.show()
            plt.close()
