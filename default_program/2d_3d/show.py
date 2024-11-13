import os
import numpy as np
import pcl
import glob
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/kai/大学/小川研/LiDAR_step_length/default_program/3dlidar')
import plot
import default_method
import original_method
import get_pcd_information
import create_gif

sec_list = ["01"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241113/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/2d/*")
    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")

        for i, cloud in enumerate(pcd_info_list.cloud_list):
            points = np.array(cloud)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(points[:,0], points[:,1], s=1)
            title = f"{pcd_info_list.dir_name}_{sec}s_{i}"
            ax = ax_set.set_ax(ax, title=title)
            plt.show()
            plt.close()
