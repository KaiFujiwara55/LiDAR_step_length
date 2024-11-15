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
    
        if True:
            time_area_points_list = []
            time_area_center_point_list = []
            for i, cloud in enumerate(pcd_info_list.cloud_list):
                # 統計的外れ値除去
                mean_k = 5
                std_dev_mul_thresh = 1.0
                filtered_cloud = def_method.statistical_outlier_removal(cloud, mean_k, std_dev_mul_thresh)

                indices, sqr_distances = def_method.kdtree_search(filtered_cloud, k=500)
                points = np.array(cloud)
                filtered_points = np.array(filtered_cloud)
                
                for i in range(len(indices)):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1, c="b")
                    ax.scatter(np.array(filtered_cloud)[indices[i], 0], np.array(filtered_cloud)[indices[i], 1], s=5, c="g")
                    ax.scatter(np.array(filtered_cloud)[i, 0], np.array(filtered_cloud)[i, 1], s=10, c="r")
                    plt.show()
                    plt.close()

                filtered_points = np.array(filtered_cloud)

                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax1.scatter(points[:, 0], points[:, 1], s=1)
                title = pcd_info_list.dir_name+"_"+str(i)
                ax1 = ax_set.set_ax(ax1, title=title)

                ax2 = fig.add_subplot(212)
                ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1)
                ax2 = ax_set.set_ax(ax2, title=title)
                plt.show()
                plt.close()
                # area_points_list, area_center_point_list = ori_method.get_neighborhood_points(cloud, radius=250, count_threshold=10)

                # 時系列の点群を保存
                # time_area_points_list.append(area_points_list)
                # time_area_center_point_list.append(area_center_point_list)
