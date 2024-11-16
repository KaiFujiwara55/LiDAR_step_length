import os
import sys
import numpy as np
import pcl
import glob
import time
import matplotlib.pyplot as plt

sys.path.append("/Users/kai/大学/小川研/LiDAR_step_length")
from default_program.class_method import plot
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import get_pcd_information
from default_program.class_method import create_gif

# sec_list = ["015", "02", "025", "03"]
sec_list = ["01", "005, 0025"]
for sec in sec_list:
    dirs = glob.glob(f"/Users/kai/大学/小川研/LiDAR_step_length/20241113/pcd_{sec}s/3d/*")

    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)

        set_ax = plot.set_plot()
        set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")
        start = time.time()

        time_cloud = []
        time_area_points_list = []
        time_area_center_point_list = []
        count = 0

        # LiDAR自体の傾きを取得
        theta_x, theta_y, theta_z = ori_method.cloud_get_tilt(pcd_info_list, upper_threshold=2000-1300)
        for cloud, cloud_name in zip(pcd_info_list.cloud_list, pcd_info_list.cloud_name_list):
            print(cloud_name, cloud.size)
            # LiDAR自体の傾きを補正
            cloud = def_method.rotate_cloud(cloud, -theta_x, theta_y)

            source_cloud = cloud
            filtered_cloud = cloud

            # 高さの正規化を行う
            if "high" in pcd_info_list.dir_name:
                correction_height = 2000
            else:
                correction_height = 1300
            source_cloud = def_method.get_cloud(def_method.get_points(source_cloud) + np.array([0, 0, correction_height]))
            filtered_cloud = def_method.get_cloud(def_method.get_points(filtered_cloud) + np.array([0, 0, correction_height]))

            tmp_cloud_list = []
            flg = False
            for i in range(10):
                # knnを使用
                # indices, sqr_distances = def_method.kdtree_search(source_cloud, k=10)

                # ダウンサンプリング
                grid_filtered_cloud = def_method.voxel_grid_filter(filtered_cloud, leaf_size=(500, 500, 500))

                # 半径外れ値除去
                # filtered_cloud = def_method.radius_outlier_removal(source_cloud, radius=100, min_neighbors=2)

                # 統計的外れ値除去
                # filtered_cloud = def_method.statistical_outlier_removal(filtered_cloud, mean_k=50, std_dev_mul_thresh=1.0)

                # 平面の抽出
                ksearch = 50
                distance_threshold = 50
                cloud_plane, cloud_non_plane, coefficients, indices = def_method.segment_plane(grid_filtered_cloud, ksearch, distance_threshold)
                if cloud_plane.size==0:
                    print("No plane")
                    try:
                        filtered_cloud = tmp_cloud_list[-3]
                    except:
                        flg = True
                        pass
                    break
                cloud_plane, cloud_non_plane = def_method.filter_points_by_distance(filtered_cloud, coefficients, distance_threshold)

                
                filtered_cloud = cloud_non_plane
                tmp_cloud_list.append(filtered_cloud)
            # 統計的外れ値除去
            mean_k = 50
            std_dev_mul_thresh = 1.0
            statistical_filtered_cloud = def_method.statistical_outlier_removal(filtered_cloud, mean_k, std_dev_mul_thresh)

            # ダウンサンプリング
            # leaf_size = (500, 500, 100)
            # grid_filtered_cloud = def_method.voxel_grid_filter(statistical_filtered_cloud, leaf_size)

            # 高さの分散を取得
            # var_threshold = 10000
            # x_step = 500
            # y_step = 500
            # var_filtered_cloud, surface_xy_list = def_method.filter_by_height_var(grid_filtered_cloud, var_threshold, x_step, y_step)

            if flg:
                continue
            # 領域内の点群を取得
            area_points_list, area_center_point_list = ori_method.get_neighborhood_points(statistical_filtered_cloud, radius=250)

            # 時系列の点群を保存
            time_cloud.append(statistical_filtered_cloud)
            time_area_points_list.append(area_points_list)
            time_area_center_point_list.append(area_center_point_list)

        print(f"処理時間 : {time.time()-start}")

        # 処理結果を保存
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/{sec}s/{pcd_info_list.dir_name}"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/{sec}s/{pcd_info_list.dir_name}"
        ori_method.save_original_data(time_area_points_list, time_area_center_point_list, area_path, center_path)
