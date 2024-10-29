import os
import numpy as np
import pcl
import plot
import glob
import time
import matplotlib.pyplot as plt
import default_method
import original_method
import get_pcd_information
import create_gif

# sec_list = ["015", "02", "025", "03"]
sec_list = ["01"]
for sec in sec_list:
    dirs = glob.glob(f"/Users/kai/大学/小川研/LiDAR_step_length/20241028/pcd_{sec}s/*")

    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if "repeat" not in dir:
            continue
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)
        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")

        if False:
            time_cloud = []
            time_area_points_list = []
            time_area_center_point_list = []
            count = 0
            start = time.time()

            for cloud, cloud_name in zip(pcd_info_list.cloud_list, pcd_info_list.cloud_name_list):
                source_cloud = cloud
                filtered_cloud = cloud

                # 高さの正規化を行う
                source_cloud = def_method.get_cloud(def_method.get_points(source_cloud) + np.array([0, 0, 1300]))
                filtered_cloud = def_method.get_cloud(def_method.get_points(filtered_cloud) + np.array([0, 0, 1300]))

                tmp_cloud_list = []
                for i in range(2):
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
                        filtered_cloud = tmp_cloud_list[-1]
                        break
                    cloud_plane, cloud_non_plane = def_method.filter_points_by_distance(filtered_cloud, coefficients, distance_threshold)

                    filtered_cloud = cloud_non_plane
                    tmp_cloud_list.append(filtered_cloud)
                # 統計的外れ値除去
                mean_k = 50
                std_dev_mul_thresh = 1.0
                statistical_filtered_cloud = def_method.statistical_outlier_removal(filtered_cloud, mean_k, std_dev_mul_thresh)

                # ダウンサンプリング
                leaf_size = (100, 100, 100)
                grid_filtered_cloud = def_method.voxel_grid_filter(statistical_filtered_cloud, leaf_size)

                # 領域内の点群を取得
                area_points_list, area_center_point_list = ori_method.get_neighborhood_points(grid_filtered_cloud, radius=250, count_threshold=10)

                # 時系列の点群を保存
                time_cloud.append(statistical_filtered_cloud)
                time_area_points_list.append(area_points_list)
                time_area_center_point_list.append(area_center_point_list)

                print(cloud_name, cloud.size, statistical_filtered_cloud.size)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax = ax_set.set_ax(ax, title=cloud_name)
                points = def_method.get_points(grid_filtered_cloud)
                ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
                for cneter_point in area_center_point_list:
                    ax.scatter(cneter_point[0], cneter_point[1], cneter_point[2], c="red", s=10)
                plt.show()
                plt.close()
            
            # 処理結果を保存
            area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_points_list/{pcd_info_list.dir_name}_{sec}s"
            center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_center_point_list/{pcd_info_list.dir_name}_{sec}s"
            ori_method.save_original_data(time_area_points_list, time_area_center_point_list, area_path, center_path)

        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_points_list/{pcd_info_list.dir_name}_{sec}s"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_center_point_list/{pcd_info_list.dir_name}_{sec}s"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

        # 点群をグループ化
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5)

        # 中心点の軌跡から新たにグループを作成
        sec_2 = 0.1
        cloud_folder_path = "/Users/kai/大学/小川研/LiDAR_step_length/20241011/pcd_"+str(sec_2).replace(".", "")+"s/"+pcd_info_list.dir_name
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, is_incline=False)

        color_list = ["red", "blue", "green", "yellow", "purple", "orange"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = ax_set.set_ax(ax, title=pcd_info_list.dir_name)
        for gropu_idx in range(len(integraded_area_center_point_list)):
            group_points_list = integraded_area_points_list[gropu_idx]
            for time_idx in range(len(group_points_list)):
                points = group_points_list[time_idx]
                if len(points)>0:
                    ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c=color_list[gropu_idx])

        plt.show()
        plt.close()

