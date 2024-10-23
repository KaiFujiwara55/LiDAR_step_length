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
    dirs = glob.glob(f"/Users/kai/大学/小川研/LiDAR_step_length/20241011/pcd/pcd_{sec}s/*")

    def_method = default_method.cloud_method()
    org_method = original_method.cloud_method()

    for dir in dirs:
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        if "fujiwara" in dir:
            continue
        
        pcd_info_list.load_pcd_dir(dir)
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)
        
        all_points = None
        for points in pcd_info_list.points_list:
            if all_points is None:
                all_points = points
            else:
                all_points = np.concatenate([all_points, points], axis=0)
        # 高さ情報を補正
        all_points[:, 2] = all_points[:, 2] + 1300

        # ダウンサンプリング
        cloud = pcl.PointCloud()
        cloud.from_array(all_points)
        grid_filtered_cloud = def_method.voxel_grid_filter(cloud)

        # 2000mm以下の点群を除去
        lower_cloud = def_method.filter_area(grid_filtered_cloud, z_max=2000)
        lower_points = np.array(lower_cloud)
        upper_cloud = def_method.filter_area(grid_filtered_cloud, z_min=2000)
        upper_points = np.array(upper_cloud)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = ax_set.set_ax(ax)
        ax.scatter(upper_points[:, 0], upper_points[:, 1], upper_points[:, 2], s=1, c="red")
        # plt.show()
        plt.close()

        # 天井を抽出
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1 = ax_set.set_ax(ax1, title="y-z", xlabel="y", ylabel="z", xlim=[-500, 500], ylim=[2000, 2500])
        ax2 = ax_set.set_ax(ax2, title="x-z", xlabel="x", ylabel="z", xlim=[0, 40000], ylim=[2000, 2500])

        ksearch = 50
        distance_threshold = 50
        cloud_plane, cloud_non_plane, coefficients, indices = def_method.segment_plane(upper_cloud, ksearch, distance_threshold)

        print("Y-Zの傾き: ", -1*(coefficients[1]/coefficients[2]), "傾き角度:", np.degrees(np.arctan(-1*(coefficients[1]/coefficients[2]))))
        print("X-Zの傾き: ", -1*(coefficients[0]/coefficients[2]), "傾き角度:", np.degrees(np.arctan(-1*(coefficients[0]/coefficients[2]))))

        ax1.plot(np.linspace(-500, 500, 100), -1*(coefficients[1]/coefficients[2])*np.linspace(-500, 500, 100)-coefficients[3]/coefficients[2], c="red")
        ax2.plot(np.linspace(0, 40000, 100), -1*(coefficients[0]/coefficients[2])*np.linspace(0, 40000, 100)-coefficients[3]/coefficients[2], c="red")

        plt.show()
        plt.close()


        ##############################
        print("傾き補正")
        # 傾き分を回転補正
        theta_x = np.arctan(-1*(coefficients[1]/coefficients[2]))*(-1)
        theta_y = np.arctan(-1*(coefficients[0]/coefficients[2]))
        print("修正角度: ", np.degrees(theta_x), np.degrees(theta_y))
        cloud = def_method.rotate_cloud(cloud, theta_x=theta_x, theta_y=theta_y)

        # ダウンサンプリング
        grid_filtered_cloud = def_method.voxel_grid_filter(cloud)

        # 2000mm以下の点群を除去
        lower_cloud = def_method.filter_area(grid_filtered_cloud, z_max=2000)
        lower_points = np.array(lower_cloud)
        upper_cloud = def_method.filter_area(grid_filtered_cloud, z_min=2000)
        upper_points = np.array(upper_cloud)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = ax_set.set_ax(ax)
        ax.scatter(upper_points[:, 0], upper_points[:, 1], upper_points[:, 2], s=1, c="red")
        # plt.show()
        plt.close()

        # 天井を抽出
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1 = ax_set.set_ax(ax1, title="y-z", xlabel="y", ylabel="z", xlim=[-500, 500], ylim=[2000, 2500])
        ax2 = ax_set.set_ax(ax2, title="x-z", xlabel="x", ylabel="z", xlim=[0, 40000], ylim=[2000, 2500])

        ksearch = 50
        distance_threshold = 50
        cloud_plane, cloud_non_plane, coefficients, indices = def_method.segment_plane(upper_cloud, ksearch, distance_threshold)

        print("Y-Zの傾き: ", -1*(coefficients[1]/coefficients[2]), "傾き角度:", np.degrees(np.arctan(-1*(coefficients[1]/coefficients[2]))))
        print("X-Zの傾き: ", -1*(coefficients[0]/coefficients[2]), "傾き角度:", np.degrees(np.arctan(-1*(coefficients[0]/coefficients[2]))))

        ax1.plot(np.linspace(-500, 500, 100), -1*(coefficients[1]/coefficients[2])*np.linspace(-500, 500, 100)-coefficients[3]/coefficients[2], c="red")
        ax2.plot(np.linspace(0, 40000, 100), -1*(coefficients[0]/coefficients[2])*np.linspace(0, 40000, 100)-coefficients[3]/coefficients[2], c="red")

        plt.show()
        plt.close()
        print()


        