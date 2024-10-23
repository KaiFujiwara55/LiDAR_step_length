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

    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if "fujiwara" in dir:
            continue
        # gif作成用のクラスをインスタンス化
        gif = create_gif.create_gif(create_flg=False)

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

            # LiDAR自体の傾きを取得
            theta_x, theta_y, theta_z = ori_method.cloud_get_tilt(pcd_info_list, upper_threshold=2000-1300)
            for cloud, cloud_name in zip(pcd_info_list.cloud_list, pcd_info_list.cloud_name_list):
                # LiDAR自体の傾きを補正
                cloud = def_method.rotate_cloud(cloud, -theta_x, theta_y)

                source_cloud = cloud
                filtered_cloud = cloud

                # 高さの正規化を行う
                source_cloud = def_method.get_cloud(def_method.get_points(source_cloud) + np.array([0, 0, 1300]))
                filtered_cloud = def_method.get_cloud(def_method.get_points(filtered_cloud) + np.array([0, 0, 1300]))

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
                    cloud_plane, cloud_non_plane = def_method.filter_points_by_distance(filtered_cloud, coefficients, distance_threshold)

                    if cloud_plane==None:
                        print("No plane")
                        break
                    
                    filtered_cloud = cloud_non_plane
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

                # 領域内の点群を取得
                area_points_list, area_center_point_list = ori_method.get_neighborhood_points(statistical_filtered_cloud, radius=250)

                # 時系列の点群を保存
                time_cloud.append(statistical_filtered_cloud)
                time_area_points_list.append(area_points_list)
                time_area_center_point_list.append(area_center_point_list)

            print(f"処理時間 : {time.time()-start}")

            # 処理結果を保存
            area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/default_program/3dlidar/tmp_folder/time_area_points_list/{pcd_info_list.dir_name}_{sec}s"
            center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/default_program/3dlidar/tmp_folder/time_area_center_point_list/{pcd_info_list.dir_name}_{sec}s"
            ori_method.save_original_data(time_area_points_list, time_area_center_point_list, area_path, center_path)

        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/default_program/3dlidar/tmp_folder/time_area_points_list/{pcd_info_list.dir_name}_{sec}s"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/default_program/3dlidar/tmp_folder/time_area_center_point_list/{pcd_info_list.dir_name}_{sec}s"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

        # 点群をグループ化
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5)

        vectros_list = ori_method.get_vector(integraded_area_center_point_list)
        move_flg_list = ori_method.judge_move(vectros_list)

        cloud_folder_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241011/pcd/pcd_01s/{pcd_info_list.dir_name}"
        new_integraded_area_points_list, new_integraded_area_center_point_list = ori_method.grouping_points_list_2(pcd_info_list, integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, 0.1)

        # グループの中心点の軌跡から、さらにグループ分けを行う
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # ax = ax_set.set_ax(ax, title=pcd_info_list.dir_name, xlim=[0, 25000])
        # color_list = ["b", "g", "r", "c", "m", "y", "k"]*10
        
        # fit_list = []
        # time_idx_list = []
        # x_list = []
        # y_list = []
        # for group_idx in range(len(integraded_area_center_point_list)):
        #     if move_flg_list[group_idx]:
        #         # 中心点のxy座標の移動に対して、近似曲線を取得
        #         time_idxs = [time_idx for time_idx, point in enumerate(integraded_area_center_point_list[group_idx]) if len(point)>0]
        #         x = [point[0] for point in integraded_area_center_point_list[group_idx] if len(point)>0]
        #         y = [point[1] for point in integraded_area_center_point_list[group_idx] if len(point)>0]

        #         res = np.polyfit(x, y, 1)
        #         y_fit = np.poly1d(res)(x)

        #         if len(fit_list)==0:
        #             fit_list.append(res[0])
        #             time_idx_list = time_idxs
        #             x_list = x
        #             y_list = y
        #         else:
        #             if abs(fit_list[-1]-res[0])<1:
        #                 fit_list.append(res[0])
        #                 time_idx_list += time_idxs
        #                 x_list += x
        #                 y_list += y

        #         # ax.plot(x, y_fit, label=f"group{group_idx}", c=color_list[group_idx])

        # before_idx = None
        # no_data = []
        # new_time_idx_list = []
        # new_x_list = []
        # new_y_list = []
        # for idx, time_idx in enumerate(time_idx_list):
        #     if before_idx is not None:
        #         if time_idx-before_time_idx>1:
        #             x_step = (x_list[idx]-x_list[before_idx])/(time_idx-before_time_idx)
        #             y_step = (y_list[idx]-y_list[before_idx])/(time_idx-before_time_idx)

        #             for i in range(1, time_idx-before_time_idx):
        #                 no_data.append(time_idx)
        #                 new_time_idx_list.append(before_time_idx+i)
        #                 new_x_list.append(x_list[before_idx]+x_step*i)
        #                 new_y_list.append(y_list[before_idx]+y_step*i)
            
        #     new_time_idx_list.append(time_idx)
        #     new_x_list.append(x_list[idx])
        #     new_y_list.append(y_list[idx])
            
        #     before_idx = idx
        #     before_time_idx = time_idx

        # # 改めて処理を行う
        # new_integraded_area_points_list = []
        # new_integraded_area_center_point_list = []
        # for group_idx in range(1):
        #     new_integraded_area_points_list.append([])
        #     new_integraded_area_center_point_list.append([])
            
        #     for time_idx in range(len(integraded_area_points_list[0])):
        #         if time_idx in new_time_idx_list:
        #             idx = new_time_idx_list.index(time_idx)

        #             cloud_path = "/Users/kai/大学/小川研/LiDAR_step_length/20241011/pcd/pcd_01s/"+pcd_info_list.dir_name+"/"+str(time_idx+1)+".pcd"
        #             pcd_info_list.load_pcd_from_file(cloud_path)
                    
        #             cloud = pcd_info_list.cloud
        #             cloud_name = pcd_info_list.dir_name+"_"+str(time_idx+1)
        #             # 高さの補正
        #             points = np.array(cloud)
        #             points[:, 2] = points[:, 2] + 1300
        #             cloud = def_method.get_cloud(points)

        #             pcd_info_list.load_pcd_from_cloud(cloud)
        #             down_cloud = def_method.voxel_grid_filter(cloud)
        #             down_points = np.array(down_cloud)

        #             base_x = new_x_list[idx]
        #             base_y = new_y_list[idx]
        #             cloud_filtered = def_method.filter_area(cloud, base_x-250, base_x+250, base_y-250, base_y+250, 0, 1700)
        #             points_filtered = np.array(cloud_filtered)
        #             center_point = np.mean(points_filtered, axis=0)
                    
        #             new_integraded_area_points_list[group_idx].append(points_filtered)
        #             new_integraded_area_center_point_list[group_idx].append(center_point)

        #             try:
        #                 fig = plt.figure()
        #                 ax = fig.add_subplot(111, projection="3d")
        #                 ax.scatter(points_filtered[:, 0], points_filtered[:, 1], points_filtered[:, 1], s=1)
        #                 ax_set.set_ax(ax, cloud_name, zlim=[0, 1600], azim=0, elev=0)
        #                 # plt.show()
        #                 plt.close()
        #             except:
        #                 print(cloud_name)
        #                 plt.close()
        #         else:
        #             new_integraded_area_points_list[group_idx].append([])
        #             new_integraded_area_center_point_list[group_idx].append([])

        integraded_area_points_list = new_integraded_area_points_list
        integraded_area_center_point_list = new_integraded_area_center_point_list
        vectros_list = ori_method.get_vector(integraded_area_center_point_list)
        move_flg_list = ori_method.judge_move(vectros_list)

        # 全体の点群を表示
        if False:
            gif = create_gif.create_gif(create_flg=False)
            color_list = ["b", "g", "r", "c", "m", "y", "k", "w"]*10
            for time_idx in range(len(integraded_area_points_list[0])):
                fig = plt.figure(figsize=(10, 10))
                ax0 = fig.add_subplot(121, projection='3d')
                ax1 = fig.add_subplot(122)
                ax0 = ax_set.set_ax(ax0, title="time_idx:"+str(time_idx))
                ax1 = ax_set.set_ax(ax1)

                for group_idx in range(len(integraded_area_points_list)):
                    group_point = integraded_area_points_list[group_idx][time_idx]
                    group_center_point = integraded_area_center_point_list[group_idx][time_idx]
                    
                    if len(group_center_point)>0:
                        ax0.scatter(np.array(group_point)[:, 0], np.array(group_point)[:, 1], np.array(group_point)[:, 2], s=1, c=color_list[group_idx])
                        ax1.scatter(np.array(group_center_point)[0], np.array(group_center_point)[1], s=10, c=color_list[group_idx])

                plt.show()
                gif.save_fig(fig)
                plt.close()
            gif.create_gif(f"/Users/kai/大学/小川研/LIDAR_step_length/20241011/gif/pcd_{sec}s/{pcd_info_list.dir_name}.gif")


        # 重ね合わせた点群を作成
        # for group_idx in range(len(integraded_area_points_list)):
        #     if not move_flg_list[group_idx]:
        #         continue
        #     combined_points = None
        #     combined_center_points = None
        #     normalized_points = None
        #     combined_normalized_points = None
        #     for time_idx in range(len(integraded_area_points_list[0])):
        #         group_point = integraded_area_points_list[group_idx][time_idx]
        #         group_center_point = integraded_area_center_point_list[group_idx][time_idx]

        #         if len(group_center_point)>0:                    
        #             tmp_point = group_center_point.copy()
        #             tmp_point[2] = 0
        #             normalized_points = group_point - tmp_point
        #             if combined_points is None:
        #                 combined_points = group_point
        #                 combined_center_points = group_center_point
        #                 combined_normalized_points = normalized_points
        #             else:
        #                 combined_points = np.concatenate([combined_points, group_point], axis=0)
        #                 combined_center_points = np.concatenate([combined_center_points, group_center_point], axis=0)
        #                 combined_normalized_points = np.concatenate([combined_normalized_points, normalized_points], axis=0)
        
        
        # 各時刻の点群を表示、身長データをまとめる
        # color_list = ["b", "g", "c", "m", "y", "k"]*10
        # heights_lists = {}
        # centers = {}
        # gif = create_gif.create_gif(create_flg=False)
        # for group_idx in range(len(integraded_area_points_list)):
        #     if move_flg_list[group_idx]:
        #         heights_list = []
        #         new_heights_list = []
        #         for time_idx in range(len(integraded_area_points_list[0])):
        #             group_point = integraded_area_points_list[group_idx][time_idx]
        #             group_center_point = integraded_area_center_point_list[group_idx][time_idx]
        #             new_group_point = new_integraded_area_points_list[0][time_idx]
        #             new_group_center_point = new_integraded_area_center_point_list[0][time_idx]

        #             if len(group_center_point)>0:
        #                 tmp_point = group_center_point.copy()
        #                 tmp_point[2] = 0
        #                 normalized_points = group_point - tmp_point
        #                 tmp_point = group_point.copy()
        #                 tmp_point = tmp_point[(tmp_point[:, 2] > 1200) & (tmp_point[:, 2] < 1400)]
        #                 height = ori_method.get_height(group_point, 40)
        #                 height = ori_method.get_bentchmark(tmp_point, 0, 100)

        #                 new_tmp_point = new_group_center_point.copy()
        #                 new_tmp_point[2] = 0
        #                 new_normalized_points = new_group_point - new_tmp_point
        #                 new_tmp_point = new_group_point.copy()
        #                 new_tmp_point = new_tmp_point[(new_tmp_point[:, 2] > 1200) & (new_tmp_point[:, 2] < 1400)]
        #                 new_height = ori_method.get_height(new_group_point, 40)
        #                 new_height = ori_method.get_bentchmark(new_tmp_point, 0, 100)

        #                 fig = plt.figure(figsize=(10, 10))
        #                 ax0 = fig.add_subplot(121, projection='3d')
        #                 ax1 = fig.add_subplot(122, projection='3d')
        #                 # ax2 = fig.add_subplot(224, projection='3d')
        #                 ax0 = ax_set.set_ax(ax0, title=pcd_info_list.dir_name+" time_idx:"+str(time_idx), xlim=[0, 30000], zlim=[-100, 1900], azim=240, elev=90)
        #                 ax1 = ax_set.set_ax(ax1, title="point_num:"+str(len(normalized_points)), xlim=[-250, 250], ylim=[-250, 250], zlim=[0, 2000])
        #                 # ax2 = ax_set.set_ax(ax2, title="point_num:"+str(len(new_normalized_points)), xlim=[-250, 250], ylim=[-250, 250], zlim=[0, 2000])
        #                 # 10%毎に高さの平均を取得
        #                 heights = []
        #                 new_heights = []
        #                 for i in range(0, 100, 10):
        #                     heights.append(ori_method.get_bentchmark(group_point, i, i+10))
        #                     new_heights.append(ori_method.get_bentchmark(new_group_point, i, i+10))
        #                 heights_list.append(heights)
        #                 new_heights_list.append(new_heights)

        #                 ax0.scatter(np.array(group_point)[:, 0], np.array(group_point)[:, 1], np.array(group_point)[:, 2], s=1, c="r")
        #                 # ax0.scatter(np.array(group_center_point)[0], np.array(group_center_point)[1], height, s=10, c="r")
                        
        #                 if group_idx not in centers.keys():
        #                     centers[group_idx] = [group_center_point]
        #                 else:
        #                     centers[group_idx].append(group_center_point)
        #                 for key, value in centers.items():
        #                     ax0.scatter(np.array(value)[:, 0], np.array(value)[:, 1], np.array(value)[:, 2], s=10, c=color_list[key])
        #                 # ax0.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], np.array(centers)[:, 2], s=10, c=color_list[group_idx])

        #                 ax1.scatter(np.array(normalized_points)[:, 0], np.array(normalized_points)[:, 1], np.array(normalized_points)[:, 2], s=1, c=color_list[group_idx])
        #                 ax1.scatter(np.zeros(10), np.zeros(10), np.array(heights), s=10, c="r")

        #                 # ax2.scatter(np.array(new_normalized_points)[:, 0], np.array(new_normalized_points)[:, 1], np.array(new_normalized_points)[:, 2], s=1, c=color_list[group_idx])
        #                 # ax2.scatter(np.zeros(10), np.zeros(10), np.array(new_heights), s=10, c="r")
                        
        #                 gif.save_fig(fig)
        #                 # plt.show()
        #             else:
        #                 heights_list.append([])
        #             plt.close()
        #         heights_lists[group_idx] = heights_list
        # gif.create_gif(f"/Users/kai/大学/小川研/LIDAR_step_length/20241011/gif/pcd_{sec}s/{pcd_info_list.dir_name}_after_method_move.gif")

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # xmin, xmax = 100, 0
        # ymin, ymax = 2000, 0
        # for group_idx, heights_list in heights_lists.items():
        #     for i in range(10):
        #         xs, ys = [], []
        #         for time_idx, heights in enumerate(heights_list):
        #             if len(heights)>0:
        #                 xs.append(time_idx)
        #                 ys.append(heights[i])
        #         ax.plot(xs, ys, label=f"{i*10}-{(i+1)*10}")
                
        #         xmin = min(xmin, min(xs))
        #         xmax = max(xmax, max(xs))
        #         ymin = min(ymin, min(ys))
        #         ymax = max(ymax, max(ys))
                
        #         # 極大値のみを取得 : (差分 * 1つシフトした差分が負) & (差分が正)、という条件
        #         sdiff = np.diff(ys)
        #         sdiff_sign = ((sdiff[:-1] * sdiff[1:]) < 0) & (sdiff[:-1] > 0)
        #         ax.plot(np.array(xs[1:-1])[sdiff_sign], np.array(ys[1:-1])[sdiff_sign], "o")

        #         print(np.array(xs[1:-1])[sdiff_sign])
            
        #     ax.legend()
        # title = f"{pcd_info_list.dir_name}, height"
        # ax = ax_set.set_ax(ax, title=title, xlabel="time", ylabel="height", xlim=[xmin-1, xmax+1], ylim=[ymin-100, ymax+100])
        # plt.pause(0.1)
        # plt.close()


        # # 重ね合わせた点群から身長を推測
        # height = ori_method.get_bentchmark(combined_normalized_points, 100, 97)
        # # 重ね合わせた点群を回転させて表示
        # for azim_num in range(0, 360, 30):
        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax = ax_set.set_ax(ax, xlim=[-250, 250], ylim=[-250, 250], zlim=[0, 2000], title=f"height:{height}mm, azim:{azim_num}", azim=azim_num)
        #     ax.set_box_aspect([1, 1, 4])

        #     # z軸の値を正規化
        #     plot_points = combined_normalized_points
        #     ax.scatter(np.array(plot_points)[:, 0], np.array(plot_points)[:, 1], np.array(plot_points)[:, 2], s=1, c=color_list[group_idx])
        #     plt.pause(0.1)
        #     plt.close()



        # 速度・加速度を取得
        step_length_accelaration_list = []
        acc_time_idx_list_list = []
        time_idx_list_list = []
        speed_list_list = []
        acceleration_list_list = []
        speed_spot_x_list = []
        speed_spot_y_list = []
        acceleration_spot_x_list = []
        acceleration_spot_y_list = []

        for group_idx in range(len(move_flg_list)):
            move_flg = move_flg_list[group_idx]
            if move_flg:
                # 速度を取得
                speed_list = []
                time_idx_list = []
                for time_idx in range(len(vectros_list[0])-1):
                    before_vector = vectros_list[group_idx][time_idx]
                    after_vector = vectros_list[group_idx][time_idx+1]

                    if len(before_vector)>0 and len(after_vector)>0:
                        distance = np.linalg.norm(after_vector - before_vector)
                        speed = (distance/1000) / 0.1 * 10

                        speed_list.append(speed)
                        time_idx_list.append(time_idx)

                # 加速度を取得
                acceleration_list = []
                for speed_idx in range(len(speed_list)-1):
                    before_speed = speed_list[speed_idx]
                    after_speed = speed_list[speed_idx+1]
                    diff_time = (time_idx_list[speed_idx+1] - time_idx_list[speed_idx]) / 10

                    acceleration = (after_speed - before_speed) / diff_time
                    acceleration_list.append(acceleration)

                # 極大値のみを取得 : (差分 * 1つシフトした差分が負) & (差分が正)、という条件
                sdiff = np.diff(speed_list)
                sdiff_sign = ((sdiff[:-1] * sdiff[1:]) < 0) & (sdiff[:-1] > 0)
                adiff = np.diff(acceleration_list)
                adiff_sign = ((adiff[:-1] * adiff[1:]) < 0) & (adiff[:-1] > 0)


                time_idx_list_list += time_idx_list
                acc_time_idx_list_list += time_idx_list[1:]
                speed_list_list += speed_list
                acceleration_list_list += acceleration_list
                speed_spot_x_list += np.array(time_idx_list[1:-1])[sdiff_sign].tolist()
                speed_spot_y_list += np.array(speed_list[1:-1])[sdiff_sign].tolist()
                acceleration_spot_x_list += np.array(time_idx_list[2:-1])[adiff_sign].tolist()
                acceleration_spot_y_list += np.array(acceleration_list[1:-1])[adiff_sign].tolist()

                # 速度・加速度のグラフを表示
                fig = plt.figure(figsize=(10, 10))
                ax0 = fig.add_subplot(211)
                ax1 = fig.add_subplot(212)
                ax0.plot(time_idx_list, speed_list)
                ax0.plot(time_idx_list, speed_list, "o")
                ax0.plot(np.array(time_idx_list[1:-1])[sdiff_sign], np.array(speed_list[1:-1])[sdiff_sign], "o")
                ax1.plot(time_idx_list[1:], acceleration_list)
                ax1.plot(time_idx_list[1:], acceleration_list, "o")
                ax1.plot(np.array(time_idx_list[2:-1])[adiff_sign], np.array(acceleration_list[1:-1])[adiff_sign], "o")

                title = f"{pcd_info_list.dir_name}"
                ax0 = ax_set.set_ax(ax0, title=pcd_info_list.dir_name+"-speed", xlabel="time", ylabel="speed", xlim=[time_idx_list[0]-1, time_idx_list[-1]+1], ylim=[min(speed_list)-1, max(speed_list)+1])
                ax1 = ax_set.set_ax(ax1, title=pcd_info_list.dir_name+"-acceleration", xlabel="time", ylabel="acceleration", xlim=[time_idx_list[0]-1, time_idx_list[-1]+1], ylim=[min(acceleration_list)-1, max(acceleration_list)+1], is_box_aspect=False)
                # plt.pause(0.1)
                plt.close()

                print("加速度を用いた歩幅の推定")
                # time_idx_list[2:-1]の部分配列を取得
                partial_time_idx_list = np.array(time_idx_list[2:-1])
                # adiff_signを使用してフィルタリング
                filtered_indices = partial_time_idx_list[adiff_sign]
                # フィルタリングされたインデックスを使用して要素にアクセス
                points = []
                for idx in filtered_indices:
                    points.append(integraded_area_center_point_list[group_idx][idx])

                # 歩幅の推測値を取得
                for i in range(len(points)-1):
                    step_length = ori_method.calc_points_distance(points[i], points[i+1])
                    print(f"step_length : {step_length}")
                    step_length_accelaration_list.append(step_length)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax.plot(time_idx_list_list, speed_list_list)
        ax.plot(time_idx_list_list, speed_list_list, "o")
        ax.plot(speed_spot_x_list, speed_spot_y_list, "o", c="g")
        ax2 = fig.add_subplot(212)
        ax2.plot(acc_time_idx_list_list, acceleration_list_list)
        ax2.plot(acc_time_idx_list_list, acceleration_list_list, "o")
        ax2.plot(acceleration_spot_x_list, acceleration_spot_y_list, "o", c="g")
        ax = ax_set.set_ax(ax, title=pcd_info_list.dir_name+"-speed", xlabel="time", ylabel="speed", xlim=[time_idx_list[0]-1, time_idx_list[-1]+1], ylim=[min(speed_list)-1, max(speed_list)+1])
        ax2 = ax_set.set_ax(ax2, title=pcd_info_list.dir_name+"-acceleration", xlabel="time", ylabel="acceleration", xlim=[time_idx_list[0]-1, time_idx_list[-1]+1], ylim=[min(acceleration_list)-1, max(acceleration_list)+1], is_box_aspect=False)

        # plt.show()
        plt.close()


        # 中心点の高さ情報の変化を取得
        step_length_height_list = []
        for group_idx in range(len(move_flg_list)):
            move_flg = move_flg_list[group_idx]
            if move_flg:
                height_list = []
                time_idx_list = []
                for time_idx in range(len(integraded_area_center_point_list[0])):
                    center_point = integraded_area_center_point_list[group_idx][time_idx]
                    if len(center_point)>0:
                        tmp_point = integraded_area_points_list[group_idx][time_idx].copy()
                        tmp_point = tmp_point[np.where(tmp_point[:, 2]>1200) and np.where(tmp_point[:, 2]<1400)]
                        height = ori_method.get_height(tmp_point, 100)/10
                        height_list.append(height)
                        time_idx_list.append(time_idx)

                # 極大値のみを取得 : (差分 * 1つシフトした差分が負) & (差分が正)、という条件
                sdiff = np.diff(height_list)
                sdiff_sign = ((sdiff[:-1] * sdiff[1:]) < 0) & (sdiff[:-1] > 0)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.plot(time_idx_list, height_list)
                ax.plot(time_idx_list, height_list, "o")
                ax.plot(np.array(time_idx_list[1:-1])[sdiff_sign], np.array(height_list[1:-1])[sdiff_sign], "o")
                ax_set.set_ax(ax, title=pcd_info_list.dir_name+" height", xlabel="time", ylabel="height (cm)", xlim=[time_idx_list[0]-1, time_idx_list[-1]+1], ylim=[min(height_list)-10, max(height_list)+10])

                plt.pause(0.1)
                plt.close()

                print("高さの変化を用いた歩幅の推定")
                # time_idx_list[2:-1]の部分配列を取得
                partial_time_idx_list = np.array(time_idx_list[1:-1])
                # sdiff_signを使用してフィルタリング
                filtered_indices = partial_time_idx_list[sdiff_sign]
                # フィルタリングされたインデックスを使用して要素にアクセス
                points = []
                for idx in filtered_indices:
                    points.append(integraded_area_center_point_list[group_idx][idx])
                
                # 歩幅の推測値を取得
                for i in range(len(points)-1):
                    point1 = np.array([points[i][0], points[i][1], 0])
                    point2 = np.array([points[i+1][0], points[i+1][1], 0])
                    step_length = ori_method.calc_points_distance(point1, point2)
                    print(f"step_length : {step_length}")
                    step_length_height_list.append(step_length)

        # 箱ひげ図を表示
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.boxplot([step_length_accelaration_list, step_length_height_list], labels=["acceleration", "height"])
        # ax_set.set_ax(ax, title=pcd_info_list.dir_name, ylabel="step_length (cm)")
        ax.set_title(pcd_info_list.dir_name)
        ax.set_ylim(0, 1100)
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax.hist(step_length_accelaration_list, bins=55, range=(0, 1100), label="acceleration")
        ax2 = fig.add_subplot(212)
        ax2.hist(step_length_height_list, bins=55, range=(200, 1100), label="height")
        ax.set_title(pcd_info_list.dir_name+"\nacceleration")
        ax2.set_title("height")

        plt.show()
        plt.close()
