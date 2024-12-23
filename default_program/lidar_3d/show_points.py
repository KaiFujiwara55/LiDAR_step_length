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

sec_list = ["01"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241218/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/3d/*")

    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if not "cose_5_1_y" in dir:
            continue
        
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=180)
        

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")
        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/{sec}s/{pcd_info_list.dir_name}"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/{sec}s/{pcd_info_list.dir_name}"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

        # for time_idx in range(len(time_area_points_list)):
        #     if len(time_area_points_list[time_idx]) == 0:
        #         continue
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax = ax_set.set_ax(ax)
        #     for group_idx in range(len(time_area_points_list[time_idx])):
        #         points = time_area_points_list[time_idx][group_idx]
        #         if len(points) == 0:
        #             continue
        #         points = np.array(points)

        #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue")
        #     plt.show()
        #     plt.close()

        # 点群をグループ化
        #######
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5, distance_threshold=250)

        # 中心点の軌跡から新たにグループを作成
        sec_2 = 0.1
        cloud_folder_path = dir_path+"pcd_"+str(sec_2).replace(".", "")+"s/3d/"+pcd_info_list.dir_name
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, judge_move_threshold=750, is_incline=False)
        move_flg_list = ori_method.judge_move(ori_method.get_vector(integraded_area_center_point_list), threshold=500)
        print(move_flg_list)
        #########

        gif = create_gif.create_gif(False)
        for group_idx in range(len(integraded_area_points_list)):
            # if move_flg_list[group_idx]:
            if True:
                for time_idx in range(len(integraded_area_points_list[0])):
                    points = integraded_area_points_list[group_idx][time_idx]
                    center_point = integraded_area_center_point_list[group_idx][time_idx]
                    if len(center_point) > 0:
                        fig = plt.figure()
                        ax1 = fig.add_subplot(121, projection='3d')
                        ax1 = ax_set.set_ax(ax1, title=f"{pcd_info_list.dir_name}_{sec}s, group_idx:{group_idx}, time_idx:{time_idx}", xlim=[0, 11000], ylim=[-5000, 5000], zlim=[0, 2000])
                        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue")
                        ax1.scatter(center_point[0], center_point[1], center_point[2], s=10, c="red")

                        plt.show()
                        gif.save_fig(fig)
                        plt.close()
        output_path = "/Users/kai/大学/小川研/LiDAR_step_length/gif/tmp/"+pcd_info_list.dir_name+".gif"
        gif.create_gif(output_path, duration=0.1)
        continue
        # 時系列ごとでの点群表示
        move_flg_list = ori_method.judge_move(integraded_area_center_point_list)
        print(move_flg_list)
        for group_idx in range(len(integraded_area_points_list)):
            if move_flg_list[group_idx]:
                all_points = None
                normarized_all_points = None

                a_list = []
                time_idx_list = []
                for time_idx in range(len(integraded_area_points_list[group_idx])):
                    points = integraded_area_points_list[group_idx][time_idx]
                    if len(points) > 0:
                        fig = plt.figure()
                        points = np.array(points)
                        center_point = integraded_area_center_point_list[group_idx][time_idx]
                        if all_points is None:
                            all_points = points.copy()

                            normarized_all_points = points.copy()
                            normarized_all_points[:, 0] -= center_point[0]
                            normarized_all_points[:, 1] -= center_point[1]
                        else:
                            all_points = np.concatenate([all_points, points], axis=0)

                            normarized_points = points.copy()
                            normarized_points[:, 0] -= center_point[0]
                            normarized_points[:, 1] -= center_point[1]
                            normarized_all_points = np.concatenate([normarized_all_points, normarized_points], axis=0)
                        
                        if True:
                            ax1 = fig.add_subplot(121, projection='3d')
                            ax1 = ax_set.set_ax(ax1)
                            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue")

                        normarized_points = points.copy()
                        normarized_points[:, 0] -= center_point[0]
                        normarized_points[:, 1] -= center_point[1]

                        lower = 600
                        upper = 1100
                        if False:    
                            ax2 = fig.add_subplot(121, projection='3d')
                            ax2 = ax_set.set_ax(ax2, xlim=[-250, 250], ylim=[-250, 250], zlim=[0, 2000])
                            ax2.scatter(normarized_points[:, 0], normarized_points[:, 1], normarized_points[:, 2], s=1, c="blue")
                        if False:
                            ax3 = fig.add_subplot(122, projection='3d')
                            ax3 = ax_set.set_ax(ax3, title=f"{lower}~{upper}", xlim=[-250, 250], ylim=[-250, 250], zlim=[lower, upper], azim=30, elev=90)
                            tmp_points = normarized_points[(normarized_points[:, 2] >= lower) & (normarized_points[:, 2] < upper)]
                            ax3.scatter(tmp_points[:, 0], tmp_points[:, 1], tmp_points[:, 2], s=1, c="blue")

                            # 近似曲線を作成
                            tmp_points = normarized_points[(normarized_points[:, 0]>-250) & (normarized_points[:, 0]<250) & (normarized_points[:, 1]>-250) & (normarized_points[:, 1]<250) & (normarized_points[:, 2] >= lower) & (normarized_points[:, 2] < upper)]
                            xy_poly = np.polyfit(tmp_points[:, 1], tmp_points[:, 0], 1)
                            y = np.linspace(-250, 250, 10)
                            x = np.poly1d(xy_poly)(y)

                            print(f"y={xy_poly[0]}x+{xy_poly[1]}")
                            ax3.plot(x, y, color="red")
                        
                        if False:
                            tmp_points = normarized_points[(normarized_points[:, 0]>-250) & (normarized_points[:, 0]<250) & (normarized_points[:, 1]>-250) & (normarized_points[:, 1]<250) & (normarized_points[:, 2] >= lower) & (normarized_points[:, 2] < upper)]
                            xy_poly = np.polyfit(tmp_points[:, 1], tmp_points[:, 0], 1)
                            a_list.append(xy_poly[0])
                            time_idx_list.append(time_idx)
                            
                            ax4 = fig.add_subplot(121, projection='3d')
                            ax4 = ax_set.set_ax(ax4)
                            if xy_poly[0] > 0:
                                ax4.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue")
                            else:
                                ax4.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="green")
                            
                            ax5 = fig.add_subplot(122, projection='3d')
                            ax5 = ax_set.set_ax(ax5, title=f"{lower}~{upper}", xlim=[-250, 250], ylim=[-250, 250], zlim=[lower, upper], azim=30, elev=90)
                            tmp_points = normarized_points[(normarized_points[:, 2] >= lower) & (normarized_points[:, 2] < upper)]
                            ax5.scatter(tmp_points[:, 0], tmp_points[:, 1], tmp_points[:, 2], s=1, c="blue")

                            y = np.linspace(-250, 250, 10)
                            x = np.poly1d(xy_poly)(y)
                            ax5.plot(x, y, color="red")

                        plt.suptitle(f"{pcd_info_list.dir_name}_{sec}s, group_idx:{group_idx}, time_idx:{time_idx}", y=0)
                        # plt.show()
                        plt.close()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(time_idx_list, a_list)
                fig.suptitle(f"{pcd_info_list.dir_name}_{sec}s, group_idx:{group_idx}", y=0)
                plt.show()
                plt.close()


                # 全ての時間軸の点群を重ね合わせして輪切りにして表示
                if False:
                    down_sampled_cloud = def_method.voxel_grid_filter(def_method.get_cloud(all_points), leaf_size=(100, 100, 100))
                    down_sampled_points = def_method.get_points(down_sampled_cloud)
                    for i in range(-500, 2000, 100):
                        fig = plt.figure()
                        ax1 = fig.add_subplot(121, projection='3d')
                        ax1 = ax_set.set_ax(ax1, title=f"{pcd_info_list.dir_name}", xlim=[-250, 250], ylim=[-250, 250], zlim=[-500, 2000])
                        ax1.scatter(down_sampled_points[:, 0], down_sampled_points[:, 1], down_sampled_points[:, 2], s=1, c="blue")

                        tmp_points = all_points[(all_points[:, 2] >= i) & (all_points[:, 2] < i+100)]
                        ax2 = fig.add_subplot(122, projection='3d')
                        ax2 = ax_set.set_ax(ax2, title=f"{i}~{i+100}, poinsts_num:{len(tmp_points)}", xlim=[-250, 250], ylim=[-250, 250], zlim=[i, i+100])
                        ax2.scatter(tmp_points[:, 0], tmp_points[:, 1], tmp_points[:, 2], s=1, c="blue")

                        if len(tmp_points) > 0:
                            plt.show()
                        plt.close()
