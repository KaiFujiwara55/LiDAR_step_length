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

sec_list = ["01"]
for sec in sec_list:
    dir_path = f"/Users/kai/大学/小川研/LiDAR_step_length/20241028/"
    dirs = glob.glob(f"{dir_path}pcd_{sec}s/*")

    # ノイズ除去のクラスをインスタンス化
    def_method = default_method.cloud_method()
    ori_method = original_method.cloud_method()
    for dir in dirs:
        if "repeat" in dir:
            continue
        # pcdファイルの情報を取得
        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(dir)
        # plot用のaxのdefault設定
        ax_set = plot.set_plot()
        ax_set.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]), ylim=(pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]), zlim=(pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]), azim=150)

        print(f"処理開始 : {pcd_info_list.dir_name}_{sec}s")
        # 処理結果を読み込み
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_points_list/{pcd_info_list.dir_name}_{sec}s"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/tmp_folder/time_area_center_point_list/{pcd_info_list.dir_name}_{sec}s"
        time_area_points_list, time_area_center_point_list = ori_method.load_original_data(area_path, center_path)

        # 点群をグループ化
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5)

        # 中心点の軌跡から新たにグループを作成
        sec_2 = 0.1
        cloud_folder_path = dir_path+"pcd_"+str(sec_2).replace(".", "")+"s/"+pcd_info_list.dir_name
        integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, is_incline=False)

        move_flg_list = ori_method.judge_move(integraded_area_center_point_list)

        # 速度・加速度を取得
        step_length_accelaration_list = []
        speed_dic = {}
        time_idx_dic = {}
        for group_idx in range(len(move_flg_list)):
            move_flg = move_flg_list[group_idx]
            if move_flg:
                # 速度を取得
                speed_list = []
                time_idx_list = []
                first = True
                for time_idx in range(len(integraded_area_points_list[group_idx])-1):
                    if len(integraded_area_points_list[group_idx][time_idx])>0 and len(integraded_area_points_list[group_idx][time_idx+1])>0:    
                        before_bentch_point = ori_method.get_bentchmark(integraded_area_points_list[group_idx][time_idx], height=[600, 1100])
                        after_bentch_point = ori_method.get_bentchmark(integraded_area_points_list[group_idx][time_idx+1], height=[600, 1100])

                        if (len(before_bentch_point)==0 or len(after_bentch_point)==0):
                            continue

                        before_bentch_point[2] = 0
                        after_bentch_point[2] = 0


                        distance = ori_method.calc_points_distance(before_bentch_point, after_bentch_point)
                        speed = (distance) / (sec_2 * 10)

                        speed_list.append(speed)
                        time_idx_list.append(time_idx)

                # 速度の移動平均を取得
                for window_size in range(1, 10):
                    window_dic = {"0.1": 4, "0.05":6}
                    window = window_dic[str(sec_2)]
                    window = window_size
                    speed_conv_list = np.convolve(speed_list, np.ones(window)/window, mode='same')
                    
                    # ピークの取得
                    # ピーク間の距離は400~600mmであることを考慮




                    # 速度の移動平均のピークを取得
                    sdiff = np.diff(speed_conv_list)
                    sdiff_sign = ((sdiff[:-1] * sdiff[1:]) < 0) & (sdiff[:-1] > 0)
                    sdiff_sign_2 = ((sdiff[:-1] * sdiff[1:]) < 0) & (sdiff[:-1] < 0)
                    peak_time_idx_list = np.array(time_idx_list[1:-1])[sdiff_sign]
                    peak_time_idx_list_2 = np.array(time_idx_list[1:-1])[sdiff_sign_2]
                    peak_speed = speed_conv_list[1:-1][sdiff_sign]
                    peak_speed_2 = speed_conv_list[1:-1][sdiff_sign_2]
                    
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax1.plot(time_idx_list, speed_list, label="speed", c="blue")
                    ax1.plot(time_idx_list, speed_conv_list, label="conv", c="red")
                    ax1.plot(peak_time_idx_list, peak_speed, "o", c="green")
                    ax1.plot(peak_time_idx_list_2, peak_speed_2, "o", c="red")
                    
                    title = f"speed, window={window}"
                    ax1 = ax_set.set_ax(ax1, title=title, xlabel="time", ylabel="speed", xlim=[0, max(time_idx_list)], ylim=[min(speed_conv_list)-10, max(speed_conv_list)+10])

                    ax1.legend()
                    # plt.show()
                    # plt.close()

                    # 歩幅を取得
                    step_length_list = []
                    # fig = plt.figure(figsize=(10, 5))
                    # ax = fig.add_subplot(111)
                    # title = f"{pcd_info_list.dir_name}_{sec}s_step, sampling={sec_2}s"
                    # ax = ax_set.set_ax(ax, title=title, xlim=[1000, 11000], ylim=[-500, 500])
                    for idx in range(len(peak_time_idx_list)-1):
                        before_time_idx = peak_time_idx_list[idx]
                        after_time_idx = peak_time_idx_list[idx+1]
                        
                        before_bentch_point = ori_method.get_bentchmark(integraded_area_points_list[group_idx][before_time_idx])
                        after_bentch_point = ori_method.get_bentchmark(integraded_area_points_list[group_idx][after_time_idx])

                        if (len(before_bentch_point)==0 or len(after_bentch_point)==0):
                            continue
                        before_bentch_point[2] = 0
                        after_bentch_point[2] = 0
                        
                        # ax.scatter(before_bentch_point[0], before_bentch_point[1], c="red", s=5)

                        step_length = ori_method.calc_points_distance(before_bentch_point, after_bentch_point)
                        step_length_list.append(step_length)

                    # plt.show()
                    # plt.close()

                    # 推定歩幅のヒストグラムを表示
                    # fig = plt.figure(figsize=(10, 10))
                    ax2 = fig.add_subplot(122)
                    ax2.hist(step_length_list, bins=55, range=(0, 1100))
                    title = f"step_hist"
                    ax2.set_title(title)

                    fig.suptitle(f"{pcd_info_list.dir_name}_{sec}s sampling={sec_2}s, window={window}", y=0)
                    plt.show()
                    plt.close()
                    
                    mae = np.sum(np.abs(np.array(step_length_list)-500))/len(step_length_list)
                    print(window, mae)
