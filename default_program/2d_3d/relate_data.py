import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot
from default_program.class_method import create_gif

from default_program.lidar_3d import get_step_length_chilt_func
from default_program.lidar_3d import get_step_length_chilt_func_show
from default_program.lidar_2d import get_step_length_half_func
from default_program.lidar_2d import get_step_length_article

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

set_ax_2d = plot.set_plot()
set_ax_3d = plot.set_plot()

sec_2d = 0.025
sec_3d_1 = 0.1
sec_3d_2 = 0.1

# データの読み込み
dir_list = glob.glob("/Users/kai/大学/小川研/Lidar_step_length/20241218/pcd_"+str(sec_3d_1).replace(".", "")+"s/3d/*")
dir_list = sorted(dir_list)

# 結果を保存するdf
length_2d_df = pd.DataFrame(columns=["course_1", "course_2", "course_3", "course_4", "course_5", "course_6"], index=["0_f", "0_t", "0_y", "1_f", "1_t", "1_y"])
length_3d_df = pd.DataFrame(columns=["course_1", "course_2", "course_3", "course_4", "course_5", "course_6"], index=["0_f", "0_t", "0_y", "1_f", "1_t", "1_y"])
length_diff_df = pd.DataFrame(columns=["course_1", "course_2", "course_3", "course_4", "course_5", "course_6"], index=["0_f", "0_t", "0_y", "1_f", "1_t", "1_y"])
step_num_2d_df = pd.DataFrame(columns=["course_1", "course_2", "course_3", "course_4", "course_5", "course_6"], index=["0_f", "0_t", "0_y", "1_f", "1_t", "1_y"])
step_num_3d_df = pd.DataFrame(columns=["course_1", "course_2", "course_3", "course_4", "course_5", "course_6"], index=["0_f", "0_t", "0_y", "1_f", "1_t", "1_y"])
step_num_diff_df = pd.DataFrame(columns=["course_1", "course_2", "course_3", "course_4", "course_5", "course_6"], index=["0_f", "0_t", "0_y", "1_f", "1_t", "1_y"])

cose_replace_mapping = {"cose_2":"course_1", "cose_3":"course_2","cose_5":"course_3","cose_6":"course_4","cose_7":"course_5","cose_8":"course_6"}
name_replace_mapping = {"0_f":"subjectA_0", "0_t":"subjectB_0", "0_y":"subjectC_0", "1_f":"subjectA_1", "1_t":"subjectB_1", "1_y":"subjectC_1"}
for dir in dir_list:
    if "cose_1" in dir or "cose_4" in dir:
        continue
    if "nothing" in dir or "obstacle" in dir:
        continue
    dir_2d = dir.replace("3d", "2d").replace(str(sec_3d_1).replace(".", ""), str(sec_2d).replace(".", ""))
    dir_3d = dir.replace("2d", "3d").replace(str(sec_2d).replace(".", ""), str(sec_3d_1).replace(".", ""))
    ####
    dir_3d = dir_3d.replace("1204", "1212")
    ####
    print(dir_3d)

    pcd_info_2d = get_pcd_information.get_pcd_information()
    pcd_info_2d.load_pcd_dir(dir_2d)
    pcd_info_3d = get_pcd_information.get_pcd_information()
    pcd_info_3d.load_pcd_dir(dir_3d)

    set_ax_2d.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_2d.get_all_min()[0], pcd_info_2d.get_all_max()[0]), ylim=(pcd_info_2d.get_all_min()[1], pcd_info_2d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)
    set_ax_3d.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_3d.get_all_min()[0], pcd_info_3d.get_all_max()[0]), ylim=(pcd_info_3d.get_all_min()[1], pcd_info_3d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)

    # 2dLidarの設置場所を設定
    # x, y, z = (mm, mm, bool)
    if "cose_1" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = 200/np.sqrt(2), 200/np.sqrt(2), False
    elif "cose_2" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = 3500/np.sqrt(2), 3500/np.sqrt(2), False
    elif "cose_3" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = (7000-200)/np.sqrt(2), (7000-200)/np.sqrt(2), False
    elif "cose_4" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2)+200/np.sqrt(2), 7000/np.sqrt(2)-200/np.sqrt(2), True
    elif "cose_5" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2)+3500/np.sqrt(2), 7000/np.sqrt(2)-3500/np.sqrt(2), True
    elif "cose_6" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2)+7000/np.sqrt(2), 7000/np.sqrt(2)-7000/np.sqrt(2), True
    elif "cose_7" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = 7000/np.sqrt(2), 7000/np.sqrt(2), True
    elif "cose_8" in dir:
        x_2d_lidar, y_2d_lidar, is_inverse = -100, 0, False

    try:
        # height_list, peak_list, step_length_list_3d = get_step_length_chilt_func.get_step(sec_3d_1, sec_3d_2, dir_3d, plt_flg=True)
        peak_list, step_length_list_3d, start_point_list, end_point_list = get_step_length_chilt_func_show.get_step(sec_3d_1, sec_3d_2, dir_3d, distance_threshold=250, judge_move_threshold=750, plt_flg=False)
        cross_points = get_step_length_article.get_step(0.025, dir_2d)

        # 3d
        peak_time_idx_3d = {}
        peak_points_3d = {}
        for group_idx, group_peak in enumerate(peak_list):
            peak_time_idx_3d[group_idx] = np.sort(np.array([x[0] for x in group_peak]))
            peak_points_3d[group_idx] = np.array([x[1] for x in group_peak])[np.argsort(np.array([x[0] for x in group_peak]))]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = set_ax_2d.set_ax(ax, title=pcd_info_2d.dir_name, xlim=[0, 10000], ylim=[-5000, 5000])
        # 補助線をひく
        square_x = np.linspace(0, 7000/np.sqrt(2), 100)
        y_square_down = np.tan(np.radians(-45))*square_x
        y_square_up = np.tan(np.radians(45))*square_x
        ax.plot(square_x, y_square_down, c="yellow", label="room_area")
        ax.plot(square_x, y_square_up, c="yellow")
        square_x = np.linspace(7000/np.sqrt(2), 7000/np.sqrt(2)*2, 100)
        y_square_down = np.tan(np.radians(-45))*square_x+7000/np.sqrt(2)*2
        y_square_up = np.tan(np.radians(45))*square_x-7000/np.sqrt(2)*2
        ax.plot(square_x, y_square_down, c="yellow")
        ax.plot(square_x, y_square_up, c="yellow")
        x = np.linspace(0, 12000, 100)
        y_down = np.tan(np.radians(-35.2))*x
        y_up = np.tan(np.radians(35.2))*x
        ax.plot(x, y_down, c="black", label="avia_range")
        ax.plot(x, y_up, c="black")

        # 3dのピーク点の線形近似を行う
        cofficient = np.polyfit(peak_points_3d[0][:, 0], peak_points_3d[0][:, 1], 1)
        x = np.linspace(0, 10000, 100)
        y = cofficient[0]*x + cofficient[1]
        collect_radian_2d = np.arctan(cofficient[0])

        # LiDARの設置場所より補正を行う
        collected_cross_points = np.array([[x[1], 0, 0] for x in cross_points])
        # 2dLidarの設置場所によって、反転させるかを分ける
        if is_inverse:
            if "cose_7" in pcd_info_2d.dir_name:
                collected_cross_points = def_method.rotate_points(collected_cross_points, theta_z=3*np.pi/2)
            else:
                collected_cross_points = def_method.rotate_points(collected_cross_points, theta_z=collect_radian_2d+np.pi)
        else:
            collected_cross_points = def_method.rotate_points(collected_cross_points, theta_z=collect_radian_2d)
        collected_cross_points += np.array([x_2d_lidar, y_2d_lidar, 0])

        # room_areaとavia_rangeの範囲内の点のみに絞る
        min_x = np.min(np.concatenate([np.array(start_point_list)[:, 0], np.array(end_point_list)[:, 0]]))
        max_x = np.max(np.concatenate([np.array(start_point_list)[:, 0], np.array(end_point_list)[:, 0]]))
        min_y = np.min(np.concatenate([np.array(start_point_list)[:, 1], np.array(end_point_list)[:, 1]]))
        max_y = np.max(np.concatenate([np.array(start_point_list)[:, 1], np.array(end_point_list)[:, 1]]))
        if "cose_7" in pcd_info_2d.dir_name:
            filter_collected_cross_points = collected_cross_points[(collected_cross_points[:, 1]>=min_y) & (collected_cross_points[:, 1]<=max_y)]
        else:
            filter_collected_cross_points = collected_cross_points[(collected_cross_points[:, 0]>=min_x) & (collected_cross_points[:, 0]<=max_x)]
        ax.scatter(filter_collected_cross_points[:, 0], filter_collected_cross_points[:, 1], s=5, c="blue", label="2d")

        for group_idx in peak_points_3d.keys():
            ax.scatter(peak_points_3d[group_idx][:, 0], peak_points_3d[group_idx][:, 1], s=5, c="r", label="3d")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
        plt.close()

        # 2dの歩幅を取得
        step_length_list_2d = []
        for idx in range(1, len(filter_collected_cross_points)):
            step_length_list_2d.append(ori_method.calc_points_distance(filter_collected_cross_points[idx-1], filter_collected_cross_points[idx]))

        if True:
            # 平均値の比較
            print("2d", np.mean(np.array(step_length_list_2d)[(np.array(step_length_list_2d)>400) & (np.array(step_length_list_2d)<1000)]))
            print("3d", np.mean(step_length_list_3d[0]))

            # 歩数の比較
            print("2d", len(step_length_list_2d))
            print("3d", len(step_length_list_3d[0]))

            column = pcd_info_2d.dir_name.split("_")[0]+"_"+pcd_info_2d.dir_name.split("_")[1]
            column = cose_replace_mapping[column]
            index = pcd_info_2d.dir_name.split("_")[2]+"_"+pcd_info_2d.dir_name.split("_")[3]
            index = name_replace_mapping[index]

            length_2d_df.at[index, column] = np.mean(np.array(step_length_list_2d))
            length_3d_df.at[index, column] = np.mean(step_length_list_3d[0])
            length_diff_df.at[index, column] = np.mean(np.mean(step_length_list_3d[0])-np.mean(np.array(step_length_list_2d)))
            step_num_2d_df.at[index, column] = len(step_length_list_2d)
            step_num_3d_df.at[index, column] = len(step_length_list_3d[0])
            step_num_diff_df.at[index, column] = len(step_length_list_3d[0])-len(step_length_list_2d)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        continue


# 絶対平均誤差を計算
length_diff_df.loc["絶対平均誤差"] = length_diff_df.abs().mean()

# 結果を保存
if True:
    str = "1218"
    os.makedirs(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}", exist_ok=True)
    length_2d_df.to_csv(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}/length_2d.csv")
    length_3d_df.to_csv(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}/length_3d.csv")
    length_diff_df.to_csv(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}/legth_diff.csv")
    step_num_2d_df.to_csv(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}/step_num_2d.csv")
    step_num_3d_df.to_csv(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}/step_num_3d.csv")
    step_num_diff_df.to_csv(f"/Users/kai/大学/小川研/LiDAR_step_length/result/{str}/step_num_diff.csv")

# 結果を読み込み
if True:
    length_2d_df = pd.read_csv("/Users/kai/大学/小川研/LiDAR_step_length/result/1218/length_2d.csv", index_col=0)
    length_3d_df = pd.read_csv("/Users/kai/大学/小川研/LiDAR_step_length/result/1218/length_3d.csv", index_col=0)
    length_diff_df = pd.read_csv("/Users/kai/大学/小川研/LiDAR_step_length/result/1218/legth_diff.csv", index_col=0)
    step_num_2d_df = pd.read_csv("/Users/kai/大学/小川研/LiDAR_step_length/result/1218/step_num_2d.csv", index_col=0)
    step_num_3d_df = pd.read_csv("/Users/kai/大学/小川研/LiDAR_step_length/result/1218/step_num_3d.csv", index_col=0)
    step_num_diff_df = pd.read_csv("/Users/kai/大学/小川研/LiDAR_step_length/result/1218/step_num_diff.csv", index_col=0)

# 結果をグラフで表示
if True:
    color_list = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)
    ax.set_xlim(500, 1000)
    ax.set_xlabel("2d_length (mm)")
    ax.set_ylim(500, 1000)
    ax.set_ylabel("3d_length (mm)")
    ax.plot([500, 1000], [500, 1000], c="black")
    ax.plot([500, 1000], [400, 900], c="black", linestyle="--")
    ax.plot([500, 1000], [600, 1100], c="black", linestyle="--")
    for idx, cose in enumerate(length_diff_df.columns):
        ax.scatter(length_2d_df[cose].values, length_3d_df[cose].values, label=cose, s=10, c=color_list[idx])

    plt.legend()
    plt.suptitle("step_length")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)
    ax.set_xlim(0, 15)
    ax.set_xlabel("2d_length")
    ax.set_ylim(0, 15)
    ax.set_ylabel("3d_length")
    ax.plot([0, 15], [0, 15], c="black")
    ax.plot([0, 15], [1, 16], c="black", linestyle="--")
    ax.plot([0, 15], [-1, 14], c="black", linestyle="--")
    for idx, cose in enumerate(step_num_diff_df.columns):
        ax.scatter(step_num_2d_df[cose].values, step_num_3d_df[cose].values, label=cose, s=10, c=color_list[idx])

    plt.legend()
    plt.suptitle("step_num")
    plt.show()
    plt.close()

# 結果をグラフで表示
if True:
    color_list = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)
    ax.set_xlim(50, 100)
    ax.set_xlabel("2d_length (cm)")
    ax.set_ylim(50, 100)
    ax.set_ylabel("3d_length (cm)")
    ax.plot([50, 100], [50, 100], c="black")
    ax.plot([50, 100], [40, 90], c="black", linestyle="--")
    ax.plot([50, 100], [60, 110], c="black", linestyle="--")
    for idx, cose in enumerate(length_diff_df.columns):
        ax.scatter(length_2d_df[cose].values/10, length_3d_df[cose].values/10, label=cose, s=10, c=color_list[idx])

    plt.legend()
    plt.suptitle("step_length")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_box_aspect(1)
    ax.set_xlim(0, 15)
    ax.set_xlabel("2d_length")
    ax.set_ylim(0, 15)
    ax.set_ylabel("3d_length")
    ax.plot([0, 15], [0, 15], c="black")
    ax.plot([0, 15], [1, 16], c="black", linestyle="--")
    ax.plot([0, 15], [-1, 14], c="black", linestyle="--")
    for idx, cose in enumerate(step_num_diff_df.columns):
        ax.scatter(step_num_2d_df[cose].values, step_num_3d_df[cose].values, label=cose, s=10, c=color_list[idx])

    plt.legend()
    plt.suptitle("step_num")
    plt.show()
    plt.close()
