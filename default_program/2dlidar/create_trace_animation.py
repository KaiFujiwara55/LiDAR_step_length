# スキャンデータから軌跡のトレースを行うGIFファイルを作成
import os
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import csv
from tqdm import tqdm

# 変数の指定
##############################
# 読み込むスキャンデータのパス, フォルダパスでも可
scan_data_path = "./default_program/test.csv"
# 出力するcsvフォルダのパス
output_data_path = "./default_program/"
# 出力するcsvフォルダのパスの区切り文字
split_char = "default_program/"

# スキャン範囲の指定がある場合は指定, Noneで全範囲
scan_array = np.array([[0.598665387, 4961], [0, 6400], [-1.01005694, 4006], [3.14159, 10]])

# 軌跡gifを作成するかのflg
rect_trace_flg = True
polar_trace_flg = True
##############################

# 直交座標での軌跡gifを作成
def create_rect_trace_gif(scan_data, scan_array, scan_data_path, output_data_path):
    # スキャンデータの極座標を直交座標に変換
    angles = [np.array(row[1::2]).astype(np.float64) for row in scan_data]
    distances = [np.array(row[2::2]).astype(np.float64) for row in scan_data]
    scan_x_list = [distances * np.cos(angle) for angle, distances in zip(angles, distances)]
    scan_y_list = [distances * np.sin(angle) for angle, distances in zip(angles, distances)]

    # 軌跡の描画
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-100, 7000)
    ax.set_ylim(-4000, 3000)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    if scan_array is not None:
        # 測定範囲の極座標を直交座標に変換
        set_x_list = scan_array[:, 1] * np.cos(scan_array[:, 0])
        set_ylist = scan_array[:, 1] * np.sin(scan_array[:, 0])
        set_xy_list = list(zip(set_x_list, set_ylist))
        # 測定範囲の描画
        mark = patches.Polygon(set_xy_list, edgecolor="black", fill=False)
        ax.add_patch(mark)

    # lidarマークの描画
    lidar_mark = patches.Circle((0, 0), 10, color="b", fill=True)
    ax.add_patch(lidar_mark)

    scatter = ax.scatter([], [], c='blue', marker='.')
    def update(frame):
        scan_x = scan_x_list[frame]
        scan_y = scan_y_list[frame]
        scatter.set_offsets(np.c_[scan_x, scan_y])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(scan_x_list), interval=50, blit=True)
    ani.save(output_data_path+"_rect.gif", writer="pillow")

# 極座標での軌跡gifを作成
def create_polar_trace_gif(scan_data, scan_data_path, output_data_path):
    # 軌跡の描画
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_ylim(0, 7000)

    # lidarマークの描画
    lidar_mark = patches.Circle((0, 0), 10, color="b", fill=True)
    ax.add_patch(lidar_mark)

    im_list = []
    # 軌跡の描画
    for row in scan_data:
        angle = np.array(row[1::2]).astype(np.float64)
        dist = np.array(row[2::2]).astype(np.float64)
        scatter = ax.scatter(angle, dist, c='blue', marker='.')
        im_list.append([scatter])

    # アニメーションの作成
    trace_animation = animation.ArtistAnimation(fig, im_list, interval=50, blit=True)
    # アニメーションを保存
    trace_animation.save(output_data_path+"_polar.gif", writer="pillow")

# 指定したパスは以下のcsvファイルを全取得
def get_files(path):
    all_files = []

    tmp_files = glob.glob(path)
    for tmp_file in tmp_files:
        if ".csv" in tmp_file:
            all_files.append(tmp_file)
        else:
            all_files += get_files(tmp_file+"/*")
    all_files.sort()
    return all_files

# output_data_pathを作成
def get_output_data_path(scan_data_file, output_data_path, split_char):
    if split_char is None:
        tmp_output_data_path = output_data_path + scan_data_file.replace(".csv", "")
    else:
        tmp_output_data_path = output_data_path + scan_data_file.split(split_char)[-1].replace(".csv", "")
    return tmp_output_data_path

def main():
    # スキャンデータのファイルを取得
    scan_data_files = get_files(scan_data_path)
    for scan_data_file in tqdm(scan_data_files):
        with open(scan_data_file, encoding="utf-8", mode="r") as f:
            reader = csv.reader(f)
            scan_data = [row for row in reader]

        tmp_output_data_path = get_output_data_path(scan_data_file, output_data_path, split_char)
        # フォルダを作成
        if not os.path.exists("/".join(tmp_output_data_path.split("/")[:-1])):
            os.makedirs("/".join(tmp_output_data_path.split("/")[:-1]), exist_ok=True)

        if rect_trace_flg:
            create_rect_trace_gif(scan_data, scan_array, scan_data_file, tmp_output_data_path)
        if polar_trace_flg:
            create_polar_trace_gif(scan_data, scan_data_file, tmp_output_data_path)

main()
