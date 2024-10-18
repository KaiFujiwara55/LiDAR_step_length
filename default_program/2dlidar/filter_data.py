# 取得したスキャンデータから特定範囲内のデータのみに整形
import os
import csv
import numpy as np
from shapely.geometry import Point, Polygon
import glob
from tqdm import tqdm

# 変数の指定
##############################
# スキャンデータの範囲を指定　[[角度(rad), 距離(mm)], [角度, 距離], ...]
scan_array = np.array([[0.598665387, 4961], [0, 6400], [-1.01005694, 4006], [3.14159, 10]])

# 読み込むスキャンデータのパス, フォルダパスでも可
scan_data_path = "./20240614/scan_data_0614/source/*"
# 出力するcsvフォルダのパス
output_data_path = "./20240614/scan_data_0614/filtered/"
# 出力するcsvフォルダのパスの区切り文字
split_char = "source/"
##############################

# 測定範囲に合わせてscan_dataを整形
def filter_scan_data(scan_data, set_polygon, set_xy_list):
    new_scan_data = []
    for row in scan_data:
        # スキャンデータの極座標を直交座標に変換
        angles = np.array(row[1::2]).astype(np.float64)
        distances = np.array(row[2::2]).astype(np.float64)
        scan_x_list = distances * np.cos(angles)
        scan_y_list = distances * np.sin(angles)

        # 測定範囲内かどうかを確認
        new_row = [row[0]]
        for idx in range(len(scan_x_list)):
            new_row.append(angles[idx])
            # lidarのスキャンデータが範囲内かどうか確認，範囲外の場合は0にする
            point = Point(scan_x_list[idx], scan_y_list[idx])
            if not set_polygon.contains(point):
                new_row.append(0)
            else:
                new_row.append(distances[idx])
        new_scan_data.append(new_row)

    return new_scan_data

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

# スキャンデータを特定範囲内のデータのみに整形
if __name__=="__main__":
    scan_data_files = get_files(scan_data_path)

    for scan_data_file in tqdm(scan_data_files):
        with open(scan_data_file, encoding="utf-8", mode="r") as f:
            reader = csv.reader(f)
            scan_data = [row for row in reader]

        # 測定範囲の極座標を直交座標に変換
        set_x_list = scan_array[:, 1] * np.cos(scan_array[:, 0])
        set_y_list = scan_array[:, 1] * np.sin(scan_array[:, 0])
        set_xy_list = list(zip(set_x_list, set_y_list))
        # 測定範囲の多角形を作成
        set_polygon = Polygon(set_xy_list)

        # 測定範囲内のみのデータに整形　
        filtered_scan_data = filter_scan_data(scan_data, set_polygon, set_xy_list)

        # 出力先のパスを作成
        tmp_output_data_path = get_output_data_path(scan_data_file, output_data_path, split_char)
        # フォルダを作成
        if not os.path.exists("/".join(tmp_output_data_path.split("/")[:-1])):
            os.makedirs("/".join(tmp_output_data_path.split("/")[:-1]), exist_ok=True)

        # csvファイルを作成
        with open(tmp_output_data_path+".csv", encoding="utf-8", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(filtered_scan_data)
