import os
import csv
import numpy as np
import glob
from tqdm import tqdm
import pcl


# 何秒分のデータを1つのPCDファイルにまとめるか
grouping_sec_list = [0.025]

for grouping_sec in grouping_sec_list:
    # 変数の指定
    ##############################
    # 読み込むスキャンデータのパス, フォルダパスでも可
    scan_data_path = "./20241204/2d/*.csv"
    # 読み込むタイムデータのパス, フォルダパスでも可
    time_data_path = "./20241204/time_list"
    # 出力するcsvフォルダのパス
    output_data_path = "./20241204/pcd_"+str(grouping_sec).replace(".", "")+"s/2d/"
    # 出力するcsvフォルダのパスの区切り文字
    split_char = "20241204/2d/"
    ##############################

    def format_livox_csv(scan_data_file, output_data_path, time_data_file):
        time_stamps = []
        angle_distances = []
        with open(scan_data_file, "r") as f:
            data = csv.reader(f)
            for row in data:
                time_stamps.append(float(row[0])/1000)
                angle_distances.append(row[1:])
        time_stamps = np.array(time_stamps)

        start_time = float(open(time_data_file, "r").read().split("\n")[0])
        end_time = float(open(time_data_file, "r").read().split("\n")[1])

        count = 0
        output_count = 0
        while (start_time + count*grouping_sec < end_time):
            range_start_time = start_time + count*grouping_sec
            range_end_time = start_time + (count+1)*grouping_sec

            count += 1

            # 特定の時間ステップのデータを抽出
            target_idx_list = np.where((time_stamps >= range_start_time) & (time_stamps < range_end_time))[0]
            
            if len(target_idx_list) == 0:
                continue

            points = None
            for target_idx in target_idx_list:
                target_data = angle_distances[target_idx]
                for row in target_data:
                    row = row.replace("(", "").replace(")", "").split(", ")
                    x, y, z = pol2cart(float(row[1]), float(row[0]))

                    if points is None:
                        points = np.array([x, y, z])
                    else:
                        points = np.vstack([points, [x, y, z]])
        
            # 点座標が(0,0,0)の場合は除去
            points = points[np.sum(points, axis=1)!=0]
            # PCLの点群オブジェクトに変換
            cloud = pcl.PointCloud()
            cloud.from_array(points.astype(np.float32))

            # 必要に応じて各時間ステップの点群データを保存
            cloud.to_file(f"{output_data_path}/{output_count}.pcd".encode('utf-8'))
            output_count += 1

    # 極座標を直交座標に変換
    def pol2cart(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0
        return x, y, z

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

    if __name__=="__main__":
        scan_data_files = get_files(scan_data_path)

        for scan_data_file in tqdm(scan_data_files):
            # 出力先のパスを作成
            tmp_output_data_path = get_output_data_path(scan_data_file, output_data_path, split_char)
            # フォルダを作成
            if not os.path.exists("/".join(tmp_output_data_path.split("/"))):
                os.makedirs("/".join(tmp_output_data_path.split("/")), exist_ok=True)
            
            time_data_file = time_data_path + "/" + scan_data_file.split("/")[-1].split(".")[0] + ".txt"
            # csvファイルを読み込み，PCDファイルに変換
            format_livox_csv(scan_data_file, tmp_output_data_path, time_data_file)
