# scanデータを時間グルーブに分割
import os
import csv
import numpy as np
import glob

# 変数の指定
###############################
# 読み込むスキャンデータのパス, フォルダパスでも可
scan_data_path = "./20240614/scan_data_0614/*"
# 出力するcsvフォルダのパス
output_data_path = "./20240614/scan_data_0614/splited_data/"
# 出力するcsvフォルダのパスの区切り文字
split_char = "scan_data_0614/"

# 分割する時間間隔の閾値(単位:ms)
split_time = 26
###############################

# スキャンデータを時間グルーブに分割
def split_scan_data(scan_data, split_time):
    # 時間を取得
    time_list = [float(row[0]) for row in scan_data]
    # 時間の差分を取得
    time_diff = [time_list[i+1] - time_list[i] for i in range(len(time_list)-1)]
    # 時間差分が閾値を超えるインデックスを取得
    split_idx = np.where(np.array(time_diff) > split_time)[0]
    print(split_idx)
    # スキャンデータを分割
    start = 0
    for i in range(len(split_idx)+1):
        print(i)
        split_scan_data = scan_data[split_idx[i]:split_idx[i+1]]
        print(split_scan_data)
        

    return split_scan_data

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

    for scan_data_file in scan_data_files:
        print(scan_data_file)
        with open(scan_data_file, encoding="utf-8", mode="r") as f:
            reader = csv.reader(f)
            scan_data = [row for row in reader]

        # フォルダを作成
        tmp_output_data_path = get_output_data_path(scan_data_file, output_data_path, split_char)
        if not os.path.exists("/".join(tmp_output_data_path.split("/")[:-1])):
            os.makedirs("/".join(tmp_output_data_path.split("/")[:-1]), exist_ok=True)

        # スキャンデータを時間グルーブに分割
        scan_data_list = split_scan_data(scan_data, split_time)
        
        # スキャンデータが分割されていない場合はスキップ
        if len(scan_data_list) == 1:
            continue
        continue
        for i, data in enumerate(scan_data_list):
            print(i, data)
            if i < 10:
                file_path = tmp_output_data_path + "_0" + str(i+1) + ".csv"
            else:
                file_path = tmp_output_data_path + "_" + str(i+1) + ".csv"
            print(file_path)
            continue
            with open(file_path, encoding="utf-8", mode="w", newline="") as f:
                writer = csv.writer(f)
                for row in data:
                    writer.writerow(row)
