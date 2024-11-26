# スキャンデータは角度毎でのデータの欠損があるため，欠損データを0で埋める

import glob
import csv
import numpy as np
import pandas as pd

# 変数の指定
##############################
sorce_folderPath = "./LIDAR/20240604/scan_data_0604/filtered/*"
formatted_folderPath = "./LIDAR/20240604/scan_data_0604/formatted/filtered/"
##############################

# スキャンデータを角度ごとのデータに整形, headerは[timestamp, angle1, angle2, ...]
def format_scan_data(source_folderPath, formatted_folderPath):
    # フォルダ内にあるファイルを取得
    scan_data_files = glob.glob(source_folderPath)

    # ファイルごとに処理
    first = True
    for scan_data_file in scan_data_files:
        print(scan_data_file)
        with open(scan_data_file, encoding="utf-8", mode="r") as f:
            reader = csv.reader(f)
            scan_data = [row for row in reader]

        if first:
            # 重複なしangle_listを取得
            angle_list = []
            for row in scan_data:
                angle = row[1::2]
                angle_list+=angle
            angles = list(set(angle_list))

            # 配列内データを数値としてソート
            angles = [float(angle) for angle in angles]
            angles.sort()
            header = [str(x) for x in angles]
            header = ["timestamp"] + header
            
            first = False
        
        # csvファイルを作成
        data = []
        for row in scan_data:
            row_data = []
            for head in header:
                if head == "timestamp":
                    row_data.append(row[0])
                elif head in row:
                    row_data.append(row[row.index(head)+1])
                else:
                    row_data.append(0)
            data.append(row_data)
        with open(formatted_folderPath+scan_data_file.split("/")[-1], encoding="utf-8", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

if __name__ == "__main__":
    format_scan_data(sorce_folderPath, formatted_folderPath)
