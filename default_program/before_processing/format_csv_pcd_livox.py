import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import pcl


# 何秒分のデータを1つのPCDファイルにまとめるか
grouping_sec_list = [0.1]

for grouping_sec in grouping_sec_list:
    # 変数の指定
    ##############################
    # 読み込むスキャンデータのパス, フォルダパスでも可
    scan_data_path = "./20241120/source/3d/*.csv"
    # 出力するcsvフォルダのパス
    output_data_path = "./20241120/pcd_"+str(grouping_sec).replace(".", "")+"s/3d/"
    # 出力するcsvフォルダのパスの区切り文字
    split_char = "source/3d/"
    ##############################

    def format_livox_csv(scan_data_file, output_data_path):
        df = pd.read_csv(scan_data_file)

        # ユニークな時間ステップを取得
        time_steps = df['Timestamp'].unique()

        # 0.1s分の点群データをまとめたPCDファイルを作成
        start_time = df.iloc[0]['Timestamp']
        end_time = df.iloc[-1]['Timestamp']


        count = 0
        while (start_time + count*grouping_sec*1e+9 < end_time):
            range_start_time = start_time + count*grouping_sec*1e+9
            range_end_time = start_time + (count+1)*grouping_sec*1e+9

            count += 1

            # 特定の時間ステップのデータを抽出
            df_time = df[(df['Timestamp'] >= range_start_time) & (df['Timestamp'] < range_end_time)]
        
            # 点群データを作成するための座標を抽出
            points = np.array(df_time[['Ori_x', 'Ori_y', 'Ori_z']].values, dtype=np.float32)
            # 点座標が(0,0,0)の場合は除去
            points = points[np.sum(points, axis=1)!=0]
            # PCLの点群オブジェクトに変換
            cloud = pcl.PointCloud()
            cloud.from_array(points.astype(np.float32))

            # 必要に応じて各時間ステップの点群データを保存
            cloud.to_file(f"{output_data_path}/{count}.pcd".encode('utf-8'))

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

            # csvファイルを読み込み，PCDファイルに変換
            format_livox_csv(scan_data_file, tmp_output_data_path)
