# livoxのLiDARのcsvファイルを整形するプログラム

import os
import pandas as pd
import glob
from tqdm import tqdm

# 変数の指定
##############################
# 読み込むスキャンデータのパス, フォルダパスでも可
scan_data_path = "./20241011/source/*.csv"
# 出力するcsvフォルダのパス
output_data_path = "./20241011/filtered/"
# 出力するcsvフォルダのパスの区切り文字
split_char = "source/"
##############################

def format_livox_csv(scan_data_file, tmp_output_data_path):
    df = pd.read_csv(scan_data_file)

    max_num = df["Timestamp"].value_counts().max()
    xyz = [f"(X,Y,Z)_{i}" for i in range(1, max_num+1)]
    reflectivity = [f"reflectivity_{i}" for i in range(1, max_num+1)]
    header = ["timestamp"] + xyz + reflectivity
    
    new_df_dic = {key:[] for key in header}
    
    for timestamp in df["Timestamp"].unique():
        time_df = df[df["Timestamp"]==timestamp]
        new_df_dic["timestamp"].append(timestamp)
        i = 0
        for idx, row in time_df.iterrows():
            new_df_dic[f"(X,Y,Z)_{i+1}"].append("["+str(row["Ori_x"])+", "+str(row["Ori_y"])+", "+str(row["Ori_z"])+"]")
            new_df_dic[f"reflectivity_{i+1}"].append(row["Reflectivity"])
            i += 1

    new_df = pd.DataFrame.from_dict(new_df_dic)
    new_df.to_csv(tmp_output_data_path+".csv", index=False)

def format_livox_csv2(scan_data_file, tmp_output_data_path):
    df = pd.read_csv(scan_data_file)
    # value_countsを使って最大数を取得
    max_num = df["Timestamp"].value_counts().max()

    # 新しいcsvのheaderを作成
    header  = ["timestamp"]
    for i in range(1, max_num+1):
        header += [f"x_{i}", f"y_{i}", f"z_{i}", f"reflectivity_{i}"]
    
    # 新しいcsvのデータを作成
    new_df_dic = {key:[] for key in header}
    unique_timestamps = df["Timestamp"].unique()
    for timestamp in unique_timestamps:
        time_df = df[df["Timestamp"]==timestamp]
        new_df_dic["timestamp"].append(timestamp)
        i = 0
        for idx, row in time_df.iterrows():
            new_df_dic[f"x_{i+1}"].append(row["Ori_x"])
            new_df_dic[f"y_{i+1}"].append(row["Ori_y"])
            new_df_dic[f"z_{i+1}"].append(row["Ori_z"])
            new_df_dic[f"reflectivity_{i+1}"].append(row["Reflectivity"])

            i += 1
    
    new_df = pd.DataFrame.from_dict(new_df_dic)
    new_df.to_csv(tmp_output_data_path + ".csv", index=False)


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
        print(scan_data_file)
        tmp_output_data_path = get_output_data_path(scan_data_file, output_data_path, split_char)
        # フォルダを作成
        if not os.path.exists("/".join(tmp_output_data_path.split("/")[:-1])):
            os.makedirs("/".join(tmp_output_data_path.split("/")[:-1]), exist_ok=True)

        format_livox_csv2(scan_data_file, tmp_output_data_path)

main()
