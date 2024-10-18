# 2次元lidarを用いて，スキャンデータを取得，csvファイルに出力

import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from hokuyolx import HokuyoLX

# 変数の指定
##############################
# scan_dataを出力するcsvPath
csv_path = "./scan_data/setting.csv"

# 45度~315度の範囲で0.25度刻みで測定可能(単位：度), Noneで全範囲
start_angle = 90
end_angle = 270
# 測定距離のフィルターをかけることが可能(単位：mm), Noneで全範囲
dmin = None
dmax = None

# 測定の時間間隔(25ms以上)を決定する(plotさせない場合用, plotさせると1秒くらいかかる)
interval = 0.0025   #単位:s
# 測定回数
scan_num = 50
# データを表示するかのflg
plot_flg = True

##############################

# スキャンする角度をlidar用にstep数に変換
def angle2scanAngle(start_angle, end_angle):
    if start_angle is None or end_angle is None:
        return 0, 1081
    elif start_angle<45:
        raise ValueError("start_angleは45度以上で指定してください")
    elif end_angle>315:
        raise ValueError("end_angleは315度以下で指定してください")
    else:
        start_angle = int((start_angle-45)*4)
        end_angle = int((end_angle-45)*4)
        return start_angle, end_angle

# スキャンデータをcsvに変換する
def scan2csv(timestamp, scan_data, csv_path):
    # scan_dataは二次元配列[[測定角度(rad), 距離(mm)]]
    angle_list = np.rad2deg(scan_data[:, 0])
    dist_list = list(np.ravel(scan_data))
    
    with open(csv_path, encoding="utf-8", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp]+dist_list)

# リアルタイムでスキャンデータをplot
def plotData(scan_data, interval, dmax):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    if dmax is not  None:
        ax.set_ylim(0, dmax+1000)
    
    scanArray = scan_data[:, 0]
    scanDistance = scan_data[:, 1]
    
    ax.scatter(scanArray, scanDistance)
    
    # pauseを使用してplot
    plt.pause(interval)


# 測定を行う
if __name__ == "__main__":
    scan_start_angle, scan_end_angle = angle2scanAngle(start_angle, end_angle)

    laser = HokuyoLX()
    for i in range(scan_num):
        try:
            # lidarでスキャン
            timestamp, scan_data = laser.get_filtered_dist(scan_start_angle, scan_end_angle, dmin=dmin, dmax=dmax)
            # スキャンデータをcsvに変換
            scan2csv(timestamp, scan_data, csv_path)
            
            # データを表示
            if plot_flg:
                plotData(scan_data, start_angle, end_angle, interval, dmax)
            else:
                time.sleep(interval)
        except Exception as e:
            break
    laser.close()
