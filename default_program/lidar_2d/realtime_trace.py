import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from hokuyolx import HokuyoLX

# スキャン範囲をget_dist関数用に変換
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

# 測定範囲に合わせてscan_dataを整形
def formatScanData(scan_data, scan_array):
    # スキャンデータの極座標を直交座標に変換
    scan_x_list = scan_data[:, 1] * np.cos(scan_data[:, 0])
    scan_y_list = scan_data[:, 1] * np.sin(scan_data[:, 0])

    # 測定範囲の極座標を直交座標に変換
    set_x_list = scan_array[:, 1] * np.cos(scan_array[:, 0])
    set_ylist = scan_array[:, 1] * np.sin(scan_array[:, 0])

    # 測定範囲の多角形を作成
    set_polygon = Polygon(set_x_list, set_ylist)

    # 測定範囲内かどうかを確認
    for idx in range(len(scan_x_list)):
        # lidarのスキャンデータが範囲内かどうか確認，範囲外の場合は0にする
        point = Point(scan_x_list[idx], scan_y_list[idx])
        if not set_polygon.contains(point):
            scan_data[idx, 1] = 0
        
    return scan_data

# plotする
def plotData(scan_data, interval, dmax):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    if dmax is not  None:
        ax.set_ylim(0, dmax+1000)
    
    scanArray = scan_data[:, 0]
    scanDistance = scan_data[:, 1]
    
    ax.plot(scanArray, scanDistance)
    
    # pauseを使用してplot
    plt.pause(interval)

# スキャンデータをcsvに変換する
def scan2csv(timestamp, scan_data, csvPath="./scan_data.csv"):
    # scan_dataは二次元配列[[測定角度(rad), 距離(mm)]]
    angle_list = np.rad2deg(scan_data[:, 0])
    dist_list = list(np.ravel(scan_data))
    
    with open(csvPath, encoding="utf-8", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp]+dist_list)

if __name__=="__main__":
    # 設定箇所
    ###############################################################################################################
    # scan_dataを出力するcsvPath
    csvPath = "./scan_data/setting.csv"
    
    # 45度~315度の範囲で0.25度刻みで測定可能(単位：度), Noneで全範囲
    start_angle = 90
    end_angle = 270
    # 測定距離のフィルターをかけることが可能(単位：mm), Noneで全範囲
    dmin = None
    dmax = None
    # 測定範囲を設定する, 角度(rad)と距離(mm)の組み合わせで設定, Noneで全範囲
    if True:
        scan_array = np.array[[90, 10000], [135, 10000*np.sqrt(5)], [225, 10000*np.sqrt(5)], [270, 10000]]
    else:
        scan_array = None

    # 測定の時間間隔を決定する(plotさせない場合用, plotさせると1秒くらいかかる)
    interval = 0.0001
    # データを表示するかのflg
    plot_flg = True
    ###############################################################################################################


    # start_angle, end_angleを関数用に変換
    scan_start_angle, scan_end_angle = angle2scanAngle(start_angle, end_angle)

    # hokuyolxオブジェクトを取得
    laser = HokuyoLX()
    
    # 測定を行う
    for i in range(50):
        print(i)
        try:
            # lidarでスキャン
            timestamp, scan_data = laser.get_filtered_dist(scan_start_angle, scan_end_angle, dmin=dmin, dmax=dmax)
            # scan_arrayの範囲に合わせる
            if scan_array is not None:
                scan_data = formatScanData(scan_data, scan_array)
            scan2csv(timestamp, scan_data, csvPath)
            
            # データを表示
            if plot_flg:
                plotData(scan_data, start_angle, end_angle, interval, dmax)
            else:
                time.sleep(interval)
        except Exception as e:
            print(e)
            break
    laser.close()
