# 移動している人の点群データを人物毎でラベル付する

import os
import imageio
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm



# 障害物点群を抽出
def get_obstacle_points(scan_data):
    # 距離データから各角度の中央値を取得
    distances = scan_data[:, 2::2]
    medians = np.median(distances, axis=0)

    # 各角度の中央値からの距離が40以下の点を障害物点群として抽出
    obstacle_points = np.zeros(distances.shape)
    for idx, median in enumerate(medians):
        if median != 0:
            point = np.where(abs(distances[:, idx]-median)<40, distances[:, idx], 0)
            obstacle_points[:, idx] = point
            # 前後のインデックスの中央値が0の場合は測定誤差で角度のずれが発生している可能性があるため，その場合も障害物点群として抽出
            if idx!=0:
                if medians[idx-1] == 0:
                    point = np.where(abs(distances[:, idx-1]-median)<40, distances[:, idx-1], 0)
                    obstacle_points[:, idx-1] = point
            if idx!=len(medians)-1:
                if medians[idx+1] == 0:
                    point = np.where(abs(distances[:, idx+1]-median)<40, distances[:, idx+1], 0)
                    obstacle_points[:, idx+1] = point

    return obstacle_points

# 外れ値点を抽出
def get_outlier_points(scan_data):
    distances = scan_data[:, 2::2]

    outlier_points = np.zeros(distances.shape)
    # 各時間で連続する5シーケンス間で点群が孤立している場合は外れ値点として抽出
    for angle_idx in range(distances.shape[0]):
        angle_distance = distances[angle_idx]
        for time_idx in range(len(angle_distance)):
            distance = angle_distance[time_idx]
            #  距離が0以外の場合は前2シーケンス, 後2シーケンスの距離が0である時外れ値点として抽出
            if not distance == 0:
                if time_idx>2 and time_idx<len(angle_distance)-2:
                    if angle_distance[time_idx-2] == 0 and angle_distance[time_idx-1] == 0 and angle_distance[time_idx+1] == 0 and angle_distance[time_idx+2] == 0:
                        outlier_points[angle_idx, time_idx] = distance
                elif time_idx<=2:
                    if angle_distance[time_idx+1] == 0 and angle_distance[time_idx+2] == 0:
                        outlier_points[angle_idx, time_idx] = distance
                elif time_idx>=len(angle_distance)-2:
                    if angle_distance[time_idx-2] == 0 and angle_distance[time_idx-1] == 0:
                        outlier_points[angle_idx, time_idx] = distance
    
    return outlier_points

# 外れ値点を抽出2
def get_outlier_points2(scan_data):
    angles = scan_data[:, 1::2]
    distances = scan_data[:, 2::2]

    outlier_points = np.zeros(distances.shape)
    # 各角度で連続する5シーケンス間で点群が孤立している場合は外れ値点として抽出
    for angle_idx in range(distances.shape[1]):
        angle_distance = distances[:, angle_idx]
        for time_idx in range(2, len(angle_distance)-2):
            distance = angle_distance[time_idx]
            #  距離が0以外の場合は時間軸の前後2シーケンスいずれかとの距離差が100mm以上の場合は外れ値点として抽出
            if not distance == 0:
                if not any(np.where(abs(angle_distance[[time_idx-2, time_idx-1, time_idx+1, time_idx+2]]-distance)<100, True, False)):
                    # 外れ値としてピックアップされた点の角度の前後2シーケンス全てとの距離差が500mm以上の場合は0に変換
                    time_distance = distances[time_idx, :]
                    if all(np.where(abs(time_distance[[angle_idx-2, angle_idx-1, angle_idx+1, angle_idx+2]]-distance)>500, True, False)):
                        outlier_points[time_idx, angle_idx] = distance
    return outlier_points

# 障害物点群を除去
def remove_obstacle_points(scan_data, obstacle_points):
    # 障害物点群を0に変換
    scan_data[:, 2::2] = np.where(obstacle_points!=0, 0, scan_data[:, 2::2])
    return scan_data

# 外れ値点を除去
def remove_outlier_points(scan_data, outlier_points):
    # 外れ値点を0に変換
    scan_data[:, 2::2] = np.where(outlier_points!=0, 0, scan_data[:, 2::2])
    return scan_data

# 人物点群を抽出
def get_person_points(scan_data):
    distances = scan_data[:, 2::2]

    person_points = np.zeros(distances.shape)
    # 各シーケンスで連続する5角度間のうち，3角度以上が0でない場合は人物点群として抽出
    for time_idx in range(distances.shape[1]):
        time_distance = distances[:, time_idx]
        for angle_idx in range(len(time_distance)):
            distance = time_distance[angle_idx]
            if not distance == 0:
                if angle_idx>2 and angle_idx<len(time_distance)-2:
                    if time_distance[angle_idx-2] != 0 and time_distance[angle_idx-1] != 0 and time_distance[angle_idx+1] != 0 and time_distance[angle_idx+2] != 0:
                        person_points[angle_idx, time_idx] = distance
                elif angle_idx<=2:
                    if time_distance[angle_idx+1] != 0 and time_distance[angle_idx+2] != 0:
                        person_points[angle_idx, time_idx] = distance
                elif angle_idx>=len(time_distance)-2:
                    if time_distance[angle_idx-2] != 0 and time_distance[angle_idx-1] != 0:
                        person_points[angle_idx, time_idx] = distance

    return person_points

# 人物点群を除去
def remove_person_points(scan_data, person_points):
    # 人物点群を0に変換
    scan_data[:, 2::2] = np.where(person_points!=0, 0, scan_data[:, 2::2])
    return scan_data

# 複数の人物点群を分離して，各人を構成する点群座標を返す
def separate_person_points(scan_data, person_points):
    angles = scan_data[:, 1::2]
    distances = person_points

    theshold = 100
    relation_idxs_list = []
    # 各時間で隣接する角度同士の距離差が50mm以下の場合は同一人物としてラベル付
    for time_idx in range(angles.shape[0]):
        time_angle = angles[time_idx, :]
        time_distance = distances[time_idx, :]
        relation_idxs = []
        for idx in range(2, time_angle.shape[0]-2):
            bench_point = [time_angle[idx], time_distance[idx]]

            point1 = [time_angle[idx-2], time_distance[idx-2]]
            point2 = [time_angle[idx-1], time_distance[idx-1]]
            point3 = [time_angle[idx+1], time_distance[idx+1]]
            point4 = [time_angle[idx+2], time_distance[idx+2]]
            points = [point1, point2, point3, point4]

            if bench_point[1] > 0:
                for i, point in enumerate(points):
                    point_distance = calc_distance(bench_point, point)
                    if point[1] > 0 and point_distance < theshold:
                        if i<2:
                            relation_idxs.append([idx, idx-2+i])
                        else:
                            relation_idxs.append([idx, idx-1+i])

                if merge_groups(relation_idxs)!=[]:
                    points = [distances[time_idx, idx] for idx in merge_groups(relation_idxs)[0]]
        relation_idxs_list.append(merge_groups(relation_idxs))

    return relation_idxs_list

# 人物点群の中心座標を計算
def calculate_person_center_points(scan_data, relation_idxs_list):
    angles = scan_data[:, 1::2]
    distances = person_points

    # 各時間の人物点群の中心を計算
    person_center_points_list = []
    for time_idx, relation_idxs in enumerate(relation_idxs_list):
        person_center_points = []
        for relation_idx in relation_idxs:
            points = [[angles[time_idx, idx], distances[time_idx, idx]] for idx in relation_idx]
            centar_point = calc_polar_centar(points)
            person_center_points.append(centar_point)
        person_center_points_list.append(person_center_points)
    
    return person_center_points_list

# 各時間の人物点群の中心を用いて人物点群をラベル付
def label_person_poinsts(person_points, person_points_list, person_center_points_list):
    new_person_points_list = []
    new_person_center_point_list = []

    person_num = -1
    theshold = 100

    # 時間軸の前2シーケンス内で人物点群の中心座標の距離差が半径50mm以内の場合は同一人物としてラベル付
    for time_idx, person_center_points in enumerate(person_center_points_list):
        for idx, person_center_point in enumerate(person_center_points):
            # 時刻0の場合は孤立点群をそれぞれの人物点群としてリストに追加
            if time_idx==0:
                new_person_points_list.append(np.zeros(person_points.shape))
                new_person_center_point_list.append(np.zeros([person_points.shape[0], 2]))
                person_num += 1
                
                new_person_points_list[person_num][time_idx, person_points_list[time_idx][idx]] = person_points[time_idx, person_points_list[time_idx][idx]]
                new_person_center_point_list[person_num][time_idx] = person_center_point
            # 時刻1の場合は時刻0の人物点群の中心座標との距離が50mm以内の場合は同一人物としてラベル付
            elif time_idx==1:
                distances = [calc_distance(person_center_point, new_person_center_point[time_idx-1]) for new_person_center_point in new_person_center_point_list]
                if min(distances) < theshold:
                    target_idx = distances.index(min(distances))
                    new_person_points_list[target_idx][time_idx] = person_points_list[time_idx][idx]
                    new_person_center_point_list[target_idx][time_idx] = person_center_point
                else:
                    new_person_points_list.append(np.zeros(person_points.shape))
                    new_person_center_point_list.append(np.zeros([person_points.shape[0], 2]))
                    person_num += 1
                    
                    new_person_points_list[person_num][time_idx, person_points_list[time_idx][idx]] = person_points[time_idx, person_points_list[time_idx][idx]]
                    new_person_center_point_list[person_num][time_idx] = person_center_point
            # 時刻tの場合は時刻t-1, t-2の人物点群の中心座標との距離が50mm以内の場合は同一人物としてラベル付
            else:
                distances1 = [calc_distance(person_center_point, person_points) for person_points in person_points_list[time_idx-1]]
                distances2 = [calc_distance(person_center_point, person_points) for person_points in person_points_list[time_idx-2]]
                if distances1!=[]:
                    if min(distances1) < theshold:
                        target_idx = distances1.index(min(distances))
                        new_person_points_list[target_idx][time_idx] = person_points_list[time_idx][idx]
                        new_person_center_point_list[target_idx][time_idx] = person_center_point
                        break
                if distances2!=[]:
                    if min(distances2) < theshold:
                        target_idx = distances2.index(min(distances))
                        new_person_points_list[target_idx][time_idx] = person_points_list[time_idx][idx]
                        new_person_center_point_list[target_idx][time_idx] = person_center_point
                        break
                
                new_person_points_list.append(np.zeros(person_points.shape))
                new_person_center_point_list.append(np.zeros([person_points.shape[0], 2]))
                person_num += 1

                new_person_points_list[person_num][time_idx, person_points_list[time_idx][idx]] = person_points[time_idx, person_points_list[time_idx][idx]]
                new_person_center_point_list[person_num][time_idx] = person_center_point

    return new_person_points_list

# 各時間の人物点群の中心を用いて人物点群をラベル付
def label_person_poinsts2(person_points, person_points_idx_list, person_center_point_list):
    time_idxs = person_points.shape[0]
    angle_idxs = person_points.shape[1]

    person_num = -1
    theshold = 100

    label_person_points_list = []
    label_person_center_points_list = []
    # 時間軸の前2シーケンス内で人物点群の中心座標の距離差が半径50mm以内の場合は同一人物としてラベル付
    for time_idx in range(time_idxs):
        person_points_idx = person_points_idx_list[time_idx]
        person_center_points = person_center_points_list[time_idx]

        # tmp_mis = [[gropu_idx, [group_idx2, diff1]], group_idx, [group_idx2, diff2]...]
        tmp_mins = []
        for group_idx in range(len(person_center_points)):
            # 各時刻，各グループの中心座標に対して処理を行う
            person_point_idx = person_points_idx[group_idx]
            person_center_point = person_center_points[group_idx]
            
            # ラベル付された点群がない場合の処理
            if len(label_person_points_list)==0:
                label_person_points_list.append(np.zeros(person_points.shape))
                label_person_center_points_list.append(np.zeros([time_idxs, 2]))

                label_person_points_list[-1][time_idx, person_point_idx] = person_points[time_idx, person_point_idx]
                label_person_center_points_list[-1][time_idx] = person_center_point
            else:
                tmp_min = None
                for group_idx2 in range(len(label_person_center_points_list)):
                    # print(f"group_idx:{group_idx}, group_idx2:{group_idx2}")
                    before_label_person_center_point1 = label_person_center_points_list[group_idx2][time_idx-1]
                    distance1 = calc_distance(person_center_point, before_label_person_center_point1)
                    if time_idx>=2:
                        before_label_person_center_point2 = label_person_center_points_list[group_idx2][time_idx-2]
                        distance2 = calc_distance(person_center_point, before_label_person_center_point2)
                        # print(f"distance1:{distance1}, distance2:{distance2}")

                    if not (before_label_person_center_point1 == np.zeros(2)).all():
                        if distance1 < theshold:
                            if tmp_min is None:
                                tmp_min = [group_idx2, distance1]
                            else:
                                if distance1 < tmp_min[1]:
                                    tmp_min = [group_idx2, distance1]
                        continue
                    if time_idx >= 2:
                        if not (before_label_person_center_point2 == np.zeros(2)).all():
                            if distance2 < theshold:
                                if tmp_min is None:
                                    tmp_min = [group_idx2, distance2]
                                else:
                                    if distance2 < tmp_min[1]:
                                        tmp_min = [group_idx2, distance2]
                            continue
                if tmp_min is not None:
                    label_person_points_list[tmp_min[0]][time_idx, person_point_idx] = person_points[time_idx, person_point_idx]
                    label_person_center_points_list[tmp_min[0]][time_idx] = person_center_point
                else:
                    label_person_points_list.append(np.zeros(person_points.shape))
                    label_person_center_points_list.append(np.zeros([time_idxs, 2]))

                    label_person_points_list[-1][time_idx, person_point_idx] = person_points[time_idx, person_point_idx]
                    label_person_center_points_list[-1][time_idx] = person_center_point
                tmp_mins.append(tmp_min)

    return label_person_points_list, label_person_center_points_list

# ラベル付された人物点群の時間軸のデータ数が5以下の場合はラベル付を取り消す
def remove_label_person_points(label_person_points_list, label_person_center_points_list):
    time_idxs = len(label_person_points_list[0])
    new_label_person_points_list = []
    new_label_person_center_points_list = []

    for label_person_points, label_person_center_points in zip(label_person_points_list, label_person_center_points_list):
        if len(np.where(label_person_points!=0)[0]) >= 5:
            new_label_person_points_list.append(label_person_points)
            new_label_person_center_points_list.append(label_person_center_points)
    
    return new_label_person_points_list, new_label_person_center_points_list

# 人物点群を囲む長方形を作成
def create_person_rectangle(person_points):
    angles = scan_data[0, 1::2]
    time_idxs = person_points.shape[0]
    
    rectangle_points_list = []
    for time_idx in range(time_idxs):
        person_point = person_points[time_idx]
        target_angle_idxs = np.where(person_point!=0)[0]

        points = []
        for target_angle_idx in target_angle_idxs:
            angle = angles[target_angle_idx]
            distance = person_point[target_angle_idx]
            points.append([angle, distance])

        if points == []:
            rectangle_points_list.append([])
        else:
            rectangle_points_list.append(calc_polar_rectangle(points))
    
    return rectangle_points_list


# 極座標の2点間の距離を計算
def calc_distance(point1 ,point2):
    angle1 = point1[0]
    angle2 = point2[0]
    distance1 = point1[1]
    distance2 = point2[1]

    return np.sqrt(distance1**2 + distance2**2 - 2*distance1*distance2*np.cos(angle1-angle2))

# インデックスのグループ化を行う
def merge_groups(pairs):
    # 各要素が属するグループを管理する辞書
    groups = {}

    # 各要素に対してその要素を含むグループのリストを初期化
    for pair in pairs:
        for item in pair:
            if item not in groups:
                groups[item] = set()
    
    # ペアを走査し、各ペアの要素を含むグループを結合
    for pair in pairs:
        set1 = groups[pair[0]]
        set2 = groups[pair[1]]
        
        # 既存のグループを結合
        unified_set = set1.union(set2)
        unified_set.update(pair)
        
        for item in unified_set:
            groups[item] = unified_set
    
    # 重複を排除して最終的なグループを作成
    unique_groups = []
    seen = set()
    for group in groups.values():
        frozen_group = frozenset(group)
        if frozen_group not in seen:
            seen.add(frozen_group)
            unique_groups.append(list(group))
    
    # グループ内の要素をソート
    for group in unique_groups:
        group.sort()
    
    return unique_groups

# 極座標の中心を計算
def calc_polar_centar(points):
    angles = [point[0] for point in points]
    distances = [point[1] for point in points]

    x = np.mean([distance*np.cos(angle) for angle, distance in zip(angles, distances)])
    y = np.mean([distance*np.sin(angle) for angle, distance in zip(angles, distances)])

    centar_point = [np.arctan2(y, x), np.sqrt(x**2 + y**2)]

    return centar_point

# 極座標の点群を囲む長方形の4点を計算
def calc_polar_rectangle(points):
    angles = [point[0] for point in points]
    distances = [point[1] for point in points]
    if np.max(angles)<0:
        angle_max = np.max(angles)*0.9
    else:
        angle_max = np.max(angles)*1.1
    
    if np.min(angles)<0:
        angle_min = np.min(angles)*1.1
    else:
        angle_min = np.min(angles)*0.9

    distance_max = np.max(distances)*1.1
    distance_min = np.min(distances)*0.9

    rectangle_points = [[angle_max, distance_max], [angle_max, distance_min], [angle_min, distance_min], [angle_min, distance_max]]

    return rectangle_points

if __name__ == "__main__":
    scan_data_path = "./20240614/scan_data_0614/filtered/2dlidar/high_position/1person_06.csv"
    # scan_data_path_list = glob.glob("./20240614/scan_data_0614/filtered/2dlidar/high_position/*")
    scan_data_path_list = [scan_data_path]
    for scan_data_path in tqdm(scan_data_path_list):
        # gifフォルダに保存されたファイルはスキップ
        # if os.path.isfile('./default_program/2dlidar/gif/'+scan_data_path.split("/")[-1].split(".")[0]+'.gif'):
        #     continue
        # if "random" in scan_data_path:
        #     continue
        try:
            with open(scan_data_path, mode="r") as f:
                scan_data = np.loadtxt(f, delimiter=",")
        except:
            continue

        obstacle_points = get_obstacle_points(scan_data)
        scan_data = remove_obstacle_points(scan_data, obstacle_points)
        outlier_points = get_outlier_points2(scan_data)
        scan_data = remove_outlier_points(scan_data, outlier_points)


        person_points = get_person_points(scan_data)
        person_point_list = separate_person_points(scan_data, person_points)
        person_center_points_list = calculate_person_center_points(scan_data, person_point_list)
        new_person_points_list, new_person_center_points_list = label_person_poinsts2(person_points, person_point_list, person_center_points_list)
        new_person_points_list, new_person_center_points_list = remove_label_person_points(new_person_points_list, new_person_center_points_list)
        
        angles = scan_data[0, 1::2]
        images = []
        for idx in tqdm(range(len(obstacle_points))):
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
            ax.set_ylim(0, 7000)
            scan_array = scan_data[idx, 1::2]
            scan_distance = scan_data[idx, 2::2]
            
            colors = ["red", "blue", "green", "orange", "purple", "pink", "black", "gray", "brown"]*100

            # for person_point in person_point_list[idx]:
            #     ax.scatter(scan_array[person_point], person_points[idx, person_point], color="blue")
            
            # for i in range(len(person_center_points_list[idx])):
            #     x = person_center_points_list[idx][i][0]
            #     y = person_center_points_list[idx][i][1]
            #     ax.scatter(x, y, color="orange")

            colors = ["red", "blue", "green", "orange", "purple", "pink", "black", "gray", "brown"]*100
            flg = False
            for person_num in range(len(new_person_center_points_list)):
                zero_flg = np.where(new_person_points_list[person_num][idx]!=0)[0]
                if len(zero_flg)>0:
                    plot_angle = angles[zero_flg]
                    plot_distance = new_person_points_list[person_num][idx, zero_flg]
                    if len(plot_angle)>0:
                        ax.scatter(plot_angle, plot_distance, color=colors[person_num])
                        
                        rectangle_points_list = create_person_rectangle(new_person_points_list[person_num])
                        points = [(x[0], x[1]) for x in rectangle_points_list[idx]]
                        if points != []:
                            flg = True
                            a = patches.Polygon(points, closed=True, fill=False, edgecolor=colors[person_num], label=f"{person_num+1} person")
                            ax.add_patch(a)
                if flg:
                    plt.legend()
            # for person_center_point in person_center_points_list[idx]:
            #     ax.scatter(person_center_point[0], person_center_point[1], color="orange")
            # ax.scatter(scan_array, obstacle_points[idx], color="red")
            # ax.scatter(scan_array, outlier_points[idx], color="green")
            # x = scan_data_path.split("/")[-1].split(".")[0]
            ax.set_title(f"{idx} scan")
            
            plt.pause(0.025)
            # plt.close()

            plt.close()

