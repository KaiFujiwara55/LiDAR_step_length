import os
import numpy as np
import glob
import pcl
import default_method

class original_method:
    """3次元点群処理のオリジナルな処理をまとめたクラス"""
    def __init__(self):
        self.def_method = default_method.cloud_method()

    def save_original_data(self, time_area_points_list, time_area_center_point_list, output_path):
        """
        時系列の点群のグループを保存
        引数:
            time_area_points_list: list
            time_area_center_point_list: list
        返り値:
            None
        """
        os.makedirs(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_center_point_list/{output_path}", exist_ok=True)
        os.makedirs(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_points_list/{output_path}", exist_ok=True)
        for time_idx in range(len(time_area_points_list)):
            center_points = time_area_center_point_list[time_idx]
            with open(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_center_point_list/{output_path}/{time_idx}.txt", "w") as f:
                for center_point in center_points:
                    f.write(f"{center_point[0]} {center_point[1]} {center_point[2]}\n")
            for group_idx in range(len(time_area_points_list[time_idx])):
                area_points = time_area_points_list[time_idx][group_idx]
                with open(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_points_list/{output_path}/{time_idx}_{group_idx}.txt", "w") as f:
                    for area_point in area_points:
                        f.write(f"{area_point[0]} {area_point[1]} {area_point[2]}\n")

    def load_original_data(self, load_path):
        """
        時系列の点群のグループを読み込み
        引数:
            load_path: str
        返り値:
            time_area_points_list: list
            time_area_center_point_list: list
        """
        time_idxs = len(glob.glob(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_center_point_list/{load_path}/*"))

        time_area_points_list = []
        time_area_center_point_list = []
        for time_idx in range(time_idxs):
            center_points = []
            with open(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_center_point_list/{load_path}/{time_idx}.txt", "r") as f:
                for line in f:
                    center_point = [float(x) for x in line.split()]
                    center_points.append(np.array(center_point, dtype=np.float32))
            area_points_list = []
            group_idxs = len(glob.glob(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_points_list/{load_path}/{time_idx}_*"))
            for group_idx in range(group_idxs):
                area_points = []
                with open(f"/Users/kai/大学/小川研/LIDAR/default_program/3dlidar/tmp_folder/time_area_points_list/{load_path}/{time_idx}_{group_idx}.txt", "r") as f:
                    for line in f:
                        area_point = [float(x) for x in line.split()]
                        area_points.append(area_point)
                area_points_list.append(np.array(area_points))
            time_area_points_list.append(area_points_list)
            time_area_center_point_list.append(center_points)
        return time_area_points_list, time_area_center_point_list

    def calc_points_distance(self, points1, points2):
        """
        2つの点群の距離を計算
        引数:
            points1: np.array
            points2: np.array
        返り値:
            distance: float
        """
        distance = np.linalg.norm(points1-points2)
        return distance

    def get_neighborhood_points(self, cloud, radius=250):
        """
        xy平面上で半径radius内の点群のグループを取得
        引数:
            cloud: pcl.PointCloud
            radius: int
        返り値:
            area_points_list: list
            area_center_point_list: list
        
        area_points_list: 点群のグループのリスト
        area_center_point_list: 点群のグループの基準点のリスト
        """
        # 点群のz座標を0に変換
        points_xy = np.array(cloud)
        points_xy[:, 2] = 0
        cloud_xy = pcl.PointCloud()
        cloud_xy.from_array(points_xy)
        
        # x,y平面上で半径radius内の円内の点群数を取得
        indicesm, sqr_distances = self.def_method.kdtree_search_all(cloud_xy)
        count_under_threshold = np.sum(sqr_distances<radius**2, axis=1) # radius**2はkdtreeの返り値に合わせて2乗

        # 各点に対する円内にある点群数が多い順にソート
        idxs = np.argsort(count_under_threshold)[::-1]
        
        # 重複がないように領域内で点群数が100以上の範囲をピックアップ
        area_points_list = []
        area_center_point_list = []
        # ピックアップされていない点群のインデックス
        reserve_points_idx = [i for i in range(len(count_under_threshold))]
        for idx in idxs:
            # 領域内の点群のインデックスを取得
            under_idx = indicesm[idx, np.where(sqr_distances[idx]<radius**2)[0]]

            # 点群数が100未満の場合 or すでにピックアップされた点群を含む場合はスキップ
            if len(under_idx)<100:
                continue
            elif not np.all(np.isin(under_idx, reserve_points_idx)):
                continue
            
            # 領域内の点群をリストに追加
            area_points_list.append(np.array(cloud)[under_idx])
            area_center_point_list.append(np.array(cloud)[idx])
            
            # 要素を引き当てて削除
            reserve_points_idx = np.setdiff1d(reserve_points_idx, under_idx)
    
        return area_points_list, area_center_point_list

    def grouping_points_list(self, time_area_points_list, time_area_center_point_list, integrade_threshold=5):
        """
        時系列の点群のグループを統合
        引数:
            time_area_points_list: list
            time_area_point_list: list
            integrade_threshold: int
        返り値:
            integraded_area_points_list: list
            integraded_area_point_list: list

        integrade_threshold: 孤立点群をノイズとして除去するための閾値
        integraded_area_points_list: 統合された点群のグループのリスト
        integraded_area_point_list: 統合された点群のグループの基準点のリスト
        """

        label_idx_list = []

        time = len(time_area_points_list)
        for time_idx in range(time):
            # 時系列の中心点のlistを取得
            area_points_list = time_area_points_list[time_idx]
            area_center_point_list = time_area_center_point_list[time_idx]
            
            theshold = 250

            # tmp_mis = [[gropu_idx, [group_idx2, diff1]], group_idx, [group_idx2, diff2]...]
            tmp_mins = []
            for group_idx in range(len(area_points_list)):
                # 各時刻，各グループの中心座標に対して処理を行う
                area_center_point_xy = area_center_point_list[group_idx]
                area_center_point_xy[2] = 0
                
                # ラベル付された点群がない場合の処理
                if time_idx==0 or len(label_idx_list)==0:
                    label_idx_list.append([-1 for i in range(time)])
                    label_idx_list[-1][time_idx] = group_idx
                else:
                    tmp_min = None
                    for group_idx2 in range(len(label_idx_list)):
                        before_label_idx1 = label_idx_list[group_idx2][time_idx-1]
                        if not before_label_idx1 == -1:
                            before_label_person_center_point1 = time_area_center_point_list[time_idx-1][before_label_idx1]
                            distance1 = self.calc_points_distance(area_center_point_xy, before_label_person_center_point1)
                            if distance1 < theshold:
                                if tmp_min is None:
                                    tmp_min = [group_idx2, distance1]
                                else:
                                    if distance1 < tmp_min[1]:
                                        tmp_min = [group_idx2, distance1]
                            continue
                        if time_idx>=2:
                            before_label_idx2 = label_idx_list[group_idx2][time_idx-2]
                            if not before_label_idx2 == -1:
                                before_label_person_center_point2 = time_area_center_point_list[time_idx-2][before_label_idx2]
                                distance2 = self.calc_points_distance(area_center_point_xy, before_label_person_center_point2)
                                
                                if distance2 < theshold:
                                    if tmp_min is None:
                                        tmp_min = [group_idx2, distance2]
                                    else:
                                        if distance2 < tmp_min[1]:
                                            tmp_min = [group_idx2, distance2]
                                continue
                    if tmp_min is not None:
                        label_idx_list[tmp_min[0]][time_idx] = group_idx
                    else:
                        label_idx_list.append([-1 for i in range(time)])

                        label_idx_list[-1][time_idx] = group_idx
                    tmp_mins.append(tmp_min)

        # 孤立点群数がintegrade_thresholdよりも小さいグループをノイズとして除去
        count_group = np.sum(np.array(label_idx_list)!=-1, axis=1)
        label_idx_list = np.array(label_idx_list)[count_group>integrade_threshold]

        integraded_area_points_list = []
        integraded_area_point_list = []
        for label_idx in label_idx_list:
            tmp_points = []
            tmp_center_points = []
            for time_idx in range(time):
                group_idx = label_idx[time_idx]
                if group_idx == -1:
                    tmp_points.append([[]])
                    tmp_center_points.append([])
                else:
                    tmp_points.append(time_area_points_list[time_idx][group_idx])
                    tmp_center_points.append(time_area_center_point_list[time_idx][group_idx])
            integraded_area_points_list.append(tmp_points)
            integraded_area_point_list.append(tmp_center_points)

        return integraded_area_points_list, integraded_area_point_list


    def get_vector(self, integraded_area_center_point_list):
        """
        統合された点群のグループの中心点からベクトルを取得
        引数:
            integraded_area_center_point_list: list
        返り値:
            vector_list: list
        """
        vectors_list = []
        for group_idx in range(len(integraded_area_center_point_list)):
            vectors = []
            for time_idx in range(1, len(integraded_area_center_point_list[0])):
                before_center_point = integraded_area_center_point_list[group_idx][time_idx-1]
                after_center_point = integraded_area_center_point_list[group_idx][time_idx]

                if len(before_center_point)==0 and len(after_center_point)==0:
                    vectors.append([])
                elif len(before_center_point)==0 and len(after_center_point)>0:
                    vectors.append([])
                elif len(before_center_point)>0 and len(after_center_point)>0:
                    vectors.append(after_center_point - before_center_point)
                elif len(before_center_point)>0 and len(after_center_point)==0:
                    vectors.append([])
            vectors_list.append(vectors)
        
        return vectors_list

    def judge_move(self, vectors_list, threshold=500):
        """
        ベクトルの合計の大きさがthresholdよりも大きいか判定
        引数:
            vectors_list: list
            threshold: int
        返り値:
            move_flg: bool
        """
        move_flg_list = []
        for vectors in vectors_list:
            all_vector = np.array([0., 0., 0.])
            for vector in vectors:
                if len(vector)>0:
                    all_vector += vector
            if np.linalg.norm(all_vector)>threshold:
                move_flg_list.append(True)
            else:
                move_flg_list.append(False)
        
        return move_flg_list
    
    def get_height(self, points, thread_num=3):
        """
        点群の高さを取得
        引数:
            points: np.array
            thread_num: int
        返り値:
            height: float
        
        points: 点群
        thread_num: 上位何%の点の平均を高さとするか
        """
        
        # 点群のz座標を取得
        z = points[:, 2]
        
        # 上位何%の点の平均を高さとする
        z = np.sort(z)[::-1]
        z = z[:int(len(z)*thread_num/100)]
        height = np.mean(z)
        
        return height

    def get_bentchmark(self, points, min_percent, max_percent):
        """
        点群のある領域の高さの平均値を取得
        引数:
            points: np.array
            min_percent: float
            max_percent: float
        返り値:
            bench_height: float

        points: 点群
        min_percent: 下位何%までの点の平均を高さとするか(0が底辺)
        max_percent: 上位何%までの点の平均を高さとするか(100が最大値)

        """
        # 点群のz座標を取得
        z = points[:, 2]
        
        
        z = np.sort(z)
        bench_height = np.mean(z[int(len(z)*min_percent/100):int(len(z)*max_percent/100)])
        
        return bench_height
