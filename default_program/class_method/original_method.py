import os
import numpy as np
import glob
import pcl

from default_program.class_method import default_method
from default_program.class_method import get_pcd_information

class cloud_method:
    """3次元点群処理のオリジナルな処理をまとめたクラス"""
    def __init__(self):
        self.def_method = default_method.cloud_method()

    def save_original_data(self, time_area_points_list, time_area_center_point_list, area_path, center_path):
        """
        時系列の点群のグループを保存
        引数:
            time_area_points_list: list
            time_area_center_point_list: list
        返り値:
            None
        """
        os.makedirs(area_path, exist_ok=True)
        os.makedirs(center_path, exist_ok=True)
        for time_idx in range(len(time_area_points_list)):
            center_points = time_area_center_point_list[time_idx]
            with open(f"{center_path}/{time_idx}.txt", "w") as f:
                for center_point in center_points:
                    f.write(f"{center_point[0]} {center_point[1]} {center_point[2]}\n")
            for group_idx in range(len(time_area_points_list[time_idx])):
                area_points = time_area_points_list[time_idx][group_idx]
                with open(f"{area_path}/{time_idx}_{group_idx}.txt", "w") as f:
                    for area_point in area_points:
                        f.write(f"{area_point[0]} {area_point[1]} {area_point[2]}\n")

    def load_original_data(self, area_path, center_path):
        """
        時系列の点群のグループを読み込み
        引数:
            load_path: str
        返り値:
            time_area_points_list: list
            time_area_center_point_list: list
        """
        time_idxs = len(glob.glob(f"{center_path}/*"))

        time_area_points_list = []
        time_area_center_point_list = []
        for time_idx in range(time_idxs):
            center_points = []
            with open(f"{center_path}/{time_idx}.txt", "r") as f:
                for line in f:
                    center_point = [float(x) for x in line.split()]
                    center_points.append(np.array(center_point, dtype=np.float32))
            area_points_list = []
            group_idxs = len(glob.glob(f"{area_path}/{time_idx}_*"))
            for group_idx in range(group_idxs):
                area_points = []
                with open(f"{area_path}/{time_idx}_{group_idx}.txt", "r") as f:
                    for line in f:
                        area_point = [float(x) for x in line.split()]
                        area_points.append(area_point)
                area_points_list.append(np.array(area_points))
            time_area_points_list.append(area_points_list)
            time_area_center_point_list.append(center_points)
        return time_area_points_list, time_area_center_point_list

    def cloud_get_tilt(self, pcd_info_list, upper_threshold=2000):
        """
        天井を抽出し、LiDAR自体の傾きを取得
        引数:
            pcd_info_list: get_pcd_information.get_pcd_information
            upper_threshold: int
        返り値:
            theta_x: float
            theta_y: float
            theta_z: float
        """
        all_points = None
        for points in pcd_info_list.points_list:
            if all_points is None:
                all_points = points
            else:
                all_points = np.concatenate([all_points, points], axis=0)

        # ダウンサンプリング
        cloud = pcl.PointCloud()
        cloud.from_array(all_points)
        grid_filtered_cloud = self.def_method.voxel_grid_filter(cloud)

        # upper_threshold以下の点群を除去
        upper_cloud = self.def_method.filter_area(grid_filtered_cloud, z_min=upper_threshold)

        # 天井を抽出
        ksearch = 50
        distance_threshold = 50
        cloud_plane, cloud_non_plane, coefficients, indices = self.def_method.segment_plane(upper_cloud, ksearch, distance_threshold)

        theta_x = np.arctan(-1*(coefficients[1]/coefficients[2]))
        theta_y = np.arctan(-1*(coefficients[0]/coefficients[2]))
        theta_z = np.arctan(-1*(coefficients[0]/coefficients[1]))

        return theta_x, theta_y, theta_z

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

    def get_neighborhood_points(self, cloud, radius=250, count_threshold=100):
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
        
        # 重複がないように領域内で点群数がcount_threshold以上の範囲をピックアップ
        area_points_list = []
        area_center_point_list = []
        # ピックアップされていない点群のインデックス
        reserve_points_idx = [i for i in range(len(count_under_threshold))]
        for idx in idxs:
            # 領域内の点群のインデックスを取得
            under_idx = indicesm[idx, np.where(sqr_distances[idx]<radius**2)[0]]

            # 点群数が100未満の場合 or すでにピックアップされた点群を含む場合はスキップ
            if len(under_idx)<count_threshold:
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
                before_bentch_point = integraded_area_center_point_list[group_idx][time_idx-1]
                after_bentch_point = integraded_area_center_point_list[group_idx][time_idx]

                if len(before_bentch_point)==0 and len(after_bentch_point)==0:
                    vectors.append([])
                elif len(before_bentch_point)==0 and len(after_bentch_point)>0:
                    vectors.append([])
                elif len(before_bentch_point)>0 and len(after_bentch_point)>0:
                    vectors.append(after_bentch_point - before_bentch_point)
                elif len(before_bentch_point)>0 and len(after_bentch_point)==0:
                    vectors.append([])
            vectors_list.append(vectors)
        
        return vectors_list
    
    def get_vector_2(self, integraded_area_points_list, percent=None, height=None):
        """
        統合された点群のグループの中心点からベクトルを取得
        引数:
            integraded_area_center_point_list: list
        返り値:
            vector_list: list
        """
        vectors_list = []
        for group_idx in range(len(integraded_area_points_list)):
            vectors = []
            for time_idx in range(1, len(integraded_area_points_list[group_idx])):
                before_count = len(integraded_area_points_list[group_idx][time_idx-1])
                after_count = len(integraded_area_points_list[group_idx][time_idx])

                if before_count>0 and after_count>0:
                    before_bentch_point = self.get_bentchmark(integraded_area_points_list[group_idx][time_idx-1], percent, height)
                    after_bentch_point = self.get_bentchmark(integraded_area_points_list[group_idx][time_idx], percent, height)
                    vectors.append(after_bentch_point - before_bentch_point)
                else:
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

    def get_bentchmark(self, points, percent=None, height=None):
        """
        点群のある領域の高さの平均値を取得
        引数:
            points: np.array
            percent: list [min_percent, max_percent]
            height: list [min_height, max_height]
        返り値:
            bench_height: float

        points: 点群
        percent: 下位何%から上位何%までの点の平均を高さとするか
        height: 下限と上限の高さ間の点の平均を高さとするか
        """
        points_sort = points[np.argsort(points[:, 2])]

        if percent is not None:
            points_sort = points_sort[int(len(points_sort)*percent[0]/100):int(len(points_sort)*percent[1]/100)]
            bench_point = np.mean(points_sort, axis=0)
        elif height is not None:
            points_sort = points_sort[(points_sort[:, 2]>height[0]) & (points_sort[:, 2]<height[1])]
            if len(points_sort)==0:
                return []
            bench_point = np.mean(points_sort, axis=0)
        else:
            bench_point = np.mean(points_sort, axis=0)
        
        return bench_point

    def grouping_points_list_2(self, integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=0.1, judge_move_threshold=1000, is_incline=True):
        fit_list = []
        time_idx_list = []
        x_list = []
        y_list = []

        move_flg_list = self.judge_move(self.get_vector(integraded_area_center_point_list), threshold=judge_move_threshold)
        for group_idx in range(len(integraded_area_center_point_list)):
            if move_flg_list[group_idx]:
                # 中心点のxy座標の移動に対して、近似曲線を取得
                time_idxs = [time_idx for time_idx, point in enumerate(integraded_area_center_point_list[group_idx]) if len(point)>0]
                x = [point[0] for point in integraded_area_center_point_list[group_idx] if len(point)>0]
                y = [point[1] for point in integraded_area_center_point_list[group_idx] if len(point)>0]

                res = np.polyfit(x, y, 1)
                if len(fit_list)==0:
                    fit_list.append(res[0])
                    time_idx_list = time_idxs
                    x_list = x
                    y_list = y
                else:
                    if abs(fit_list[-1]-res[0])<1:
                        fit_list.append(res[0])
                        time_idx_list += time_idxs
                        x_list += x
                        y_list += y
        
        before_idx = None
        no_data = []
        new_time_idx_list = []
        new_x_list = []
        new_y_list = []
        for idx, time_idx in enumerate(time_idx_list):
            if before_idx is not None:
                if time_idx-before_time_idx>1:
                    x_step = (x_list[idx]-x_list[before_idx])/(time_idx-before_time_idx)
                    y_step = (y_list[idx]-y_list[before_idx])/(time_idx-before_time_idx)

                    for i in range(1, time_idx-before_time_idx):
                        no_data.append(time_idx)
                        new_time_idx_list.append(before_time_idx+i)
                        new_x_list.append(x_list[before_idx]+x_step*i)
                        new_y_list.append(y_list[before_idx]+y_step*i)
            
            new_time_idx_list.append(time_idx)
            new_x_list.append(x_list[idx])
            new_y_list.append(y_list[idx])
            
            before_idx = idx
            before_time_idx = time_idx

        # step_secを変更する場合
        new_time_idx_list_2 = []
        new_x_list_2 = []
        new_y_list_2 = []
        if sec<0.1:
            for i in range(1, len(new_time_idx_list)):
                before_time_idx = new_time_idx_list[i-1]
                after_time_idx = new_time_idx_list[i]
                before_x = new_x_list[i-1]
                after_x = new_x_list[i]
                before_y = new_y_list[i-1]
                afetr_y = new_y_list[i]

                step_time = sec/0.1
                step_x = (after_x-before_x)/(0.1/sec)
                step_y = (afetr_y-before_y)/(0.1/sec)
                for j in range(int(0.1/sec)):
                    new_time_idx_list_2.append(before_time_idx+step_time*j)
                    new_x_list_2.append(before_x+step_x*j)
                    new_y_list_2.append(before_y+step_y*j)
            else:
                new_time_idx_list_2.append(after_time_idx)
                new_x_list_2.append(after_x)
                new_y_list_2.append(afetr_y)
        else:
            new_time_idx_list_2 = new_time_idx_list
            new_x_list_2 = new_x_list
            new_y_list_2 = new_y_list

        new_integraded_area_points_list = [] 
        new_integraded_area_center_point_list = []

        # 傾きを取得
        if is_incline:
            pcd_info_list = get_pcd_information.get_pcd_information()
            pcd_info_list.load_pcd_dir(cloud_folder_path)
            theta_x, theta_y, theta_z = self.cloud_get_tilt(pcd_info_list, upper_threshold=2000-1300)
        else:
            theta_x = 0
            theta_y = 0
            theta_z = 0

        for group_idx in range(1):
            new_integraded_area_points_list.append([])
            new_integraded_area_center_point_list.append([])

            for idx in range(len(integraded_area_points_list[0])):
                for step in range(int(0.1/sec)):
                    time_idx = idx+(sec/0.1)*step
                    if time_idx in new_time_idx_list_2:
                        idx_2 = new_time_idx_list_2.index(time_idx)
                        cloud_path = f"{cloud_folder_path}/{str(int(time_idx*(0.1/sec)+1))}.pcd"
                        if os.path.exists(cloud_path):
                            cloud = pcl.load(cloud_path)

                            # 傾きの補正
                            cloud = self.def_method.rotate_cloud(cloud, -theta_x, theta_y)
                            # 高さの補正
                            points = np.array(cloud)
                            points[:, 2] = points[:, 2] + 1300
                            cloud = self.def_method.get_cloud(points)

                            base_x = new_x_list_2[idx_2]
                            base_y = new_y_list_2[idx_2]
                            cloud_filtered = self.def_method.filter_area(cloud, base_x-250, base_x+250, base_y-250, base_y+250, 0, 1700)
                            
                            if cloud_filtered.size>0:
                                points_filtered = np.array(cloud_filtered)
                                center_point = np.mean(points_filtered, axis=0)
                                
                                new_integraded_area_points_list[group_idx].append(points_filtered)
                                new_integraded_area_center_point_list[group_idx].append(center_point)
                            else:
                                new_integraded_area_points_list[group_idx].append([])
                                new_integraded_area_center_point_list[group_idx].append([])
                        else:
                            new_integraded_area_points_list[group_idx].append([])
                            new_integraded_area_center_point_list[group_idx].append([])
        
        return new_integraded_area_points_list, new_integraded_area_center_point_list


    ## 回収中
    def grouping_points_list_2_new(self, integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=0.1, is_incline=True):
        fit_list_dic = {}
        time_idx_list_dic = {}
        x_list_dic = {}
        y_list_dic = {}
        gropu_idx_dic = {}

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        move_flg_list = self.judge_move(self.get_vector(integraded_area_center_point_list), threshold=1000)
        for group_idx in range(len(integraded_area_center_point_list)):
            if move_flg_list[group_idx]:
                # 中心点のxy座標の移動に対して、近似曲線を取得
                time_idxs = [time_idx for time_idx, point in enumerate(integraded_area_center_point_list[group_idx]) if len(point)>0]
                x = [point[0] for point in integraded_area_center_point_list[group_idx] if len(point)>0]
                y = [point[1] for point in integraded_area_center_point_list[group_idx] if len(point)>0]

                res = np.polyfit(x, y, 1)
                ax.plot(x, y, label=f"{group_idx}")
                ax.plot(x, np.poly1d(res)(x), label=f"{group_idx}_fit")
                if len(fit_list_dic)==0:
                    fit_list_dic[0] = [res[0]]
                    time_idx_list_dic[0] = time_idxs
                    x_list_dic[0] = x
                    y_list_dic[0] = y
                    gropu_idx_dic[0] = [group_idx]
                else:
                    for key in list(fit_list_dic.keys()):
                        points1 = np.array([x_list_dic[key][-1], y_list_dic[key][-1]])
                        points2 = np.array([x[0], y[0]])
                        if abs(fit_list_dic[key][-1]-res[0])<1 and abs(self.calc_points_distance(points1, points2))<500:
                            fit_list_dic[key].append(res[0])
                            time_idx_list_dic[key] += time_idxs
                            x_list_dic[key] += x
                            y_list_dic[key] += y
                            gropu_idx_dic[key].append(group_idx)
                            break
                        else:
                            new_key = len(fit_list_dic.keys())
                            fit_list_dic[new_key] = [res[0]]
                            time_idx_list_dic[new_key] = time_idxs
                            x_list_dic[new_key] = x
                            y_list_dic[new_key] = y
                            gropu_idx_dic[new_key] = [group_idx]
                            break
        ax.legend()
        plt.show()
        plt.close()
        
        before_idx = None
        no_data = []
        new_time_idx_list = []
        new_x_list = []
        new_y_list = []
        
        new_integraded_area_points_list = [] 
        new_integraded_area_center_point_list = []
        for key in list(fit_list_dic.keys()):
            fit_list = fit_list_dic[key]
            time_idx_list = time_idx_list_dic[key]
            x_list = x_list_dic[key]
            y_list = y_list_dic[key]
            for idx, time_idx in enumerate(time_idx_list):
                if before_idx is not None:
                    if time_idx-before_time_idx>1:
                        x_step = (x_list[idx]-x_list[before_idx])/(time_idx-before_time_idx)
                        y_step = (y_list[idx]-y_list[before_idx])/(time_idx-before_time_idx)

                        for i in range(1, time_idx-before_time_idx):
                            no_data.append(time_idx)
                            new_time_idx_list.append(before_time_idx+i)
                            new_x_list.append(x_list[before_idx]+x_step*i)
                            new_y_list.append(y_list[before_idx]+y_step*i)
                
                new_time_idx_list.append(time_idx)
                new_x_list.append(x_list[idx])
                new_y_list.append(y_list[idx])
                
                before_idx = idx
                before_time_idx = time_idx

            # step_secを変更する場合
            new_time_idx_list_2 = []
            new_x_list_2 = []
            new_y_list_2 = []
            if sec<0.1:
                for i in range(1, len(new_time_idx_list)):
                    before_time_idx = new_time_idx_list[i-1]
                    after_time_idx = new_time_idx_list[i]
                    before_x = new_x_list[i-1]
                    after_x = new_x_list[i]
                    before_y = new_y_list[i-1]
                    afetr_y = new_y_list[i]

                    step_time = sec/0.1
                    step_x = (after_x-before_x)/(0.1/sec)
                    step_y = (afetr_y-before_y)/(0.1/sec)
                    for j in range(int(0.1/sec)):
                        new_time_idx_list_2.append(before_time_idx+step_time*j)
                        new_x_list_2.append(before_x+step_x*j)
                        new_y_list_2.append(before_y+step_y*j)
                else:
                    new_time_idx_list_2.append(after_time_idx)
                    new_x_list_2.append(after_x)
                    new_y_list_2.append(afetr_y)
            else:
                new_time_idx_list_2 = new_time_idx_list
                new_x_list_2 = new_x_list
                new_y_list_2 = new_y_list

            # 傾きを取得
            if is_incline:
                pcd_info_list = get_pcd_information.get_pcd_information()
                pcd_info_list.load_pcd_dir(cloud_folder_path)
                theta_x, theta_y, theta_z = self.cloud_get_tilt(pcd_info_list, upper_threshold=2000-1300)
            else:
                theta_x = 0
                theta_y = 0
                theta_z = 0
            new_integraded_area_points_list.append([])
            new_integraded_area_center_point_list.append([])
            for idx in range(len(integraded_area_points_list[0])):
                for step in range(int(0.1/sec)):
                    time_idx = idx+(sec/0.1)*step
                    if time_idx in new_time_idx_list_2:
                        idx_2 = new_time_idx_list_2.index(time_idx)
                        cloud_path = f"{cloud_folder_path}/{str(int(time_idx*(0.1/sec)+1))}.pcd"
                        if os.path.exists(cloud_path):
                            cloud = pcl.load(cloud_path)

                            # 傾きの補正
                            cloud = self.def_method.rotate_cloud(cloud, -theta_x, theta_y)
                            # 高さの補正
                            points = np.array(cloud)
                            points[:, 2] = points[:, 2] + 1300
                            cloud = self.def_method.get_cloud(points)

                            base_x = new_x_list_2[idx_2]
                            base_y = new_y_list_2[idx_2]
                            cloud_filtered = self.def_method.filter_area(cloud, base_x-250, base_x+250, base_y-250, base_y+250, 0, 1700)
                            if cloud_filtered.size>0:
                                points_filtered = np.array(cloud_filtered)
                                center_point = np.mean(points_filtered, axis=0)
                                
                                new_integraded_area_points_list[-1].append(points_filtered)
                                new_integraded_area_center_point_list[-1].append(center_point)
                            else:
                                new_integraded_area_points_list[-1].append([])
                                new_integraded_area_center_point_list[-1].append([])
                        else:
                            new_integraded_area_points_list[-1].append([])
                            new_integraded_area_center_point_list[-1].append([])
            
        return new_integraded_area_points_list, new_integraded_area_center_point_list

    # 身長を取得
    def get_height_all(self, cloud_list, top_percent=0.5):
        all_points = None
        for cloud in cloud_list:
            if all_points is None:
                all_points = np.array(cloud)
            else:
                all_points = np.vstack((all_points, np.array(cloud)))

        z = all_points[:, 2]
        z = np.sort(z)[::-1]
        z = z[:int(len(z)*top_percent/100)]

        return np.mean(z)
    
    # 進行方向に沿って, y軸が基準になるように回転させる角度を取得
    def get_collect_theta_z(self, integraded_area_center_point_list):
        vector_list = self.get_vector(integraded_area_center_point_list)
        move_flg_list = self.judge_move(vector_list, threshold=1000)
        collect_theta_z_list = []

        for group_idx in range(len(integraded_area_center_point_list)):
            if not move_flg_list[group_idx]:
                collect_theta_z_list.append(0)
            else:
                vector = vector_list[group_idx]
                all_vector = np.sum(vector, axis=0)
                theta_z = np.arctan(-1*(all_vector[0]/all_vector[1]))
                collect_theta_z_list.append(theta_z)

        return collect_theta_z_list

    def normalization_points(self, points, center_point, x_flg=True, y_flg=True, z_flg=False):
        """
        点群を中心点をcenter_pointに正規化
        引数:
            points: np.array
            center_point: np.array
            x_flg: bool
            y_flg: bool
            z_flg: bool
        返り値:
            points: np.array
        """
        normalized_points = points.copy()
        if x_flg:
            normalized_points[:, 0] = points[:, 0] - center_point[0]
        if y_flg:
            normalized_points[:, 1] = points[:, 1] - center_point[1]
        if z_flg:
            normalized_points[:, 2] = points[:, 2] - center_point[2]
        
        return normalized_points

    # 進行方向に沿って, y軸が基準になるように回転
    def rotate_collect_cloud(self, cloud, theta_z):
        cloud = self.def_method.rotate_cloud(cloud, 0, 0, theta_z)
        return cloud
