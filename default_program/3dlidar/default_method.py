import numpy as np
import pcl

class cloud_method:
    """3次元点群処理の基本的な処理をまとめたクラス"""
    
    def get_points(self, cloud):
        return np.array(cloud)
    
    def get_cloud(self, points):
        cloud = pcl.PointCloud()
        cloud.from_array(points.astype(np.float32))
        return cloud

    def kdtree_search(self, cloud, k=10):
        """
        K近傍法
        引数:
            cloud: pcl.PointCloud
            k: int
        返り値:
            indices: np.array
            sqr_distances: np.array
        
        k: クエリ点に対して最近傍点の数
        indices: クエリ点に最も近い点のインデックス
        sqr_distances: クエリ点と最も近い点の距離
        https://github.com/strawlab/python-pcl/blob/master/examples/official/kdtree/kdtree_search.py
        """
        kdtree = cloud.make_kdtree_flann()
        indices, sqr_distances = kdtree.nearest_k_search_for_cloud(cloud, k)

        return indices, sqr_distances

    def kdtree_search_all(self, cloud):
        """
        全点のK近傍法
        引数:
            cloud: pcl.PointCloud
        返り値:
            indices: np.array
            sqr_distances: np.array
        
        indices: クエリ点に最も近い点のインデックス
        sqr_distances: クエリ点と最も近い点の距離
        """
        kdtree = cloud.make_kdtree_flann()
        indices, sqr_distances = kdtree.nearest_k_search_for_cloud(cloud, len(np.array(cloud)))

        return indices, sqr_distances

    def voxel_grid_filter(self, cloud, leaf_size=(100, 100, 100)):
        """
        VoxelGridフィルタによるダウンサンプリング
        引数:
            cloud: pcl.PointCloud
            leaf_size: tuple
        返り値:
            cloud_filtered: pcl.PointCloud
        
        leaf_size: グリッドの一辺の長さ
        https://github.com/strawlab/python-pcl/blob/master/examples/official/Filtering/VoxelGrid_160.py
        """
        sor = cloud.make_voxel_grid_filter()
        sor.set_leaf_size(*leaf_size)
        cloud_filtered = sor.filter()

        return cloud_filtered

    # うまく動かん
    def radius_outlier_removal(self, cloud, radius=10, min_neighbors=0):
        """
        半径外れ値除去
        引数:
            cloud: pcl.PointCloud
            radius: float
            min_neighbors: int
        返り値:
            cloud_filtered: pcl.PointCloud

        radius: クエリ点に対して近傍点を探す半径
        min_neighbors: クエリ点に対して近傍点の最小数
        https://github.com/strawlab/python-pcl/blob/master/examples/official/Filtering/remove_outliers.py
        """
        sor = cloud.make_RadiusOutlierRemoval()
        sor.set_radius_search(radius)
        sor.set_MinNeighborsInRadius(min_neighbors)
        cloud_filtered = sor.filter()

        return cloud_filtered

    def statistical_outlier_removal(self, cloud, mean_k=50, std_dev_mul_thresh=1.0):
        """
        統計的外れ値除去
        引数:
            cloud: pcl.PointCloud
            mean_k: int
            std_dev_mul_thresh: float
        返り値:
            cloud_filtered: pcl.PointCloud
        
        mean_k: 分散計算を行う近傍点の数
        std_dev_mul_thresh: 外れ値と見なす分散の割合
        https://github.com/strawlab/python-pcl/blob/master/examples/official/Segmentation/Plane_model_segmentation.py
        """
        sor = cloud.make_statistical_outlier_filter()
        sor.set_mean_k(mean_k)
        sor.set_std_dev_mul_thresh(std_dev_mul_thresh)
        cloud_filtered = sor.filter()

        return cloud_filtered

    def segment_plane(self, cloud, ksearch=50, distance_threshold=10):
        """
        平面のセグメンテーション
        引数:
            cloud: pcl.PointCloud
            ksearch: int
            distance_threshold: float
        返り値:
            cloud_plane: pcl.PointCloud
            cloud_non_plane: pcl.PointCloud
            coefficients: np.array
            inliers: np.array

        ksearch: 平面の方程式を計算する際に使用する近傍点の数？
        distance_threshold: 平面として認識する最大距離
        cloud_plane: 平面として認識された点群
        cloud_non_plane: 平面として認識されなかった点群
        coefficients: 平面の方程式の係数
        inliers: 平面として認識された点のインデックス
        https://github.com/strawlab/python-pcl/blob/master/examples/official/Segmentation/Plane_model_segmentation.py
        """
        seg = cloud.make_segmenter_normals(ksearch)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(distance_threshold)
        seg.set_normal_distance_weight(0.1)
        seg.set_max_iterations(100)
        indices, coefficients = seg.segment()

        cloud_plane = cloud.extract(indices, negative=False)
        cloud_non_plane = cloud.extract(indices, negative=True)

        return cloud_plane, cloud_non_plane, coefficients, indices

    def calculate_distance_to_plane(self, cloud, coefficients):
        """
        平面からの距離を計算
        引数:
            cloud: pcl.PointCloud
            coefficients: np.array
        返り値:
            distances: np.array

        coefficients: 平面の方程式の係数
        distances: クエリ点から平面までの距離
        """
        a, b, c, d = coefficients
        points = np.array(cloud)
        distances = np.abs(a*points[:, 0] + b*points[:, 1] + c*points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

        return distances
    
    def filter_points_by_distance(self, cloud, coefficients, distance_threshold):
        """
        平面からの距離で点群をフィルタリング
        引数:
            cloud: pcl.PointCloud
            coefficients: np.array
            distance_threshold: float
        返り値:
            cloud_plane: pcl.PointCloud
            cloud_non_plane: pcl.PointCloud

        coefficients: 平面の方程式の係数
        distance_threshold: フィルタリングの閾値
        cloud_plane: 平面として認識された点群
        cloud_non_plane: 平面として認識されなかった点群
        """
        distances = self.calculate_distance_to_plane(cloud, coefficients)
        indices = np.where(distances < distance_threshold)[0]
        
        cloud_plane = cloud.extract(indices)
        cloud_non_plane = cloud.extract(indices, negative=True)

        return cloud_plane, cloud_non_plane

    def filter_by_height_var(self, cloud, threshold=100000, x_step=500, y_step=500):
        """
        高さの分散で点群をフィルタリング
        引数:
            cloud: pcl.PointCloud
            threshold: float
            x_step: int
            y_step: int
        返り値:
            cloud_filtered: pcl.PointCloud
            surface_xy_list: list

        x_step: 走査する領域のx方向の幅
        y_step: 走査する領域のy方向の幅
        threshold: フィルタリングの閾値
        surface_xy_list: 高さの分散が閾値以上の領域のxy座標のリスト
        cloud_filtered: 高さの分散が閾値以上の点群
        """
        points = np.array(cloud)
        points_filtered = []
        surface_xy_list = []
        for x in range(int(np.min(points[:, 0])), int(np.max(points[:, 0])), x_step):
            for y in range(int(np.min(points[:, 1])), int(np.max(points[:, 1])), y_step):
                points_in_square = points[(points[:, 0] > x) & (points[:, 0] < x+x_step) & (points[:, 1] > y) & (points[:, 1] < y+y_step)]
                if len(points_in_square) > 0:
                    height_variance = np.var(points_in_square[:, 2])
                    if height_variance > threshold:
                        surface_xy_list.append([x, y])
                        points_filtered.append(points_in_square)
        points_filtered = np.concatenate(points_filtered, axis=0)
        cloud_filtered = pcl.PointCloud()
        cloud_filtered.from_array(points_filtered)

        return cloud_filtered, surface_xy_list

    def filter_area(self, cloud, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
        """
        領域で点群をフィルタリング
        引数:
            cloud: pcl.PointCloud
            x_min: float
            x_max: float
            y_min: float
            y_max: float
            z_min: float
            z_max: float
        返り値:
            cloud_filtered: pcl.PointCloud
        
        領域の指定値がNoneの場合、その点群の最大値・最小値を使用
        """
        points = np.array(cloud)
        if x_min is None:
            x_min = np.min(points[:, 0])
        if x_max is None:
            x_max = np.max(points[:, 0])
        if y_min is None:
            y_min = np.min(points[:, 1])
        if y_max is None:
            y_max = np.max(points[:, 1])
        if z_min is None:
            z_min = np.min(points[:, 2])
        if z_max is None:
            z_max = np.max(points[:, 2])
        
        indices = np.where((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max))[0]
        cloud_filtered = cloud.extract(indices)

        return cloud_filtered

    def rotate_cloud(self, cloud, theta_x=0, theta_y=0, theta_z=0):
        """
        3次元点群の回転
        引数:
            cloud: pcl.PointCloud
            theta_x: float (radian)
            theta_y: float (radian)
            theta_z: float (radian)
        返り値:
            rot_pointcloud: np.array
            rot_matrix: np.array
        
        theta_x: x軸周りの回転角度
        theta_y: y軸周りの回転角度
        theta_z: z軸周りの回転角度
        
        https://tech-deliberate-jiro.com/pcl-rot/
        """

        points = np.array(cloud)

        rot_x = np.array(
            [[ 1, 0, 0],
            [ 0, np.cos(theta_x), -np.sin(theta_x)],
            [ 0, np.sin(theta_x),  np.cos(theta_x)]]
            )

        rot_y = np.array(
            [[ np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]]
            )

        rot_z = np.array(
            [[ np.cos(theta_z), -np.sin(theta_z), 0],
            [ np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]]
            )

        rot_matrix = rot_z.dot(rot_y.dot(rot_x))
        rot_points = rot_matrix.dot(points.T).T
        
        rot_cloud = pcl.PointCloud()
        rot_cloud.from_array(rot_points.astype(np.float32))

        return rot_cloud
