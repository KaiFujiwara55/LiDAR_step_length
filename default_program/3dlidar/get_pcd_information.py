import numpy as np
import glob
import pcl

class get_pcd_information:
    def __init__(self):
        self.cloud = None
        self.points = None
        self.dir_name = None
        self.cloud_list = None
        self.points_list = None
        self.cloud_name_list = None

    def load_pcd_dir(self, dir_path):
        """時系列分のPCDファイル全てを読み込む"""
        file_paths = glob.glob(dir_path + "/**/*.pcd", recursive=True)
        file_paths = self.sort_pcd_file_paths(file_paths)

        cloud_list = []
        points_list = []
        cloud_name_list = []
        for file_path in file_paths:
            self.load_pcd_from_file(file_path)
            cloud_list.append(self.cloud)
            points_list.append(self.points)
            cloud_name_list.append("_".join(file_path.split("/")[-2:]).split(".")[0])
        
        self.dir_name = dir_path.split("/")[-1]
        self.cloud_list = cloud_list
        self.points_list = points_list
        self.cloud_name_list = cloud_name_list
        
    def sort_pcd_file_paths(self, file_paths):
        """ファイルの文字列の数値タイトル順に変形する"""
        file_idx = [[idx, int(file_name.split("/")[-1].split(".")[0])] for idx, file_name in enumerate(file_paths)]
        file_idx = sorted(file_idx, key=lambda x: x[1])
        file_paths = [file_paths[idx[0]] for idx in file_idx]

        return file_paths

    def load_pcd_from_file(self, file_path):
        """PCDファイルの読み込み"""
        cloud = pcl.load(file_path)
        points = np.array(cloud)

        self.cloud = cloud
        self.points = points

    def load_pcd_from_cloud(self, cloud):
        """PCDファイルの読み込み"""
        points = np.array(cloud)

        self.cloud = cloud
        self.points = points


    def get_all_max(self):
        """時系列分のPCDファイル内での点群の座標の最大値を取得"""
        max = []
        for points in self.points_list:
            max.append(self.get_max(points))
        return np.max(max, axis=0)
    
    def get_all_min(self):
        """時系列分のPCDファイル内での点群の座標の最小値を取得"""
        min = []
        for points in self.points_list:
            min.append(self.get_min(points))

        return np.min(min, axis=0)

    def get_max(self, points):
        """点群の座標の最大値を取得"""
        return [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]

    def get_min(self, points):
        """点群の座標の最小値を取得"""
        return [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]

