import pcl
import pcl.pcl_visualization
import matplotlib.pyplot as plt
import numpy as np

def detect_ceiling_plane(cloud, distance_threshold=100, max_iterations=1000, probability=0.99):
    # SACSegmentationオブジェクトの作成
    seg = cloud.make_segmenter()

    # 平面モデルの設定
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)

    # 平面のセグメンテーションを実行
    indices, coefficients = seg.segment()

    if len(indices) == 0:
        return None, None

    # 平面の係数を取得
    a, b, c, d = coefficients

    return cloud.extract(indices, negative=False), cloud.extract(indices, negative=True), coefficients, indices

# どの平面かを判定する
def decide_plane(cloud, ceiling_cloud, ceiling_coefficients):
    points = np.array(cloud)
    max_width = np.max(points[:, 0])
    max_height = np.max(points[:, 2])
    min_height = np.min(points[:, 2])
    
    a, b, c, d = ceiling_coefficients
    # 天井・床面の抽出
    if abs(a)<0.2 and abs(b)<0.2 and abs(c)>0.8:
        if d/c*(-1)>max_height*0.75:
            return "upper"
        elif d/c*(-1)<min_height*0.75:
            return "lower"
        
    # 壁面の抽出
    if abs(a)>0.8 and abs(b)<0.2 and abs(c)<0.2:
        if d/a*(-1)>max_width*0.6:
            return "wall"
    return "else"

def remove_points(input_cloud, indices_cloud):
    # indices_cloudの点をセットに変換 (set) します
    indices_points = set(map(tuple, np.asarray(indices_cloud)))
    
    # input_cloudの点をフィルタリングし、indices_cloudに含まれない点だけを残します
    filtered_points = np.array([point for point in np.asarray(input_cloud) if tuple(point) not in indices_points])
    
    # 新しい点群を作成
    filtered_cloud = pcl.PointCloud()
    filtered_cloud.from_array(filtered_points)
    
    return filtered_cloud


def main(scan_path):
    # PCDファイルの読み込み
    source_cloud = pcl.load(scan_path)
    points = np.array(source_cloud)
    
    # 天井の平面を検出
    tmp_cloud = source_cloud
    filtered_cloud = source_cloud
    for i in range(10):
        ceiling_cloud, non_ceiling_cloud, ceiling_coefficients, indices = detect_ceiling_plane(tmp_cloud, 100)

        plane = decide_plane(tmp_cloud, ceiling_cloud, ceiling_coefficients)
        if plane=="upper" or plane=="lower" or plane=="wall":
            filtered_cloud = remove_points(filtered_cloud, ceiling_cloud)
        
        tmp_cloud = non_ceiling_cloud

    return filtered_cloud
