import pcl
import pcl.pcl_visualization
import matplotlib.pyplot as plt
import numpy as np
import get_segment
import glob

import create_gif
import get_pcd_information

def surface_integrate(surface_xy_list, threshold=500):
    idx_groups = []
    idx_pairs = []
    pair_flg = {idx:False for idx in range(len(surface_xy_list))}

    for idx in range(len(surface_xy_list)):
        x, y = surface_xy_list[idx]
        for idx2 in range(idx+1, len(surface_xy_list)):
            x2, y2 = surface_xy_list[idx2]
            if abs(x+y-x2-y2)<=threshold:
                # 統合させないためにフラグを立てる
                if False:
                    idx_pairs.append([idx, idx2])
                    pair_flg[idx] = True
                    pair_flg[idx2] = True

    idx_groups = merge_groups(idx_pairs)

    for idx in range(len(surface_xy_list)):
        if not pair_flg[idx]:
            idx_groups.append([idx])

    integraded_surface_xy_list = []
    for idx_group in idx_groups:
        tmp = []
        for idx in idx_group:
            tmp.append(surface_xy_list[idx])
        integraded_surface_xy_list.append(tmp)

    return integraded_surface_xy_list

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

# PCDファイルの読み込み
folders = glob.glob("/Users/kai/大学/4年春/小川研/LIDAR/20240703/lidar_scan_data_0703/pcd/*")

for folder in folders:
    print(folder)
    files = glob.glob(folder+"/*")
    # ファイルの文字列の数値タイトル順に変形する
    file_idx = [[idx, int(file_name.split("/")[-1].split(".")[0])] for idx, file_name in enumerate(files)]
    file_idx = sorted(file_idx, key=lambda x: x[1])
    files = [files[idx[0]] for idx in file_idx]
    gif = create_gif.create_gif(False)
    
    pcd_information = get_pcd_information.get_pcd_information()
    pcd_information.load_pcd_dir(folder)

    try:
        for file in files:
            cloud = get_segment.main(file)
            points = np.array(cloud)
            threshold = 100000
            
            # 50cm四方の領域内の点群の高さの分散を出す
            fig = plt.figure()
            ax_2d = fig.add_subplot(121)
            ax_3d = fig.add_subplot(122, projection='3d')

            surface_xy_list = []
            for x in range(int(pcd_information.get_all_min()[0]), int(pcd_information.get_all_max()[0]), 500):
                for y in range(int(pcd_information.get_all_min()[1]), int(pcd_information.get_all_max()[1]), 500):
                    # 50cm四方の領域内の点群を取得
                    points_in_square = points[(points[:, 0] > x) & (points[:, 0] < x+500) & (points[:, 1] > y) & (points[:, 1] < y+500)]
                    # 高さの分散を出す
                    if len(points_in_square)>0:
                        height_variance = np.var(points_in_square[:, 2])
                        if height_variance > threshold:
                            square_points = np.array(points_in_square)
                            surface_xy_list.append([x, y])
            
            # 50cm四方の領域内の点群を統合
            integraded_surface_xy_list = surface_integrate(surface_xy_list, threshold=500)
            color_list = ["r", "g", "c", "m", "y", "k", "w"]*100
            for idx, surface_xy_list in enumerate(integraded_surface_xy_list):
                points_len_list = [len(points[(points[:, 0] > x) & (points[:, 0] < x+500) & (points[:, 1] > y) & (points[:, 1] < y+500)]) for x, y in surface_xy_list]
                sorted_idx = np.argsort(points_len_list)
                print(sorted_idx)
                for surface_xy in surface_xy_list:
                    x = surface_xy[0]
                    y = surface_xy[1]
                    points_in_square = points[(points[:, 0] > x) & (points[:, 0] < x+500) & (points[:, 1] > y) & (points[:, 1] < y+500)]
                    ax_3d.scatter(points_in_square[:, 0], points_in_square[:, 1], points_in_square[:, 2], s=1, c=color_list[idx], label=f"{idx}" )

                    ax_2d.scatter(points_in_square[:, 0], points_in_square[:, 1], s=1, c=color_list[idx], label=f"{idx}")
                    # center_point = np.mean(points_in_square, axis=0)
                    # ax_3d.scatter(center_point[0], center_point[1], center_point[2], s=10, c="orange")



            
            if True:
                # 軸の範囲の設定
                ax_2d.set_xlim(pcd_information.get_all_min()[0], pcd_information.get_all_max()[0])
                ax_2d.set_ylim(pcd_information.get_all_min()[1], pcd_information.get_all_max()[1])

                ax_3d.set_xlim(pcd_information.get_all_min()[0], pcd_information.get_all_max()[0])
                ax_3d.set_ylim(pcd_information.get_all_min()[1], pcd_information.get_all_max()[1])
                ax_3d.set_zlim(pcd_information.get_all_min()[2], pcd_information.get_all_max()[2])
                # 軸ラベルの設定
                ax_2d.set_xlabel('X')
                ax_2d.set_ylabel('Y')

                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                # 凡例の設定
                ax_2d.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=5)
                # ax_3d.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=5)
                # 視点の設定
                ax_3d.view_init(azim=150)
                # タイトルの設定
                title = file.split("/")[-2].split(".")[0]+"_"+file.split("/")[-1].split(".")[0]

                ax_2d.set_title("2d_"+title)
                ax_3d.set_title("3d_"+title)
                
                plt.pause(0.025)

                gif.save_fig(fig)
                
        plt.close(fig)
        output_path = '/Users/kai/大学/4年春/小川研/LIDAR/20240703/lidar_scan_data_0703/gif/'+folder.split('/')[-1]+'.gif'
        # gif.create_gif(output_path, duration=0.1)
        gif.remove()
    except KeyboardInterrupt:
        gif.remove()
