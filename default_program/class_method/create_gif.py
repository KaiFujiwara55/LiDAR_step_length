import os
import datetime
import imageio
import matplotlib.pyplot as plt

class create_gif:
    """画像をGIFに変換するクラス"""
    def __init__(self, create_flg=True):
        self.create_flg = create_flg
        if not self.create_flg:
            return

        self.image_paths = []
        self.tmp_dir = f"{os.getcwd()}/tmp_imgage_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 一時ディレクトリを作成
        self.create_dic(self.tmp_dir)

    def create_dic(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_fig(self, fig: plt.Figure):
        """GIF画像用にfig毎に保存する関数"""
        if not self.create_flg:
            return
        # 画像を保存
        image_path = os.path.join(self.tmp_dir, f"plot_{len(self.image_paths)}.png")
        fig.savefig(image_path)
        self.image_paths.append(image_path)

    def create_gif(self, output_path: str, duration: float=0.1, remove_flg=True):
        """GIF画像を作成する関数"""
        if not self.create_flg:
            return
        
        # 一時保存された画像がない場合は処理を終了
        if len(self.image_paths) == 0:
            self.remove()
            return

        # 出力先のディレクトリが存在しない場合は作成
        self.create_dic("/".join(output_path.split("/")[:-1]))

        # 画像をGIFに変換
        images = []
        for image_path in self.image_paths:
            images.append(imageio.v2.imread(image_path))

        fps_num = 1/duration
        imageio.mimsave(output_path, images, fps=fps_num, loop=0)

        if remove_flg:
            self.remove()

    def remove(self):
        """一時ディレクトリ・ファイルを削除する関数"""
        if not self.create_flg:
            return
        self.remove_images()
        self.remove_dir()

    def remove_images(self):
        """一時ディレクトリ内の画像を削除する関数"""
        if not self.create_flg:
            return
        for image_path in self.image_paths:
            os.remove(image_path)

    def remove_dir(self):
        """一時ディレクトリを削除する関数"""
        if not self.create_flg:
            return
        os.rmdir(self.tmp_dir)


