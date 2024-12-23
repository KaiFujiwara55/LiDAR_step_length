import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import glob
from tqdm import tqdm
from scipy.signal import find_peaks

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot
from default_program.class_method import create_gif

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()
set_ax = plot.set_plot()
set_ax.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=[0, 10000], ylim=[-5000, 5000], zlim=[0, 2000], azim=150)

# 原点から円の接線を求める
def get_tangent_line(circle_center, circle_radius):
    a = circle_center[0]
    b = circle_center[1]
    r = circle_radius
    
    # 接線の傾きを求める方程式をとく
    c1 = (a*b + np.sqrt(a**2*r**2 + b**2*r**2 - r**4))/(a**2 - b**2)
    c2 = (a*b - np.sqrt(a**2*r**2 + b**2*r**2 - r**4))/(a**2 - b**2)


    # 接点を求める
    x1 = (a + b*c1)/(c1**2 + 1)
    y1 = c1*x1
    x2 = (a + b*c2)/(c2**2 + 1)
    y2 = c2*x2

    return x1, y1, x2, y2

# 原点から楕円の接線を求める
def get_tangent_line_ellipse(ellipse_center, ellipse_minor_axis, elipse_major_axis):
    a = ellipse_center[0]
    b = ellipse_center[1]
    minor_axis = ellipse_minor_axis
    major_axis = elipse_major_axis
    
    # 接線の傾きを求める方程式をとく
    c1 = (-a*b + np.sqrt(major_axis**2*b**2 + minor_axis**2*a**2 - major_axis**2*minor_axis**2))/(major_axis**2 - a**2)
    c2 = (-a*b - np.sqrt(major_axis**2*b**2 + minor_axis**2*a**2 - major_axis**2*minor_axis**2))/(major_axis**2 - a**2)

    return c1, c2

def get_fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = set_ax.set_ax(ax, title="projection", xlim=[0, 10000], ylim=[-5000, 5000])
    # 補助線をひく
    square_x = np.linspace(0, 7000/np.sqrt(2), 100)
    y_square_down = np.tan(np.radians(-45))*square_x
    y_square_up = np.tan(np.radians(45))*square_x
    ax.plot(square_x, y_square_down, c="yellow", label="room_area")
    ax.plot(square_x, y_square_up, c="yellow")
    square_x = np.linspace(7000/np.sqrt(2), 7000/np.sqrt(2)*2, 100)
    y_square_down = np.tan(np.radians(-45))*square_x+7000/np.sqrt(2)*2
    y_square_up = np.tan(np.radians(45))*square_x-7000/np.sqrt(2)*2
    ax.plot(square_x, y_square_down, c="yellow")
    ax.plot(square_x, y_square_up, c="yellow")
    x = np.linspace(0, 12000, 100)
    y_down = np.tan(np.radians(-35.2))*x
    y_up = np.tan(np.radians(35.2))*x
    ax.plot(x, y_down, c="black", label="avia_range")
    ax.plot(x, y_up, c="black")

    return fig, ax

# 人物の軌跡の出発点・到着点を設定
start_point = [7000/np.sqrt(2)/2, 7000/np.sqrt(2)/2]
end_point = [7000/np.sqrt(2)*1.5, -7000/np.sqrt(2)/2]
angle = 0

for i in range(1, 100):
    x = (end_point[0] - start_point[0]) * i / 100 + start_point[0]
    y = (end_point[1] - start_point[1]) * i / 100 + start_point[1]
    
    fig, ax = get_fig_ax()
    
    circle_center = (x, y)
    circle_radius = 250
    

    # x1, y1, x2, y2 = get_tangent_line(circle_center, circle_radius)
    # print((x1, y1), (x2, y2))

    # 楕円を表示
    eclips_center = (x, y)
    minor_axis = 50
    major_axis = 250
    angle = 45

    ellipse = patches.Ellipse(xy=eclips_center, width=major_axis*2, height=minor_axis*2, angle=angle, fill=False)
    ax.add_patch(ellipse)

    c1, c2 = get_tangent_line_ellipse(eclips_center, minor_axis, major_axis)

    x1 = np.linspace(0, 10000, 100)
    y1 = c1*x1
    x2 = np.linspace(0, 10000, 100)
    y2 = c2*x2

    ax.plot(x1, y1, c="red")
    ax.plot(x2, y2, c="red")

    plt.show()
    plt.close()


