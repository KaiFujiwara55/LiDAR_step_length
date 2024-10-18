import numpy as np
import matplotlib.pyplot as plt
import math


readpath = 'C:/Users/ogawalab/Desktop/hokuyo/hokuyolx-master/hokuyolx-master/examples/test/'

# 時刻情報
read_time = np.loadtxt(readpath+'scandata3_diff_time.csv', delimiter=',')
# 角度情報
read_angle = np.loadtxt(readpath+'scandata0_diff.csv', delimiter=',')
# 測定情報
read_dist = np.loadtxt(readpath+'scandata1_diff.csv', delimiter=',')
# 追跡情報
read_track = np.loadtxt(readpath+'scandata1_diff_trackall.csv', delimiter=',')


fig = plt.figure()
ax = plt.subplot(111, projection='polar')
# 軸を90度回転
ax.set_theta_offset(math.pi/2)
plot1 = ax.plot([], [], '.',color ='g')[0]
plot2 = ax.plot([], [], 'o', markersize = 10, color ='r')[0]
text = plt.text(0, 1, '', transform=ax.transAxes)
# 軸の長さ
ax.set_rmax(6000)
ax.grid(True)

for num in range (len(read_time)):
    plot1.set_data(read_angle[num], read_dist[num])
    if sum(read_track[num])>0:
        plot2.set_data(read_angle[num][np.where(read_track[num]>0)], read_track[num][np.where(read_track[num]>0)])
    else:
        plot2.set_data([], [])
    text.set_text('t: %.3lf' % float(read_time[num]))
    plt.draw()
    if (num < len(read_time)-1):
        plt.pause(float(read_time[num+1])-float(read_time[num]))
plt.close()