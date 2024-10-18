'''Animates distances using single measurment mode'''
from hokuyolx import HokuyoLX
import matplotlib.pyplot as plt
import math
import csv
import numpy as np
import sys
import keyboard

DMAX = 10000
diff_distance = 50
diff_number = 5

#filter_angle の最大値は270
filter_angle = 180

# angle data -540, -539, ..., 539, 540. Number of data is 1081.
# degree from -135 to 135 total 270.
# angle data -280, -279, ..., 279, 280. Number of data is 561.
# degree from -70 to 70 total 140.

filter_angle_low = int(-filter_angle/2/0.25+540)
filter_angle_high = int(filter_angle_low+filter_angle/0.25+1)

angle_alldata=np.arange(-540, 541)*(0.25/360*(2*math.pi))
angle_alldata=np.ndarray.tolist(angle_alldata)

# 事前セットデータ
pathset = 'C:/Users/ogawalab/Desktop/hokuyo/hokuyolx-master/hokuyolx-master/examples/'
set_dist=np.loadtxt(pathset + 'scandata1_set.csv', delimiter=",")

def nulldatafunc(angle_rawdata, dist_rawdata):
    # angle_rawdata: scan.T[0], dist_rawdata: scan.T[1]
    # angle_rawdata is converted to index number
    index_num=(angle_rawdata+2.35619449019234)/(0.25/360*(2*math.pi))
    index_num=list(map(round, index_num))
    index_num=list(map(int, index_num))
    # Number of data is 1081.
    output_dist=np.zeros((1, 1081))
    output_dist=np.ndarray.tolist(output_dist)[0]

    for index, x in enumerate(index_num):
        output_dist[x]=dist_rawdata[index]
    return output_dist

def checksequence(x):
    # 連続した数に分割する
    result = []
    tmp = [x[0]]
    for i in range(len(x)-1):
        if x[i+1] - x[i] == 1:
            tmp.append(x[i+1])
        else:
            if len(tmp) > 0:
                result.append(tmp)
            tmp = []
            tmp.append(x[i+1])
    # 連続した数の集合を探す
    len_index=0
    len_max=0
    for i in range(len(result)):
        if len(result[i])>len_max:
            len_max=len(result[i])
            len_index=i
    return(result[len_index])

plt.ion()
laser = HokuyoLX()
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111, projection='polar')
# 軸を90度回転
ax.set_theta_offset(math.pi/2)
plot = ax.plot([], [], '.',color ='b')[0]
plot1 = ax.plot([], [], '.',color ='g')[0]
plot2 = ax.plot([], [], 'o', markersize = 10, color ='r')[0]

text = plt.text(0, 1, '', transform=ax.transAxes)
# 軸の長さ
ax.set_rmax(3000)
ax.grid(True)

f0=open('./scandata/'+sys.argv[1]+'_scandata0.csv', 'w', newline="")
f1=open('./scandata/'+sys.argv[1]+'_scandata1.csv', 'w', newline="")
f2=open('./scandata/'+sys.argv[1]+'_track.csv', 'w', newline="")
f3=open('./scandata/'+sys.argv[1]+'_trackall.csv', 'w', newline="")
f4=open('./scandata/'+sys.argv[1]+'_time.csv', 'w', newline="")

csvWriter0 = csv.writer(f0)
csvWriter1 = csv.writer(f1)
csvWriter2 = csv.writer(f2)
csvWriter3 = csv.writer(f3)
csvWriter4 = csv.writer(f4)


#for num in range(100):
while True:
    timestamp, scan = laser.get_filtered_dist(dmax=DMAX)
    
    # データ補完
    dist_alldata = nulldatafunc(scan.T[0], scan.T[1])
    dist_alldata_list = np.vstack([angle_alldata,dist_alldata])
    filter_dist_alldata_list = dist_alldata_list[:,filter_angle_low:filter_angle_high]

    plot.set_data(filter_dist_alldata_list[:,np.where(filter_dist_alldata_list[1,:]>0)])
      
    # 事前セットデータとの差分　diff_distanceは閾値，測定できていないデータは省く
    diff_dist = abs(dist_alldata-set_dist)
    checklist=np.zeros(len(set_dist))
    checklist_count0 = np.zeros(len(set_dist))
    checklist[np.where((diff_dist>diff_distance) & ((np.array(dist_alldata)>0) | (np.array(set_dist)>0)))]=np.array(dist_alldata)[np.where((diff_dist>diff_distance) & ((np.array(dist_alldata)>0) | (np.array(set_dist)>0)))]

    #print(np.where(diff_dist>diff_distance))
    checklist_diff = np.vstack([angle_alldata, checklist])  
    filter_checklist_diff = checklist_diff[:,filter_angle_low:filter_angle_high]
    plot1.set_data(filter_checklist_diff[:,np.where(filter_checklist_diff[1,:]>0)])
    
    checklist_count0[np.where((diff_dist>diff_distance) & ((np.array(dist_alldata)>0) | (np.array(set_dist)>0)))]=1   
    checklist_count1 = np.zeros(len(set_dist))
    for check_i in range(1, len(checklist_count0)):
        if checklist_count0[check_i]==1:
            checklist_count1[check_i]=checklist_count0[check_i]+checklist_count1[check_i-1]
    # diff_number以上で連続して，diff_disntance以上
    if max(checklist_count1)>=diff_number:
        ## 複数検出された場合は最大のインデックス
        #check_index0 = np.where(checklist_count1==max(checklist_count1))[0]-int(max(checklist_count1)/2)
        #check_index = check_index0[-1]
        
        # diff_number以上で連続して，diff_disntance以上でかつ、連続した数の集合の最大のもの
        check_index0 = checksequence(np.where(checklist_count0>0)[0])
        # 連続した数の集合の真ん中のインデックス
        check_index = check_index0[0]+int(len(check_index0)/2)
        
        if (check_index > filter_angle_low) & (check_index < filter_angle_high):
            track_index = check_index
            track_angle = angle_alldata[check_index]
            track_dist = dist_alldata[check_index]
            plot2.set_data([track_angle, track_dist])
            #print(timestamp/1000,angle_alldata[check_index[0]],dist_alldata[check_index[0]])
            csvWriter2.writerow([timestamp/1000,track_angle,track_dist])
        else:
            plot2.set_data([[],[]])
            track_dist = 0
    else:
        plot2.set_data([[],[]])
        track_dist = 0
      
    text.set_text('t: %s' % str(timestamp/1000))
    plt.draw()

    csvWriter0.writerow(angle_alldata)
    csvWriter1.writerow(dist_alldata)    

    dist_alldata_track = np.zeros(len(angle_alldata))
    if track_dist>0:
        dist_alldata_track[track_index] = track_dist
    csvWriter3.writerow(dist_alldata_track)  
    csvWriter4.writerow(np.array([timestamp/1000]))  
    
    #print(len(dist_alldata))

    plt.pause(0.001)

    if keyboard.is_pressed('Esc'): 
        print("You pressed ESC")
        laser.close()
        f0.close()
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        plt.close()
        break
laser.close()
f0.close()
f1.close()
f2.close() 
f3.close()
f4.close()
plt.close()
