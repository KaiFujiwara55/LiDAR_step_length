import pandas as pd

df = pd.read_csv("/Users/kai/大学/4年春/小川研/LIDAR/20240703/lidar_scan_data_0703/csv/1person_walking.csv")

df = df[["Timestamp", "Ori_x", "Ori_y", "Ori_z"]]



