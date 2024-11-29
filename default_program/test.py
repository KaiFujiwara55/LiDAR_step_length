import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("/Users/kai/大学/小川研/LiDAR_step_length")
from default_program.class_method import default_method
from default_program.class_method import original_method

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

point = np.array([-14.79025354, -7997.39894095, 1172.16821289])
theta_z = 1.5647306849397995
print(np.degrees(theta_z))
rotated_point = def_method.rotate_points(point.reshape(1, 3), theta_z=theta_z)
rotated_rotated_point = def_method.rotate_points(rotated_point, theta_z=-theta_z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(point[0], point[1], color="red")
ax.scatter(rotated_point[0][0], rotated_point[0][1], color="blue")
ax.scatter(rotated_rotated_point[0][0], rotated_rotated_point[0][1], color="green")

plt.show()
plt.close()
