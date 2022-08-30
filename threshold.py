from ctypes import sizeof
from math import pi
from statistics import median
from mpl_toolkits.mplot3d import Axes3D
from time import time
from numpy.linalg import norm
from numpy import average, diff, shape, size
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import os
import csv
from numpy.linalg import norm
import xml.etree.ElementTree as ET
from os.path import exists

# IMU sensitivity
acc_sens = 0.122 / 1000.0 * 9.81  # sens in mg to m/s^2
gyro_sens = 17.5 / 1000.0 * pi / 180.0  # sens in mdps to rad/s


path = ['New track with impacts data/one movement', 'New track with impacts data/Continous movement',
        'New track with impacts data/boxing', 'New track with impacts data/Golf',
        'New track with impacts data/sword cutting', 'New track with impacts data/one impact',
        'New track with impacts data/Multiple impacts', 'New track with impacts data/impacts more',
        'New track with impacts data/chair mapping', 'New track with impacts data/mapping']

# Extract IMU and Visual Data from sensors
xml_tree = ET.parse(path[1] + '/' + 'rpd_out.xml')
xml_root = xml_tree.getroot()
xml_frames = xml_root.find('Frames')

t = []

pos_x = []
pos_y = []
pos_z = []

ta = []

ax = []
ay = []
az = []

gx = []
gy = []
gz = []

# OBTAINING THE IMU AND PHASESPACE DATA FROM THE XML FILE
for frame in xml_frames:
    markers = frame.find('Markers')
    if int(markers.get('num_markers')) > 0:
        for marker in markers:
            if marker.get('id') == '1':
                t.append(int(frame.get('time')) * 10**(-3))
                pos_x.append(float(marker.find('x').text) * 10**(-3))
                pos_y.append(float(marker.find('y').text) * 10**(-3))
                pos_z.append(float(marker.find('z').text) * 10**(-3))
    if frame.find('IMU'):
        ta.append(int(frame.get('time')) * 10**(-3))
        # print(int(frame.get('time')))
        ax.append(float(frame.find('IMU').find('ax').text) * acc_sens)
        ay.append(float(frame.find('IMU').find('ay').text) * acc_sens)
        az.append(float(frame.find('IMU').find('az').text) * acc_sens)
        gx.append(float(frame.find('IMU').find('gx').text) * gyro_sens)
        gy.append(float(frame.find('IMU').find('gy').text) * gyro_sens)
        gz.append(float(frame.find('IMU').find('gz').text) * gyro_sens)


acceleration = [ax, ay, az]
acc_array = np.array(acceleration).T
x, y = shape(acc_array)
norm_acc = []
for i in range(x):
    norm_acc.append(abs(np.linalg.norm(acc_array[i, :]))-9.81)

time_array = np.array(ta)

time_taken = ta[-1] - ta[0]

# Initialise window length
interval = 0.01
num_loop = int(time_taken/interval)

acc_cov = np.zeros((num_loop, 1))

# Scan through the data with the window length and compute the maximum variance
start_point = 0
end_point = np.argmax(time_array >= (time_array[start_point] + interval))
interval = abs(time_array[start_point] - time_array[end_point])
for i in range(num_loop):
    acc_cov[i] = np.var(norm_acc[start_point:end_point])

    start_point = int(np.where(time_array == (time_array[end_point]))[0])
    end_point = np.argmax(time_array >= (time_array[start_point] + interval))

acc_threshold = np.max(acc_cov)  # Computing the threshold
# print('The threshold is', acc_cov[2497])

acc_threshold = 0
for covariance in acc_cov:
    if covariance > acc_threshold:
        acc_threshold = covariance

# acc_threshold = 25 (for refernce)
# plt.plot(acc_cov)
# plt.show()
