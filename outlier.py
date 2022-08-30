from math import pi
from statistics import median
from mpl_toolkits.mplot3d import Axes3D
from time import time
from numpy.linalg import norm
from numpy import average, diff, shape
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import os
import csv
from numpy.linalg import norm
import xml.etree.ElementTree as ET
from os.path import exists


def outlier_detection(norm_acc, ta, time_idx):
    # This function takes in the resultant accerlation (norm_acc), time array(ta) and the min length of time before collisions occur(time_idx)
    # If a spike is seen before time_idx, then it should remove that datapoint.

    idx = np.argmax(ta >= time_idx)
    outliers = []
    for i in range(idx):
        if(abs(norm_acc[i]) > 1):
            outliers.append(i)

    outliers = np.array(outliers)
    norm_acc = np.delete(norm_acc, outliers)
    ta = np.delete(ta, outliers)

    return norm_acc, ta


# The rest of the code was just to test
if __name__ == '__main__':
    acc_sens = 0.122 / 1000.0 * 9.81  # sens in mg to m/s^2
    gyro_sens = 17.5 / 1000.0 * pi / 180.0  # sens in mdps to rad/s

    path = ['New track with impacts data/one movement', 'New track with impacts data/Continous movement',
            'New track with impacts data/boxing', 'New track with impacts data/Golf',
            'New track with impacts data/sword cutting', 'New track with impacts data/one impact',
            'New track with impacts data/Multiple impacts', 'New track with impacts data/impacts more',
            'New track with impacts data/chair mapping', 'New track with impacts data/mapping']
    for j in range(len(path)):
        xml_tree = ET.parse(path[j] + '/' + 'rpd_out.xml')
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

        velocity = [gx, gy, gz]
        acceleration = [ax, ay, az]
        position = [pos_x, pos_y, pos_z]

        pos_array = np.array(position).T
        acc_array = np.array(acceleration).T
        vel_array = np.array(velocity).T
        x, y = shape(acc_array)
        norm_acc = []
        norm_vel = []
        norm_pos = []
        for i in range(x):
            norm_acc.append((np.linalg.norm(acc_array[i, :]))-9.81)
            norm_vel.append(abs(np.linalg.norm(vel_array[i, :])))

        x, y = shape(pos_array)
        for i in range(x):
            norm_pos.append(np.linalg.norm(pos_array[i, :]))

        # if path == 2:
        # time == 28

        # if path == 3:
        # time == 33

        # if path == 7:
        # time == 20

        # if path == 8:
        # time == 20

        # if path == 33:
        # time == 30

        if (j == 2 or j == 3 or j == 7 or j == 8 or j == 9):
            if(j == 2):
                norm_acc, ta = outlier_detection(norm_acc, ta, 28)
            if(j == 3):
                norm_acc, ta = outlier_detection(norm_acc, ta, 33)
            if(j == 7 or j == 8):
                norm_acc, ta = outlier_detection(norm_acc, ta, 20)
            if(j == 9):
                norm_acc, ta = outlier_detection(norm_acc, ta, 33)
        plt.plot(ta, norm_acc)
        plt.show()
