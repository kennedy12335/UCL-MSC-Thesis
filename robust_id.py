from numpy import shape
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import csv
from numpy.linalg import norm
import xml.etree.ElementTree as ET
from threshold import acc_threshold, interval, acc_sens, gyro_sens
from outlier import outlier_detection

########################## Identify Impacts #####################################################################################################################################################################
if __name__ == '__main__':

    np.set_printoptions(suppress=True, formatter={
                        'float_kind': '{:0.8f}'.format})

    path = ['New track with impacts data/one movement', 'New track with impacts data/Continous movement',
            'New track with impacts data/boxing', 'New track with impacts data/Golf',
            'New track with impacts data/sword cutting', 'New track with impacts data/one impact',
            'New track with impacts data/Multiple impacts', 'New track with impacts data/impacts more',
            'New track with impacts data/chair mapping', 'New track with impacts data/mapping']

    # for j in range(len(path)):
    xml_tree = ET.parse(path[8] + '/' + 'rpd_out.xml')
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

    t = np.array(t)
    # DONE SO THAT THE TIME CAN START FROM 0 SECONDS. THE  VARIABLE IS NEEDED FOR THE PHASESPACE PLOTS/MANIPULATION
    t = t - t[0]

    # MAKING THE POSITION AND THE SUBSEQUENT INTEGRALS A NUMPY ARRAY FOR EASIER MATH MANIPULATIONS
    velocity = [gx, gy, gz]
    acceleration = [ax, ay, az]
    position = [pos_x, pos_y, pos_z]

    pos_array = np.array(position).T
    acc_array = np.array(acceleration).T
    vel_array = np.array(velocity).T

    # FINDING THE RESULTANT VELOCITY AND ACCELRATION
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

    # DONE SO THAT THE TIME CAN START FROM 0 SECONDS. THE VARIABLE IS NEEDED FOR THE IMU PLOTS/MANIPULATION
    ta = np.array(ta)
    ta = ta - ta[0]

    # OUTLIER DETECTION IF REQUIRED. THIS IS A VERY ADHOC APPROACH
    # if (j == 2 or j == 3 or j == 7 or j == 8 or j == 9):
    #     if(j == 2):
    #         norm_acc, ta = outlier_detection(norm_acc, ta, 28)
    #     if(j == 3):
    #         norm_acc, ta = outlier_detection(norm_acc, ta, 33)
    #     if(j == 7 or j == 8):
    #         norm_acc, ta = outlier_detection(norm_acc, ta, 20)
    #     if(j == 9):
    #         norm_acc, ta = outlier_detection(norm_acc, ta, 33)

    norm_acc, ta = outlier_detection(norm_acc, ta, 20)

    # FROM HERE THE ALGORITHM TO DETECT IMPACTS START

    # Initialise window length
    time_array = ta
    time_taken = ta[-1]  # TOTAL TIME TAKEN
    num_loop = int(time_taken/interval)

    # TO STORE ALL THE COVARIANCES PER WINDOW
    acc_cov = np.zeros((num_loop, 1))

    # Initialise first window
    start_point = 0
    end_point = np.argmax(time_array >= (
        time_array[start_point] + interval))

    # exactly the same value as before but just included for consistency
    interval = abs(time_array[start_point] - time_array[end_point])

    time_start_checks = []
    time_end_checks = []
    order_list = []
    impact_points = []
    for i in range(num_loop):
        acc_cov[i] = np.var(norm_acc[start_point:end_point])

        # If the variance at the current window is larger than the threshold, the index of the window is store
        if(acc_cov[i] > acc_threshold):
            impact_points.append(i)

        # This next 3 lines stores the time values of the start and end points in each window.
        order_list.append(i)
        time_start_checks.append(time_array[start_point])
        time_end_checks.append(time_array[end_point])

        # This moves the window with an interval step for the next iteration
        start_point = int(
            np.where(time_array == (time_array[end_point]))[0])
        end_point = np.argmax(time_array >= (
            time_array[start_point] + interval))

    impact_list = []
    for loop in range(len(impact_points)):
        est_point = impact_points[loop]

        first_impact = int(
            np.where(time_array == time_start_checks[est_point])[0])  # Find the value of the time epoch every time an impact occured
        end_impact = int(
            np.where(time_array == time_end_checks[est_point])[0])

        # Store that time value in a list
        impact_list.append(time_array[first_impact])


# # PLOTS ARE DONE HERE
    # plt.plot(ta, norm_acc)
    # for i in range(len(impact_list)):
    #     plt.axvline(impact_list[i],
    #                 color='r', ls='--', lw=0.5)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Resultant Acceleration (m/s^2)')
    # plt.show()

#     # POSITION PLOTS:
#     # # fig, (ax1, ax2, ax3, ) = plt.subplots(3)
#     # fig.suptitle('Subplots position')

#     # ax1.plot(t, pos_x)
#     # for i in range(len(impact_list)):
#     #     ax1.axvline(impact_list[i],
#     #                 color='r', ls='--', lw=0.5)
#     # ax1.set_ylabel('Position X(m)')

#     # ax2.plot(t, pos_y)
#     # for i in range(len(impact_list)):
#     #     ax2.axvline(impact_list[i],
#     #                 color='r', ls='--', lw=0.5)
#     # ax2.set_ylabel('Position X(m)')

#     # ax3.plot(t, pos_z)
#     # for i in range(len(impact_list)):
#     #     ax3.axvline(impact_list[i],
#     #                 color='r', ls='--', lw=0.5)
#     # ax3.set_ylabel('Position X(m)')

#     # plt.show()

#     # EXPORTING THE DATA TO USE IN CPP PROGRAMS
#     # Exporting the data to use in Cpp programs

#     pos_x = np.array(pos_x)
#     pos_y = np.array(pos_y)
#     pos_z = np.array(pos_z)

#     np.savetxt('../XPosition.csv', pos_x, delimiter=",")
#     np.savetxt('../YPosition.csv', pos_y, delimiter=",")
#     np.savetxt('../ZPosition.csv', pos_z, delimiter=",")

#     # impact_list = np.array(impact_list)
#     # np.savetxt('Impact.txt', impact_list)

#     num_impacts = len(impact_list)
#     idx = np.zeros((num_impacts, 1))
#     for i in range(len(impact_list)):
#         idx[i] = np.argmax(t >= impact_list[i])

#     num_points = np.zeros((1, 1))
#     num_points[0] = x

#     np.savetxt('../Cindices.csv', idx, delimiter=",")
#     np.savetxt('../Csize.csv', num_points, delimiter=",")

#     print(pos_x)
