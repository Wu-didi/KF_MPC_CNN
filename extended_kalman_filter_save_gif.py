"""
Extended Kalman Filter (EKF) localization sample with GIF saving

Author: Atsushi Sakai (@Atsushi_twi)
Modified: with GIF saving by ChatGPT
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from utils.plot import plot_covariance_ellipse  # 需要你有这个工具函数

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

# Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z


def jacob_f(x, u):
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return jF


def jacob_h():
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH


def ekf_estimation(xEst, PEst, z, u):
    # Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    # Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    history = []  # 用于存储动画帧

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        # 存入帧历史
        history.append((
            hxEst.copy(), hxDR.copy(), hxTrue.copy(), hz.copy(), xEst.copy(), PEst.copy()
        ))

    # 开始绘制动画并保存
    fig, ax = plt.subplots()

    def update(i):
        ax.cla()
        hxEst_i, hxDR_i, hxTrue_i, hz_i, xEst_i, PEst_i = history[i]

        ax.plot(hz_i[0, :], hz_i[1, :], ".g", label="GPS")
        ax.plot(hxTrue_i[0, :].flatten(), hxTrue_i[1, :].flatten(), "-b", label="True")
        ax.plot(hxDR_i[0, :].flatten(), hxDR_i[1, :].flatten(), "-k", label="Dead Reckoning")
        ax.plot(hxEst_i[0, :].flatten(), hxEst_i[1, :].flatten(), "-r", label="EKF Estimate")
        plot_covariance_ellipse(xEst_i[0, 0], xEst_i[1, 0], PEst_i)

        ax.axis("equal")
        ax.grid(True)
        
        ax.set_title(f"Time {i * DT:.1f} s")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100)
    plt.show()  # 先弹出交互窗口，实时播放
    ani.save("ekf_localization.gif", writer=animation.PillowWriter(fps=10))
    print("Animation saved as ekf_localization.gif 🎬")


if __name__ == '__main__':
    main()
