import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================================================
# 2-D Kalman Filter vs. Dead-Reckoning（含真实漂移）
# --------------------------------------------------------------
# 目标：
#   1. 模拟一辆匀速行驶的车辆（真值 vx = 1.0, vy = 0.5）。
#   2. Dead-Reckoning（里程计积分）假设速度存在 5% 偏差，
#      且每步叠加过程噪声 → 产生逐步漂移。
#   3. Kalman Filter 使用带噪 GPS 测量持续纠偏 → 估计更接近真值。
# ==============================================================

# ---------- 1. 全局参数与随机种子 ----------
np.random.seed(0)        # 固定随机种子，便于课堂复现
num_steps = 80           # 模拟 80 个时间步
dt = 1.0                 # 采样周期 1 s

# ---------- 2. 真实速度 & DR 偏差速度 ----------
true_vx, true_vy = 1.0, 0.5        # 真实速度 (m/s)
dr_vx,   dr_vy   = 0.95, 0.525     # DR 使用的速度，5 % 偏差

# ---------- 3. 噪声方差 ----------
process_var = 2e-2  # → 改成 1e-4（模型很好） or 5e-2（模型很差）
meas_var = 6.0  # → 改成 1.0（GPS 很准） or 20.0（GPS 很差）

# ---------- 4. 状态空间模型 ----------
# 状态向量 x = [x, y, vx, vy]ᵀ
A = np.array([[1, 0, dt, 0],   # 位置由上一步位置 + 速度×dt 得到
              [0, 1, 0, dt],
              [0, 0, 1,  0],   # 速度恒定
              [0, 0, 0,  1]])
H = np.array([[1, 0, 0, 0],    # 观测只包含位置 (x, y)
              [0, 1, 0, 0]])

Q = np.eye(4) * process_var     # 过程噪声协方差矩阵
R = np.eye(2) * meas_var        # 观测噪声协方差矩阵

# ---------- 5. 初始状态 ----------
x_true = np.array([[0.], [0.], [true_vx], [true_vy]])   # 真实状态
x_est  = np.array([[0.], [0.], [0.8], [0.6]])           # KF 初值（存在偏差）
P_est  = np.eye(4) * 400                                # KF 协方差（大不确定）
x_dr   = np.array([[0.], [0.], [dr_vx], [dr_vy]])       # DR 初值（带偏差）

# ---------- 6. 轨迹容器（用于可视化） ----------
true_xy, meas_xy, kf_xy, dr_xy = [], [], [], []

# ---------- 7. 主循环 ----------
for _ in range(num_steps):

    # -------- 7.1 真值推进（理想匀速） --------
    # x_true = A · x_true  ⇒ 位置累加真实速度
    x_true = A @ x_true
    true_xy.append(x_true[:2, 0])   # 仅保存 (x, y)

    # -------- 7.2 生成带噪 GPS 测量 --------
    # z = H·x_true + v，v ~ N(0, R)
    z = H @ x_true + np.random.multivariate_normal([0, 0], R).reshape(2, 1)
    meas_xy.append(z[:, 0])

    # -------- 7.3 Kalman Filter --------
    # ① 预测（Prediction）
    x_pred = A @ x_est
    P_pred = A @ P_est @ A.T + Q
    # ② 更新（Update）
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)  # 卡尔曼增益
    x_est = x_pred + K @ (z - H @ x_pred)                   # 校正状态
    P_est = (np.eye(4) - K @ H) @ P_pred                    # 校正协方差
    kf_xy.append(x_est[:2, 0])

    # -------- 7.4 Dead-Reckoning --------
    # 仅用模型积分 + 小过程噪声，完全不利用 GPS
    x_dr = A @ x_dr + np.random.multivariate_normal([0, 0, 0, 0], Q).reshape(4, 1)
    dr_xy.append(x_dr[:2, 0])

# ---------- 8. 转为 NumPy 数组 ----------
true_xy = np.array(true_xy)
meas_xy = np.array(meas_xy)
kf_xy   = np.array(kf_xy)
dr_xy   = np.array(dr_xy)

# ==============================================================
# 可视化：动态 & 静态
# ==============================================================

# ---------- 9. 动态动画 ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Kalman Filter vs. Dead-Reckoning (drift visible)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_xlim(-5, np.max(true_xy[:, 0]) + 10)
ax.set_ylim(-5, np.max(true_xy[:, 1]) + 10)
ax.set_aspect('equal')
ax.grid(True)

# 各轨迹绘图对象（初始化为空）
true_line, = ax.plot([], [], color='blue',  lw=2, label="True Path")
meas_scatter = ax.scatter([], [], marker='x', color='orange', label="Measurements")
kf_line, = ax.plot([], [], color='green', lw=2, label="KF Estimate")
dr_line, = ax.plot([], [], color='red',   lw=2, linestyle='--', label="Dead-Reckoning")
ax.legend(loc='upper left')

def init():
    """初始化动画帧：清空数据"""
    true_line.set_data([], [])
    kf_line.set_data([], [])
    dr_line.set_data([], [])
    meas_scatter.set_offsets(np.empty((0, 2)))
    return true_line, kf_line, dr_line, meas_scatter

def animate(i):
    """更新到第 i 帧时应显示的数据"""
    true_line.set_data(true_xy[:i+1, 0], true_xy[:i+1, 1])
    kf_line.set_data(kf_xy[:i+1, 0],     kf_xy[:i+1, 1])
    dr_line.set_data(dr_xy[:i+1, 0],     dr_xy[:i+1, 1])
    meas_scatter.set_offsets(meas_xy[:i+1])
    return true_line, kf_line, dr_line, meas_scatter

# 创建动画：interval = 250 ms 每帧
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=num_steps, interval=250, blit=True)

# 保存 GIF（放进 PPT）
ani.save("./kf_vs_drift_dr.gif", writer="pillow", fps=4)

# ---------- 10. 静态终帧 ----------
plt.figure(figsize=(8, 6))
plt.plot(true_xy[:, 0], true_xy[:, 1], color='blue',  lw=2, label="True Path")
plt.scatter(meas_xy[:, 0], meas_xy[:, 1], color='orange', marker='x',
            label="Measurements", alpha=0.5)
plt.plot(kf_xy[:, 0], kf_xy[:, 1], color='green', lw=2, label="KF Estimate")
plt.plot(dr_xy[:, 0], dr_xy[:, 1], color='red',   lw=2, linestyle='--',
         label="Dead-Reckoning")
plt.title("Static view: KF corrects the drifting Dead-Reckoning")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("./kf_vs_drift_dr_static.png", dpi=150)
plt.show()
