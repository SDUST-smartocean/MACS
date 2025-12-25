import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 新增导入刻度模块

# 读取数据
data = []
with open("angle_selection.txt", "r") as f:
    for line in f:
        # 解析每一行数据
        t, ra, a, re, e, d, theta_e = line.strip().split(',')
        t = int(t.split('=')[1])
        ra = float(ra.split('=')[1])
        a = float(a.split('=')[1])
        re = float(re.split('=')[1])
        e = float(e.split('=')[1])
        data.append((t, ra, a, re, e))

# 提取数据
time_steps = [entry[0] for entry in data]
ra_values = [entry[1] for entry in data]
a_values = [entry[2] for entry in data]
re_values = [entry[3] for entry in data]
e_values = [entry[4] for entry in data]

# 创建长条形图形和双纵坐标轴
fig, ax1 = plt.subplots(figsize=(12,4))

# 绘制 Azimuth angle 的曲线（左侧纵轴）
ax1.plot(time_steps, ra_values, color='b', linestyle='-', label='Relative azimuth', linewidth=2)
ax1.plot(time_steps, a_values, color='b', linestyle='--', label='azimuth', linewidth=2)
ax1.set_xlabel('Time step')
ax1.set_ylabel('Azimuth angle', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(bottom=0)


# 创建右侧纵轴
ax2 = ax1.twinx()
ax2.plot(time_steps, re_values, color='orange', linestyle='-', label='Relative elevation', linewidth=2)
ax2.plot(time_steps, e_values, color='orange', linestyle='--', label='elevation', linewidth=2)
ax2.set_ylabel('Elevation angle', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(bottom=0)

# 设置图例
fig.legend(loc='lower right', bbox_to_anchor=(0.945, 0.145))

ax1.set_xlim(0, 200)  # 控制x轴范围

# 新增代码：设置网格间隔
ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))   # 横轴每20步
ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))   # 左侧纵轴每10度


# 新增代码：启用网格线
ax1.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=1)
#ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()

