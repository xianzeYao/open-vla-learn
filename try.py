import numpy as np

# 动作<->token初始化工作
min_action, max_action, n_bins = -1, 1,256
bins = np.linspace(min_action, max_action, n_bins)
bin_centers = (bins[:-1] + bins[1:]) / 2.0

# 动作->token
action = np.random.uniform(low = -1.5,high=1.5,size=7) #以一个7dot的机械臂为例
action = np.clip(action, a_min=float(min_action),a_max=float(max_action))
discretized_action = np.digitize(action, bins)