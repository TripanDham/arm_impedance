import pandas as pd
import matplotlib.pyplot as plt
import time  # Optional: for slowing down and observing
from myosuite.utils import gym
import numpy as np

env = gym.make('myoChallengeRelocateP1-v0')
env.reset()
model = env.unwrapped.sim.model
data = env.unwrapped.sim.data
env.unwrapped.sim.forward()

joint_names = [
    # 'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 
    'elv_angle',
    'elbow_flexion',
    # 'pro_sup', 
    # 'flexion', 'deviation', 
    # 'shoulder1_r2',
    'shoulder_elv',
    'shoulder_rot', 
    # 'sternoclavicular_r2', 'sternoclavicular_r3',
    # 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
]
original_qpos = data.qpos.copy()

# Load the DataFrame (assuming it's already loaded as `df`)
df_all = pd.read_pickle('data.pkl')  # if not already loaded

# # Columns to plot
# columns_to_plot = ['a_elv_angle', 'a_shoulder_elv', 'a_shoulder_rot', 'a_elbow_flexion']

# # Optional: check if time column exists
# if 'time' in df.columns:
#     x = df['time']
# else:
#     x = df.index  # fallback to index if no time column

# # Plotting
# plt.figure(figsize=(12, 6))
# for col in columns_to_plot:
#     plt.plot(x, df[col], label=col)

# plt.title("Joint Angles Over Time")
# plt.xlabel("Time" if 'time' in df.columns else "Index")
# plt.ylabel("Angle (degrees)")  # or appropriate unit
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


multiple_region_flg = True
plt_flg = False

df_all['region'] = (df_all['time'] == 0).cumsum()
# Filter data for region 1
df_region_1 = df_all[df_all['region'] == 401]
# Plot
# plt.figure(figsize=(10, 5))
# plt.plot(df_region_1['time'], df_region_1['m_elbow_flexion'], label='Speed-Torque', alpha=0.8)
# plt.xlabel("speed (m/s)")
# plt.ylabel("Torque (Nm)")
# plt.legend()
# plt.grid()
# plt.show()
if multiple_region_flg == True:
    # Define region range
    i = 1
    j = df_all['region'].max()
    # j = 200
    # Get all regions from i to j
    df_selected = df_all[df_all['region'].between(i, j)]
    # Initialize empty list to hold adjusted dataframes
    df_list = []
    # Start cumulative time from 0
    time_offset = 0
    # Loop through each region
    for region_id in range(i, j + 1):
        df_region = df_selected[df_selected['region'] == region_id].copy()
        # Shift time by cumulative offset
        df_region['time'] = df_region['time'] + time_offset
        # Update time offset by DURATION, not final time value
        duration = df_region['time'].iloc[-1] - df_region['time'].iloc[0]
        time_offset += duration + (df_region['time'].diff().dropna().min())  # small delta to avoid overlap
        # Store adjusted region
        df_list.append(df_region)
    # Concatenate all adjusted regions
    df_cumulative = pd.concat(df_list, ignore_index=True)
    df_region_1 = df_cumulative
    if plt_flg == True:
        plt.figure(figsize=(10, 5))
        plt.plot(df_list[0]['time'], np.deg2rad(df_list[0]['vel_shoulder_rot']), label = 'shoulder_rot', alpha=0.8)
        plt.plot(df_list[0]['time'], np.deg2rad(df_list[0]['vel_shoulder_elv']), label = 'shoulder_elv', alpha=0.8)
        plt.plot(df_list[0]['time'], np.deg2rad(df_list[0]['vel_elv_angle']), label = 'elv_angle', alpha=0.8)
        plt.plot(df_list[0]['time'], np.deg2rad(df_list[0]['vel_elbow_flexion']), label = 'elbow_flexion', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (rad/s)")
        plt.legend()
        plt.grid()
        plt.show()

df_1 = df_list[0]
for index, row in df_1.iterrows():
    for a in joint_names:
        data.qpos[model.name2id(a, 'joint')] = np.deg2rad(row[f'a_{a}'])
        data.qvel[model.name2id(a, 'joint')] = np.deg2rad(row[f'vel_{a}'])
        data.qacc[model.name2id(a, 'joint')] = np.deg2rad(row[f'acc_{a}'])
    
    env.sim.forward()
    env.mj_render()
