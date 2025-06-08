import numpy as np
import matplotlib.pyplot as plt
from myosuite.utils import gym
from matplotlib.patches import Rectangle

# Load environment and data
env = gym.make('myoChallengeRelocateP1-v0')
env.reset()
model = env.unwrapped.sim.model
data = env.unwrapped.sim.data
data.qpos[:] = np.zeros(model.nq)
env.sim.forward()

# Full Jacobian: 63 tendons × 44 joints
J_full = data.ten_J[:63, :]

# All joint and tendon names
joint_names = [
    'OBJRx', 'OBJRy', 'OBJRz', 'OBJTx', 'OBJTy', 'OBJTz', 
    'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 
    'cmc_abduction', 'cmc_flexion', 
    'deviation', 
    'elbow_flexion', 
    'elv_angle', 'flexion', 'ip_flexion', 
    'mcp2_abduction', 'mcp2_flexion', 'mcp3_abduction', 'mcp3_flexion', 'mcp4_abduction', 'mcp4_flexion', 'mcp5_abduction', 'mcp5_flexion', 'md2_flexion', 'md3_flexion', 'md4_flexion', 'md5_flexion', 'mp_flexion', 'pm2_flexion', 'pm3_flexion', 'pm4_flexion', 'pm5_flexion', 
    'pro_sup', 
    'shoulder1_r2', 'shoulder_elv', 'shoulder_rot', 'sternoclavicular_r2', 'sternoclavicular_r3', 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
]

tendon_names = [
    'ANC', 'APL', 'BIClong', 'BICshort', 'BRA', 'BRD', 'CORB', 'DELT1', 'DELT2', 'DELT3',
    'ECRB', 'ECRL', 'ECU', 'EDC2', 'EDC3', 'EDC4', 'EDC5', 'EDM', 'EIP', 'EPB', 'EPL',
    'FCR', 'FCU', 'FDP2', 'FDP3', 'FDP4', 'FDP5', 'FDS2', 'FDS3', 'FDS4', 'FDS5', 'FPL',
    'INFSP', 'LAT1', 'LAT2', 'LAT3', 'LU_RB2', 'LU_RB3', 'LU_RB4', 'LU_RB5', 'OP',
    'PECM1', 'PECM2', 'PECM3', 'PL', 'PQ', 'PT', 'RI2', 'RI3', 'RI4', 'RI5', 'SUBSC',
    'SUP', 'SUPSP', 'TMAJ', 'TMIN', 'TRIlat', 'TRIlong', 'TRImed', 'UI_UB2', 'UI_UB3',
    'UI_UB4', 'UI_UB5'
]

selected_tendon_indices = [0,1,2,3,4,5,6,7,8,9,32,33,34,35,41,42,57,58,59]       
selected_joint_indices = [6,7,8, 12, 13, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]


J_sub = J_full[np.array(selected_tendon_indices)[:, None], selected_joint_indices].squeeze()
print(J_sub)
tendon_labels = [tendon_names[i] for i in selected_tendon_indices]
joint_labels = [joint_names[i] for i in selected_joint_indices]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
im = plt.imshow(np.abs(J_sub), cmap='viridis', aspect='auto')

# Add colorbar
plt.colorbar(im, label='|Jacobian Entry|')

# Label axes
ax.set_xticks(np.arange(len(joint_labels)))
ax.set_xticklabels(joint_labels, fontsize=6, rotation=90)
ax.set_yticks(np.arange(len(tendon_labels)))
ax.set_yticklabels(tendon_labels, fontsize=6)
ax.set_xlabel("Joint", fontsize=10)
ax.set_ylabel("Tendon", fontsize=10)

for i in range(J_sub.shape[0]):
    for j in range(J_sub.shape[1]):
        value = J_sub[i, j]
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='white', linewidth=0.5))
        ax.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=6, color='white')

# Add gridlines
# plt.grid(which='both', color='gray', linestyle='-', linewidth=0.2)
ax.set_title("Subset of Tendon-Joint Jacobian", fontsize=12)

plt.tight_layout()
plt.show()
