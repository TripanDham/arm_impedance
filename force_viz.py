from myosuite.utils import gym
import numpy as np
import matplotlib.pyplot as plt

# Create a Myosuite environment
env = gym.make("myoChallengeRelocateP1-v0")  # Replace with other envs if desired

# Reset the environment
env.reset()

# Parameters
num_steps = 500
# activation_value = 0.4  # Set all muscle activations to this constant

# Record torques
torque_history = []

model = env.unwrapped.sim.model
data = env.unwrapped.sim.data
env.unwrapped.sim.forward()

init_state = env.unwrapped.sim.get_state()
# Run simulation
for _ in range(num_steps):
    # action = np.ones(env.action_space.shape) * activation_value
    # data.act[:] = init_state['act']
    data.act[model.name2id('BIClong', 'actuator')] = 1
    
    env.unwrapped.sim.forward()
    env.mj_render()
    # Get joint torques from qfrc_actuator
    torques = (data.qfrc_passive.copy() + data.qfrc_actuator.copy() + data.qfrc_applied.copy())[6:]
    torque_history.append(torques)

# Convert to array
torque_history = np.array(torque_history)

joint_names = [
    # 'OBJRx', 'OBJRy', 'OBJRz', 'OBJTx', 'OBJTy', 'OBJTz', 
    'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 
    # 'cmc_abduction', 'cmc_flexion', 
    # 'deviation', 
    # 'elbow_flexion', 
    'elv_angle', 
    # 'flexion', 'ip_flexion', 
    # 'mcp2_abduction', 'mcp2_flexion', 'mcp3_abduction', 'mcp3_flexion', 'mcp4_abduction', 'mcp4_flexion', 'mcp5_abduction', 'mcp5_flexion', 'md2_flexion', 'md3_flexion', 'md4_flexion', 'md5_flexion', 'mp_flexion', 'pm2_flexion', 'pm3_flexion', 'pm4_flexion', 'pm5_flexion', 
    # 'pro_sup', 
    'shoulder1_r2', 'shoulder_elv', 'shoulder_rot', 'sternoclavicular_r2', 'sternoclavicular_r3', 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
]

num_joints = len(joint_names)

# Ensure plotting only the torques for joints (might be longer if Myosuite adds padding)
torque_history = torque_history[:, :num_joints]

print("Final joint torques (qfrc_actuator):")
for i in range(num_joints):
    final_torque = torque_history[-1, i]
    print(f"{joint_names[i]}: {final_torque:.4f}")
    
# plt.figure(figsize=(12, 6))
# for i in range(num_joints):
#     plt.plot(torque_history[:, i], label=joint_names[i])
#     # Add label at the end of the curve
#     plt.text(
#         x=num_steps - 1,
#         y=torque_history[-1, i],
#         s=joint_names[i],
#         fontsize=9,
#         verticalalignment='center',
#         horizontalalignment='left'
#     )

# plt.xlabel("Simulation Step")
# plt.ylabel("Joint Torque (qfrc_actuator)")
# plt.title("Joint Torques Over Time")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
