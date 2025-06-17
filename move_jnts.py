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
    'shoulder_elv',
    'elv_angle',
    'shoulder_rot',
    'elbow_flexion',
    # 'pro_sup', 
    # 'flexion', 'deviation', 'shoulder1_r2',  
    # 'sternoclavicular_r2', 'sternoclavicular_r3',
    # 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
]
original_qpos = data.qpos.copy()

# print(model.tendon_lengthspring)

position = [1,0,1.57,0]

# q = [0.17281, 0.168055, -0.109975, 1.11181] #env reset pos
q = [1.57, 1.0, 0.2, 1.11181]

for i in range(len(joint_names)):   
    jnt_id = model.name2id(joint_names[i], 'joint')
    qpos_adr = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_adr] = q[i]
env.sim.forward()
env.mj_render()
time.sleep(3)
env.close()
# for joint_name in joint_names:
#     try:
#         jnt_id = model.name2id(joint_name, 'joint')
#         qpos_adr = model.jnt_qposadr[jnt_id]
        
#         child_body_id = model.jnt_bodyid[jnt_id]
#         parent_body_id = model.body_parentid[child_body_id]

#         child_body_name = model.id2name(child_body_id, 'body')
#         parent_body_name = model.id2name(parent_body_id, 'body')
#     except Exception as e:
#         print(f"Skipping {joint_name}: {e}")
#         continue

#     print(f"Joint '{joint_name}' connects:")
#     print(f"  Parent Body: {parent_body_name}")
#     print(f"  Child Body : {child_body_name}")

#     for _ in range(500):
#         env.mj_render()
#         new_qpos = data.qpos.copy()
#         new_qvel = data.qvel.copy()

#         # act = np.zeros(len(data.act))
#         # act[21] = 0.5
#         new_qpos[qpos_adr] += 0.005 
#         new_qvel[:] = 0.0

#         data.qpos[:] = new_qpos
#         data.qvel[:] = new_qvel
#         # env.step(act)
#         env.sim.forward()
#         x = data.ten_J
#         # print(x)

#     env.mj_render()
#     new_qpos = data.qpos.copy()
#     new_qvel = data.qvel.copy()

#     new_qpos = original_qpos
#     new_qvel[:] = 0.0

#     data.qpos[:] = new_qpos
#     data.qvel[:] = new_qvel
#     env.sim.forward()
# print("\nAll joints moved.")
