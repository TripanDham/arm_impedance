# This example requires the installation of myosuite
# pip install myosuite

import time

import myosuite  # noqa
from myosuite.utils import gym

import deprl

env = gym.make("myoChallengeRelocateP1-v0", reset_type="random")
policy = deprl.load_baseline(env)

env.seed(0)
model = env.unwrapped.sim.model
data = env.unwrapped.sim.data

joint_names = [
    # 'OBJRx', 'OBJRy', 'OBJRz', 'OBJTx', 'OBJTy', 'OBJTz', 
    # 'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 
    # 'cmc_abduction', 'cmc_flexion', 
    # 'deviation', 
    'elbow_flexion', 
    'elv_angle', 
    # 'flexion', 'ip_flexion', 
    # 'mcp2_abduction', 'mcp2_flexion', 'mcp3_abduction', 'mcp3_flexion', 'mcp4_abduction', 'mcp4_flexion', 'mcp5_abduction', 'mcp5_flexion', 'md2_flexion', 'md3_flexion', 'md4_flexion', 'md5_flexion', 'mp_flexion', 'pm2_flexion', 'pm3_flexion', 'pm4_flexion', 'pm5_flexion', 
    # 'pro_sup', 
    'shoulder_elv',
    'shoulder_rot',
    # 'shoulder1_r2','sternoclavicular_r2', 'sternoclavicular_r3', 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
]

muscle_names = [
    "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
    "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB", "TRIlong",
    "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"
]

for ep in range(10):
    ep_steps = 0
    ep_tot_reward = 0
    state, _ = env.reset()

    while True:
        # samples random action
        action = policy(state)
        # applies action and advances environment by one step
        state, reward, terminated, truncated, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward
        env.mj_render()
        time.sleep(0.01)

        # check if done
        done = terminated or truncated
        for name in joint_names:
            torque = data.qfrc_passive[model.name2id(name, 'joint')] + data.qfrc_actuator[model.name2id(name, 'joint')] + data.qfrc_applied[model.name2id(name, 'joint')]
            print(f"{name}: {torque}")       
        
        for name in muscle_names:
            act = data.act[model.name2id(name, 'actuator')]
            print(f"{name}: {act}")       

        if done or (ep_steps >= 1000):
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f};"
            )
            env.reset()
            break