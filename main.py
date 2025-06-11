from myosuite.utils import gym
import mujoco
import numpy as np
import scipy.optimize
import time
import matplotlib.pyplot as plt

env = gym.make('myoChallengeRelocateP1-v0')
env.reset()
model = env.unwrapped.sim.model
data = env.unwrapped.sim.data
original_qpos = data.qpos.copy()
original_qvel = data.qvel.copy()

init_state = env.unwrapped.sim.get_state()

#DOF order: shoulder_elv_plane, shoulder_elv_angle, shoulder_rot, elbow_flex
torq = np.array([5, -5, -2, 2])
# torq = 10
# force_proj = [10,10,10,10,10,10] #this is not the end effector force, it is the projection of the joint forces on the end site

muscle_names = [
    "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
    "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB", "TRIlong",
    "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"
]

muscle_indices = [model.name2id(name, 'actuator') for name in muscle_names]

initial_guess = np.ones(23) * 0.1

def cost_fn(activations):
    return np.sum(activations**2)

def torque_constraint(activations, req_torq):
    ctrl = np.zeros(model.na)

    for idx, act in zip(muscle_indices, activations):
        ctrl[idx] = act

    data.time = init_state['time']
    data.qpos[:] = init_state['qpos']
    data.qvel[:] = init_state['qvel']
    
    data.act[:] = ctrl
    # env.step(ctrl)
    env.unwrapped.sim.forward()
    # elv_plane_torq = data.qfrc_actuator[model.name2id('acromioclavicular_r2', 'joint')] + data.qfrc_actuator[model.name2id('elv_angle', 'joint')]
    # elv_angle_torq = data.qfrc_actuator[model.name2id('acromioclavicular_r3', 'joint')] + data.qfrc_actuator[model.name2id('shoulder_elv', 'joint')]
    # shoulder_rot_torq = data.qfrc_actuator[model.name2id('shoulder1_r2', 'joint')] + data.qfrc_actuator[model.name2id('shoulder_rot', 'joint')]
    elv_plane_id = model.name2id('shoulder_elv', 'joint')
    elv_plane_torq = data.qfrc_actuator[elv_plane_id] + data.qfrc_passive[elv_plane_id] + data.qfrc_applied[elv_plane_id]

    elv_angle_id = model.name2id('elv_angle', 'joint')
    elv_angle_torq = data.qfrc_actuator[elv_angle_id] + data.qfrc_passive[elv_angle_id] + data.qfrc_applied[elv_angle_id]

    rot_id = model.name2id('shoulder_rot', 'joint')
    shoulder_rot_torq = data.qfrc_actuator[rot_id] + data.qfrc_passive[rot_id] + data.qfrc_applied[rot_id]
    
    elbow_id = model.name2id('elbow_flexion', 'joint')
    elbow_torq = data.qfrc_actuator[elbow_id] + data.qfrc_passive[elbow_id] + data.qfrc_applied[elbow_id]

    torq = [elv_plane_torq, elv_angle_torq, shoulder_rot_torq, elbow_torq]
    error = np.array(torq) - req_torq
    return error

constraints = {
    'type': 'eq',
    'fun': lambda x: torque_constraint(x, torq),
}

bounds = [(0, 1)] * 23

start_t = time.time_ns()

res = scipy.optimize.minimize(
    fun=cost_fn,
    x0=initial_guess,
    method='SLSQP',
    constraints=[constraints],
    bounds=bounds,
    options={'disp': True, 'maxiter': 10}
)
end_t=time.time_ns()

print("Optimal activations:", res.x)
print("Final cost:", res.fun)
print("Time taken: ", (end_t - start_t)/1e9)

activations = res.x
ctrl = np.zeros(model.na)
ctrl[:23] = np.clip(activations, 0, 1)

env.reset()
env.step(ctrl)

elv_plane_id = model.name2id('shoulder_elv', 'joint')
elv_plane_torq = data.qfrc_actuator[elv_plane_id] + data.qfrc_passive[elv_plane_id] + data.qfrc_applied[elv_plane_id]

elv_angle_id = model.name2id('elv_angle', 'joint')
elv_angle_torq = data.qfrc_actuator[elv_angle_id] + data.qfrc_passive[elv_angle_id] + data.qfrc_applied[elv_angle_id]

rot_id = model.name2id('shoulder_rot', 'joint')
shoulder_rot_torq = data.qfrc_actuator[rot_id] + data.qfrc_passive[rot_id] + data.qfrc_applied[rot_id]

elbow_id = model.name2id('elbow_flexion', 'joint')
elbow_torq = data.qfrc_actuator[elbow_id] + data.qfrc_passive[elbow_id] + data.qfrc_applied[elbow_id]

torq = [elv_plane_torq, elv_angle_torq, shoulder_rot_torq, elbow_torq]

print("Final torques:", torq)

env.mj_render()
time.sleep(0.1)

ctrl_values = []
act_values = []
id = model.name2id('DELT3', 'actuator')

for _ in range(300):
    env.step(ctrl)
    env.mj_render()

    # Access and store one element from ctrl and act
    ctrl_values.append(data.ctrl[id])
    act_values.append(data.act[id])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot control input
ax1.plot(ctrl_values, color='tab:blue')
ax1.set_ylabel('Control Input')
ax1.set_title('Control vs Time')
ax1.grid(True)

# Plot activation
ax2.plot(act_values, color='tab:orange')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Muscle Activation')
ax2.set_title('Activation vs Time')
ax2.grid(True)

plt.tight_layout()
plt.show()