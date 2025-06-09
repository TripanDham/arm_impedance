from myosuite.utils import gym
import mujoco
import numpy as np
import scipy.optimize
import time

env = gym.make('myoChallengeRelocateP1-v0')
env.reset()
model = env.unwrapped.sim.model
data = env.unwrapped.sim.data
original_qpos = data.qpos.copy()
original_qvel = data.qvel.copy()

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

    start_t = time.time_ns()
    env.reset()
    end_t = time.time_ns()
    print("reset time: ", (end_t-start_t)/1e9)
    data.qpos = original_qpos
    data.qvel = original_qvel
    
    start_t = time.time_ns()
    env.step(ctrl)
    end_t = time.time_ns()
    print("step time: ", (end_t-start_t)/1e9)
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
    return 0.1 - abs(error)

constraints = {
    'type': 'eq',
    'fun': lambda x: torque_constraint(x, torq),
    'tol': 0.1
}

bounds = [(0, 1)] * 23

start_t = time.time_ns()

res = scipy.optimize.minimize(
    fun=cost_fn,
    x0=initial_guess,
    method='SLSQP',
    constraints=[constraints],
    bounds=bounds,
    options={'disp': True, 'maxiter': 5}
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
for _ in range(200):
    env.step(ctrl)
    env.mj_render()
time.sleep(100)
