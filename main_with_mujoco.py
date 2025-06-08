import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.optimize
from roboticstoolbox import DHRobot, RevoluteDH

model_path = "myo_sim/arm/myoarm.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

# robot = DHRobot([
#     RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(-90)),
#     RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(90)),
#     RevoluteDH(a=0.0,     d=-(l0_y_Humerus),                alpha=np.deg2rad(-90)),
#     RevoluteDH(a=-l0_x_Humerus, d=(l0_z_Ulna - l0_z_Humerus), alpha=np.deg2rad(-90)),
#     RevoluteDH(a=0.0,     d=-(l0_y_Ulna + l0_y_Radius + l0_y1_Radius),     alpha=np.deg2rad(-90)),
#     # RevoluteDH(a=l0_a_Radius, d= -l0_d_Radius,                        alpha=np.deg2rad(90.0)),
#     # RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(-90.0))
#     # RevoluteDH(a=-0.08,   d=0,                              alpha=np.deg2rad(-90)) 
# ], name='HumanArm') 



# required_torques = np.array([10, 10, 10, 10]) 
force_projection = [10,10,10,-10,-10,-10] #projection of force on lunate body
torque = []

qp = data.qpos.copy()
qv = data.qvel.copy()

muscle_names = [
    "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
    "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB", "TRIlong",
    "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"
]

joint_names = ['acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 'elbow_flexion', 'elv_angle', 
               'shoulder1_r2', 'shoulder_elv', 'shoulder_rot', 'sternoclavicular_r2', 'sternoclavicular_r3', 
               'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 
               'unrotscap_r2', 'unrotscap_r3']

muscle_indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in muscle_names]
joint_indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]

body_name = "lunate"  
body_id = model.body(name=body_name).id

jacp = np.zeros((3, model.nv))  
jacr = np.zeros((3, model.nv))  

mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)

J = np.vstack((jacp, jacr)) 

def cost_fn(activations):
    return np.sum(activations**2)

def torque_constraint(activations, force_projection, qp, qv):
    data.qpos[:] = qp
    data.qvel[:] = qv
    ctrl = np.zeros(model.na)
    for idx, a in zip(muscle_indices, activations):
        ctrl[idx] = a
    data.ctrl[:] = ctrl

    mujoco.mj_step(model, data)

    torques = data.qfrc_actuator[joint_indices]
    target_torq = J.T @ force_projection
    target_torq = target_torq[joint_indices]
    error = abs(torques - target_torq)
    return 0.01 - error

initial_guess = np.ones(23) * 0.1
constraints = {'type': 'ineq', 'fun': lambda x: torque_constraint(x, force_projection, qp, qv)}
bounds = [(0, 1)] * 23

res = scipy.optimize.minimize(
    fun=cost_fn,
    x0=initial_guess,
    method='COBYLA',
    bounds=bounds,
    constraints=[constraints],
    options={'disp': True, 'maxiter': 1000}
)

activations = np.clip(res.x, 0, 1)
ctrl = np.zeros(model.na)
for idx, a in zip(muscle_indices, activations):
    ctrl[idx] = a
data.ctrl[:] = ctrl

mujoco.mj_step(model, data)

print("Optimal Activations:", activations)
# print("Final Torques:", data.qfrc_actuator[joint_indices])
print("Final Torques:", J @ data.qfrc_actuator)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)