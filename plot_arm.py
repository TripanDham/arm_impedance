import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import csv

class MuscleModel:
    def __init__(self, scale: float, wrist_on: bool = False, model_path: str = "model.xml", emg_muscles = []):
        self.scale = scale
        self.wrist_on = wrist_on
        self.model_path = model_path

        if self.wrist_on:
            self.muscle_names = []
            self.joint_names = []  
        else:
            # self.muscle_names = [
            #     "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
            #     "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB", "TRIlong",
            #     "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"
            # ]

            # LAT1 not working for some reason

            self.muscle_names = [
                "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
                "PECM1", "PECM2", "PECM3", "LAT2", "LAT3", "CORB", "TRIlong",
                "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"
            ]

            self.joint_names = [
                # 'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 
                'elv_angle',
                'shoulder_elv',
                'shoulder_rot', 
                'elbow_flexion',
                # 'pro_sup', 
                # 'flexion', 'deviation', 
                # 'shoulder1_r2',
                # 'sternoclavicular_r2', 'sternoclavicular_r3',
                # 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
            ]

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        # mujoco.mj_forward(self.model, self.data)
        # self._scale_body_mass() 

        self.muscle_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.muscle_names]
        
        self.act = np.zeros_like(self.data.act)
        
        self.act[self.muscle_ids] = 0.1

        self.elv_angle_id = self.model.joint('elv_angle').dofadr
        self.shoulder_elv_id = self.model.joint('shoulder_elv').dofadr
        self.rot_id = self.model.joint('shoulder_rot').dofadr
        self.elbow_id = self.model.joint('elbow_flexion').dofadr

        self.joint_id_list = [self.elv_angle_id, self.shoulder_elv_id, self.rot_id, self.elbow_id]

        self.model.opt.timestep = 0.005

    def optimize(self, req_torques: list, emg_act = []):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.model, spec)
        state0 = np.empty(size, np.float64)
        mujoco.mj_getState(self.model, self.data, state0, spec)

        #assuming act contains all the correct activations from previous instant, enforce emg based activations
        for i in range(len(emg_act)):
            self.act[self.emg_muscle_ids[i]] = emg_act[i]
        
        initial_guess = [self.act[i] for i in self.muscle_ids] #set initial guess to activations from previous step
        
        def cost_fn(activations):
            return np.sum(activations**2)

        def torque_constraint(activations):
            mujoco.mj_setState(self.model, self.data, state0, spec)
            # mujoco.mj_step(self.model, self.data)

            # for i in range(len(self.joint_names)):
            #     self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])] = qpos[i]
            #     self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])] = qvel[i]
            
            ctrl = self.data.ctrl
            
            for idx, act in zip(self.muscle_ids, activations):
                ctrl[idx] = act

            self.data.ctrl = ctrl

            for _ in range(2):
                self.data.ctrl = ctrl
                mujoco.mj_step(self.model, self.data)
            
            elv_angle_torq = self.data.qfrc_actuator[self.elv_angle_id]
            shoulder_elv_torq = self.data.qfrc_actuator[self.shoulder_elv_id]
            shoulder_rot_torq = self.data.qfrc_actuator[self.rot_id]
            elbow_torq = self.data.qfrc_actuator[self.elbow_id]

            torq = [elv_angle_torq[0], shoulder_elv_torq[0], shoulder_rot_torq[0], elbow_torq[0]]
            
            # print(activations)
            # print(torq)
            return req_torques - torq

        constraints = {
            'type': 'eq',
            'fun': torque_constraint,
        }

        bounds = [(0, 1)] * len(self.muscle_ids)

        start_t = time.time_ns()
        res = scipy.optimize.minimize(
            fun=cost_fn,
            x0=initial_guess,
            method='SLSQP',
            constraints=[constraints],
            bounds=bounds,
            options={'disp': False, 'maxiter': 100}
        )
        end_t = time.time_ns()
        print(f"Optimization took {(end_t - start_t) / 1e6} ms")

        if not res.success:
            print("Optimization failed:", res.message)

        elv_angle_torq = self.data.qfrc_actuator[self.elv_angle_id] # + self.data.qfrc_passive[self.elv_angle_id] + self.data.qfrc_applied[self.elv_angle_id]
        shoulder_elv_torq = self.data.qfrc_actuator[self.shoulder_elv_id] # + self.data.qfrc_passive[self.shoulder_elv_id] + self.data.qfrc_applied[self.shoulder_elv_id]
        shoulder_rot_torq = self.data.qfrc_actuator[self.rot_id] # + self.data.qfrc_passive[self.rot_id] + self.data.qfrc_applied[self.rot_id]
        elbow_torq = self.data.qfrc_actuator[self.elbow_id] # + self.data.qfrc_passive[self.elbow_id] + self.data.qfrc_applied[self.elbow_id]

        torq = [elv_angle_torq[0], shoulder_elv_torq[0], shoulder_rot_torq[0], elbow_torq[0]]

        return res.x, res.fun, res.success, end_t - start_t, torq
    
    def stiffness(self):
        K_muscle = []
        alpha = 23.4
        # alpha = 10
        muscle_forces = []

        for id in self.muscle_ids:
            f = - self.data.actuator_force[id]
            l = self.data.actuator_length[id]
            km = alpha * f/l
            K_muscle.append(km)
            muscle_forces.append(f)
        
        muscle_forces = np.array(muscle_forces)

        jac = self.data.ten_J
        jac_1 = []
        jac_t1 = []
        dJ = []
        
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.model, spec)
        state1 = np.empty(size, np.float64)
        mujoco.mj_getState(model.model, model.data, state1, spec)

        for i in range(self.model.na):
            if i in self.muscle_ids:
                jac_1.append([])
                for j in range(self.model.nq):
                    if j in self.joint_id_list:
                        jac_1[-1].append(jac[i+4, j])

        # print(jac_1)

        for id in self.joint_id_list:
            self.data.qpos[id] += 0.001
        
        mujoco.mj_forward(self.model, self.data)
        jac = self.data.ten_J

        for i in range(self.model.na):
            if i in self.muscle_ids:
                jac_t1.append([])
                for j in range(self.model.nq):
                    if j in self.joint_id_list:
                        jac_t1[-1].append(jac[i+4, j])
        
        jac_1 = np.array(jac_1)
        jac_t1 = np.array(jac_t1)
        
        dJ = jac_t1.T - jac_1.T
        k_diffterm_col = dJ @ muscle_forces
        k_diffterm = [k_diffterm_col for _ in range(len(self.joint_id_list))]
        k_diffterm = np.array(k_diffterm)
        k_diffterm = k_diffterm.T

        print(k_diffterm)

        k_joints = jac_1.T @ np.diag(K_muscle) @ jac_1
        k_joints = k_joints + k_diffterm
        return k_joints
    
model_path = "myo_sim/arm/myoarm.xml" 
model = MuscleModel(
    scale=1.0,
    wrist_on=False,
    model_path=model_path,
)

spec = mujoco.mjtState.mjSTATE_INTEGRATION
size = mujoco.mj_stateSize(model.model, spec)
state1 = np.empty(size, np.float64)
mujoco.mj_getState(model.model, model.data, state1, spec)

def find_stiffness(q):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(model.model, spec) 
    mujoco.mj_setState(model.model, model.data, state1, spec)
    
    torque = [10,0.2,-1,2]
    torque = np.array(torque)

    for i in range(len(model.joint_names)):
        model.data.qpos[model.model.joint(model.joint_names[i]).qposadr] = q[i]
        model.data.qvel[model.model.joint(model.joint_names[i]).qposadr] = 0
        model.data.qacc[model.model.joint(model.joint_names[i]).qposadr] = 0

    mujoco.mj_forward(model.model, model.data)

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(model.model, spec)
    state0 = np.empty(size, np.float64)
    
    mujoco.mj_getState(model.model, model.data, state0, spec)

    # mujoco.mj_inverse(model.model, model.data)
    
    # torque = np.array([model.data.qfrc_inverse[id][0] for id in model.joint_id_list])
    # torque[0] = -2
    # torque[3] = 2
        
    activations, err, success, t, achieved_torque = model.optimize(req_torques=torque)
    k = model.stiffness()
    print("Stiffness: ", [k[0,0],k[1,1],k[2,2],k[3,3]])
    print("Achieved torque: ", achieved_torque)
    print("Reqd torque = ", torque)
    print("Activiations: ", activations)

    # k = np.array([[2.22612592e+01 -1.03751631e+01 -4.63603935e+00  7.32344336e-01],[-1.03751631e+01  5.75668051e+01  2.50795380e+00  2.96334339e-01],[-4.63603935e+00  2.50795380e+00  4.28175375e+00  8.00523337e-03],[ 7.32344336e-01  2.96334339e-01  8.00523337e-03  1.00387523e+01]])
    # k = np.array([[22, -10.3, -4.6, 0.73], [-10.3, 57, 2.5, 0.296], [-4.63, 2.5, 4.28, 0.008], [0.73, 0.296, 0.008, 5*10]])
    mujoco.mj_setState(model.model, model.data, state0, spec)

    ctrl = model.data.ctrl
                
    for idx, act in zip(model.muscle_ids, activations):
        ctrl[idx] = act

    model.data.ctrl = ctrl

    body_name = "lunate"  
    body_id = model.model.body(name=body_name).id

    jacp = np.zeros((3, model.model.nv))  
    jacr = np.zeros((3, model.model.nv))  

    mujoco.mj_jacBodyCom(model.model, model.data, jacp, jacr, body_id)

    J = jacp
    # J = np.vstack((jacp, jacr)) 
    J = J[:, model.joint_id_list]
    J = J[:,:,0]
    
    J_plus = J.T @ np.linalg.inv(J @ J.T)

    K_end = J_plus.T @ k @ J_plus
    K_end = K_end[:3,:3]
    return K_end, k

sh_arr = [0.5, 0.7, 1.0, 1.2, 1.4]
el_arr = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]

q_arr = []
for i in sh_arr:
    for j in el_arr:
        q_arr.append([i, 1.6, 0.1, j])

data = {'Km': [], 'Ke': [], 'el_pos': [], 'wrist_pos': [], 'error': []}

with mujoco.viewer.launch_passive(model.model, model.data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

    # for _ in range(100):
    #     mujoco.mj_step(model.model, model.data)
    for i in range(len(q_arr)):
        # try:
        k, k_m = find_stiffness(q_arr[i])
        # except:
        #     print('error')
        viewer.sync()
        
        body_name = "lunate"  
        body_id = model.model.body(name=body_name).id
        wrist_pos = model.data.xpos[body_id]
        
        elbow_site = "TRI_site_BRA_side"
        site_id = mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_SITE, elbow_site)
        elbow_pos = model.data.site_xpos[site_id]

        data['Km'].append(k_m)
        data['Ke'].append(k)
        data['el_pos'].append([elbow_pos[0], elbow_pos[1]])
        data['wrist_pos'].append([wrist_pos[0], wrist_pos[1]])

pio.renderers.default = 'browser'

fig = go.Figure()
res = 30
t = np.linspace(0, 2*np.pi, res)
x = np.cos(t)
y = np.sin(t)       
circle = np.vstack((x,y))

for i in range(len(data['Km'])):
    k_e = data['Ke'][i]
    eigvals, eigvecs = np.linalg.eigh(k_e[:2, :2])
    ellipsoid = eigvecs @ np.diag(eigvals) @ circle
    ellipsoid[0,:] = (ellipsoid[0,:])/5e4 + data['wrist_pos'][i][0]
    ellipsoid[1,:] = (ellipsoid[1,:])/5e4 + data['wrist_pos'][i][1]
    
    fig.add_trace(go.Scatter(x=ellipsoid[0,:], y=ellipsoid[1,:]))
    fig.add_trace(go.Scatter(x = [0, data['el_pos'][i][0]], y = [0, data['el_pos'][i][1]], mode='lines', line=dict(color='rgba(0,255,0,0.3)', width=0.5)))
    fig.add_trace(go.Scatter(x = [data['el_pos'][i][0], data['wrist_pos'][i][0]], y = [data['el_pos'][i][1], data['wrist_pos'][i][1]], mode='lines', line=dict(color='rgba(0,0,255,0.3)', width=0.5)))

fig.update_layout(
    width= 800,
    height = 800,
    title = "Endpoint ellipses",
    xaxis=dict(scaleanchor = 'y', range = [-0.8, 0.2]),
    yaxis=dict(range = [-0.8, 0.2]),
)

fig.show()
