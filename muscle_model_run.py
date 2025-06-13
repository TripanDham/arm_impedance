import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import socket
import struct

udp_ip = '127.0.0.1'
udp_recv = 12348
udp_send = 12349
BUFFER_SIZE = 8192

recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_socket.bind((udp_ip, udp_recv))

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class MuscleModel:
    def __init__(self, scale: float, wrist_on: bool = False, model_path: str = "model.xml", target_body: str = "body_to_scale", emg_muscles = []):
        self.scale = scale
        self.wrist_on = wrist_on
        self.model_path = model_path
        self.target_body = target_body

        if self.wrist_on:
            self.muscle_names = []
            self.joint_names = []  
        else:
            self.muscle_names = [
                "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
                "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB", "TRIlong",
                "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD"
            ]

            self.joint_names = [
                # 'acromioclavicular_r1', 'acromioclavicular_r2', 'acromioclavicular_r3', 
                'shoulder_elv',
                'elv_angle',
                'shoulder_rot', 
                'elbow_flexion',
                # 'pro_sup', 
                # 'flexion', 'deviation', 
                # 'shoulder1_r2',
                # 'sternoclavicular_r2', 'sternoclavicular_r3',
                # 'unrothum_r1', 'unrothum_r2', 'unrothum_r3', 'unrotscap_r2', 'unrotscap_r3'
            ]

        # Set EMG muscles (subset or separate set from muscle_names)
        self.emg_muscles = emg_muscles  # Fill this list with muscle names of interest for EMG

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        # mujoco.mj_forward(self.model, self.data)
        # self._scale_body_mass() 

        self.muscle_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.muscle_names]
        self.emg_muscle_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.emg_muscles]
        
        self.act = np.zeros_like(self.data.act)
        
        self.act[self.muscle_ids] = 0.1

        self.opt_muscle_ids = []
        for i in self.muscle_ids:
            if i not in self.emg_muscle_ids:
                self.opt_muscle_ids.append(i)

        self.elv_plane_id = self.model.joint('shoulder_elv').dofadr
        self.elv_angle_id = self.model.joint('elv_angle').dofadr
        self.rot_id = self.model.joint('shoulder_rot').dofadr
        self.elbow_id = self.model.joint('elbow_flexion').dofadr

    # def _scale_body_mass(self):
    #     try:
    #         body_id = self.model.body(self.target_body)
    #         original_mass = self.model.body_mass[body_id]
    #         self.model.body_mass[body_id] = original_mass * self.scale
    #         print(f"Scaled mass of '{self.target_body}' from {original_mass} to {self.model.body_mass[body_id]}")
    #     except Exception as e:
    #         print(f"Error scaling body mass: {e}")

    def optimize(self, req_torques: list, emg_act = []):
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.size = mujoco.mj_stateSize(self.model, self.spec)
        self.state0 = np.empty(self.size, np.float64)
        mujoco.mj_getState(self.model, self.data, self.state0, self.spec)

        #assuming act contains all the correct activations from previous instant, enforce emg based activations
        for i in range(len(emg_act)):
            self.act[self.emg_muscle_ids[i]] = emg_act[i]
        
        initial_guess = [self.act[i] for i in self.opt_muscle_ids] #set initial guess to activations from previous step

        self.init_state = {}
        self.init_state['qpos'] = self.data.qpos
        self.init_state['qvel'] = self.data.qvel
        self.init_state['act'] = self.act
        
        def cost_fn(activations):
            return np.sum(activations**2)

        def torque_constraint(activations):
            
            mujoco.mj_setState(self.model, self.data, self.state0, self.spec)
            # mujoco.mj_step(self.model, self.data)

            # for i in range(len(self.joint_names)):
            #     self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])] = qpos[i]
            #     self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])] = qvel[i]
            
            ctrl = self.data.ctrl
            
            for idx, act in zip(self.opt_muscle_ids, activations):
                ctrl[idx] = act

            self.data.ctrl = ctrl
            for _ in range(2):
                self.data.act = ctrl
                mujoco.mj_step(self.model, self.data)

            elv_plane_torq = self.data.qfrc_actuator[self.elv_plane_id] + self.data.qfrc_passive[self.elv_plane_id] + self.data.qfrc_applied[self.elv_plane_id]
            elv_angle_torq = self.data.qfrc_actuator[self.elv_angle_id] + self.data.qfrc_passive[self.elv_angle_id] + self.data.qfrc_applied[self.elv_angle_id]
            shoulder_rot_torq = self.data.qfrc_actuator[self.rot_id] + self.data.qfrc_passive[self.rot_id] + self.data.qfrc_applied[self.rot_id]
            elbow_torq = self.data.qfrc_actuator[self.elbow_id] + self.data.qfrc_passive[self.elbow_id] + self.data.qfrc_applied[self.elbow_id]

            torq = [elv_plane_torq[0], elv_angle_torq[0], shoulder_rot_torq[0], elbow_torq[0]]
            # print(activations)
            # print(torq)
            return req_torques - torq

        constraints = {
            'type': 'eq',
            'fun': torque_constraint,
        }

        bounds = [(0, 1)] * len(self.opt_muscle_ids)

        start_t = time.time_ns()
        res = scipy.optimize.minimize(
            fun=cost_fn,
            x0=initial_guess,
            method='SLSQP',
            constraints=[constraints],
            bounds=bounds,
            options={'disp': False, 'maxiter': 10}
        )
        end_t = time.time_ns()
        print(f"Optimization took {(end_t - start_t) / 1e6} ms")

        if not res.success:
            print("Optimization failed:", res.message)

        return res.x, res.fun, res.success, end_t - start_t
    
    def stiffness(self):
        K_muscle = []
        alpha = 23.4
        for id in self.muscle_ids:
            f = self.data.actuator_force[id]
            l = self.data.actuator_length[id]
            km = alpha * f/l
            K_muscle.append(km)

        jac = self.data.ten_J
        jac_1 = []
        for i in range(self.model.na):
            if i in self.muscle_ids:
                jac_1.append([])
                for j in range(self.model.nq):
                    if j in [self.elv_plane_id, self.elv_angle_id, self.rot_id, self.elbow_id]:
                        jac_1[i].append(jac[i, j])
            
        jac_1 = np.array(jac_1)
        k_joints = jac_1.T @ np.diag(K_muscle) @ jac_1
        return k_joints

model_path = "myo_sim/arm/myoarm.xml" 
model = MuscleModel(
    scale=1.0,
    wrist_on=False,
    model_path=model_path,
    target_body="humerus" # just to change weight later on
)

emg_ids = {'bicep': 0, 'tricep': 1, 'wrist_flex': 2, 'wrist_ext': 3, 'delc': 4, 'pec': 5, 'dels': 6}

############################################################### MAIN LOOP ####################################################################
while True:
    data, addr = recv_socket.recvfrom(BUFFER_SIZE)
    if len(data) == 12 * 8:
        values = struct.unpack('20d', data)
        #unpack order: 4 joint angles, 4 joint velocities, 4 torques, 8 EMG values
        torq = []
        qpos = values[0:4]
        qvel = values[4:8]
        for i in range(len(model.joint_names)):
            model.data.qpos[model.model.joint(model.joint_names[i]).qposadr] = values[i]
            model.data.qvel[model.model.joint(model.joint_names[i]).qposadr] = values[4+i]
            torq.append(values[8+i])
        emg_vals = values[12:]
    else:
        print("Invalid packet size")
    
    torq = np.array(torq)
    mujoco.mj_forward(model.model, model.data)
    activations, err, success, t = model.optimize(req_torques=torq)

    mujoco.mj_setState(model.model, model.data, model.state0, model.spec)
    ctrl = model.data.ctrl
    
    for idx, act in zip(model.opt_muscle_ids, activations):
        ctrl[idx] = act

    for name in ["BIClong", "BICshort", "BRA", "BRD"]:
        a = ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]
        ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = 0.1*a + 0.9*emg_vals[emg_ids['bicep']]
    
    for name in ["TRIlong","TRIlat", "TRImed"]:
        a = ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]
        ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = 0.1*a + 0.9*emg_vals[emg_ids['tricep']]
    
    for _ in range(3):
        model.data.act = ctrl
        mujoco.mj_step(model.model, model.data)

    elbow_pos = model.data.qpos[model.elbow_id]
    k = np.abs(np.diag(model.stiffness()))

    send_data = struct.pack('2d', elbow_pos, k)
    send_socket.sendto(send_data, (udp_ip, udp_send))

