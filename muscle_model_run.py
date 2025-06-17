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
    def __init__(self, humerus_mass: float, forearm_mass: float, wrist_on: bool = False, model_path: str = "model.xml", emg_muscles = []):
        self.humerus_mass = humerus_mass
        self.forearm_mass = forearm_mass
        self.wrist_on = wrist_on
        self.model_path = model_path

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
        self._scale_body_mass() 

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

        self.prev_jac = []

    def _scale_body_mass(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'humerus')
        original_mass = self.model.body_mass[body_id]
        self.model.body_mass[body_id] = self.humerus_mass
        inertia = self.model.body_inertia[body_id]
        self.model.body_inertia[body_id] = self.humerus_mass/original_mass * inertia

        ulna_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ulna')
        ulna_mass = self.model.body_mass[ulna_id]
        ulna_inertia = self.model.body_inertia[ulna_id]
        radius_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'radius')
        radius_mass = self.model.body_mass[radius_id]
        radius_inertia = self.model.body_inertia[radius_id]
        self.model.body_mass[ulna_id] = ulna_mass/(ulna_mass+radius_mass) * self.forearm_mass
        self.model.body_inertia[ulna_id] = self.model.body_mass[ulna_id]/ulna_mass * ulna_inertia
        self.model.body_mass[radius_id] = radius_mass/(ulna_mass+radius_mass) * self.forearm_mass
        self.model.body_inertia[radius_id] = self.model.body_mass[radius_id]/radius_mass * radius_inertia

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

        return res.x, res.fun, res.success, (end_t - start_t)/1e6
    
    def stiffness(self):
        K_muscle = []
        alpha = 23.4
        muscle_forces = []

        for id in self.muscle_ids:
            f = self.data.actuator_force[id]
            muscle_forces.append(f)
            l = self.data.actuator_length[id]
            km = alpha * f/l
            K_muscle.append(km)
        
        muscle_forces = np.array(muscle_forces)

        jac = self.data.ten_J
        jac_1 = []
        joint_id_list = [self.elv_plane_id, self.elv_angle_id, self.rot_id, self.elbow_id]
        jac_dot = []
        for i in range(self.model.na):
            if i in self.muscle_ids:
                jac_1.append([])
                jac_dot.append([])
                for j in range(self.model.nq):
                    if j in joint_id_list:
                        jac_1[i].append(jac[i, j])
                        if self.prev_jac:
                            jac_dot.append((jac[i,j] - self.prev_jac[i,(len(jac_1[i]) - 1)])/(self.data.qpos[j] - self.prev_qpos[len(jac_1[i])-1]))

        jac_1 = np.array(jac_1)
        jac_dot = np.array(jac_dot)
        self.prev_jac = jac_1
        self.prev_qpos = [self.data.qpos[id] for id in joint_id_list]
        k_joints = jac_1.T @ np.diag(K_muscle) @ jac_1 + jac_dot.T * muscle_forces
        return k_joints

model_path = "myo_sim/arm/myoarm.xml" 
model = MuscleModel(
    wrist_on=False,
    model_path=model_path,
    humerus_mass=1.8,
    forearm_mass=1.1
)

emg_ids = {'bicep': 0, 'tricep': 1, 'wrist_flex': 2, 'wrist_ext': 3, 'delc': 4, 'pec': 5, 'dels': 6}

fail_count = 0
time_count = 0
error = 0

############################################################### MAIN LOOP ####################################################################
while True:
    data, addr = recv_socket.recvfrom(BUFFER_SIZE)
    if len(data) == 26 * 8:
        values = struct.unpack('26d', data)
        torq = []
        shoulder_emg = values[0:4]
        arm_emg = values[4:8] #order: black, red, green, blue
        q_elbow = values[8] + np.pi/2
        q_shoulder = values[20:23]
        q_shoulder = [0,0,0]
        vel_elbow = values[12]
        vel_shoulder = values[23:26]
        vel_shoulder = [0,0,0]
        qpos = list(q_shoulder) + [q_elbow]
        qvel = list(vel_shoulder) + [vel_elbow]
        for i in range(len(model.joint_names)):
            model.data.qpos[model.model.joint(model.joint_names[i]).qposadr] = qpos[i]
            model.data.qvel[model.model.joint(model.joint_names[i]).qposadr] = qvel[i]
            # torq.append(values[8+i])
        emg_vals = values[12:]
    else:
        print("Invalid packet size")
    torq = [0,0,0,2]
    torq = np.array(torq)
    mujoco.mj_forward(model.model, model.data) #joint positions set above, so run fwd to set sim state
    
    activations, err, success, t = model.optimize(req_torques=torq)

    if not(success):
        fail_count += 1
    if fail_count > 0:
        if success:
            fail_count -= 1
    if t > 40:
        time_count += 1
    if time_count > 0:
        if t < 40:
            time_count -= 1
    
    if fail_count > 100 or time_count > 100:
        error = 1
    else:
        error = 0

    mujoco.mj_setState(model.model, model.data, model.state0, model.spec)
    ctrl = model.data.ctrl
    
    for idx, act in zip(model.opt_muscle_ids, activations):
        ctrl[idx] = act

    blend = 0.1
    bicep_emg = (arm_emg[0] - 200)/4000
    tricep_emg = (arm_emg[1] - 200)/4000
    if bicep_emg < 0:
        bicep_emg = 0
    if tricep_emg > 1:
        tricep_emg = 1
    print("Bicep activation: ", bicep_emg)
    print("Tricep activation: ", tricep_emg)
    for name in ["BIClong", "BICshort", "BRA", "BRD"]:
        a = ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]
        ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = blend*a + (1-blend)*bicep_emg
    
    for name in ["TRIlong","TRIlat", "TRImed"]:
        a = ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]
        ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = blend*a + (1-blend)*tricep_emg
    
    for _ in range(3):
        model.data.ctrl = ctrl
        mujoco.mj_step(model.model, model.data)
    elbow_pos = model.data.qpos[model.elbow_id]
    print(f"Elbow position: {elbow_pos}")
    print(f"Elbow velocity: {model.data.qvel[model.elbow_id]}")
    print(f"Elbow torque: {model.data.qfrc_actuator[model.elbow_id] + model.data.qfrc_passive[model.elbow_id] + model.data.qfrc_applied[model.elbow_id]}")
    
    k = np.abs(model.stiffness())

    elbow_pos = float(elbow_pos)
    k = float(k[3,3])
    error = float(error)

    send_data = struct.pack('3d', elbow_pos, k, error)
    send_socket.sendto(send_data, (udp_ip, udp_send))

