import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import socket
import struct
import threading

udp_ip = '127.0.0.1'
udp_recv = 12348
udp_send = 12349
BUFFER_SIZE = 8192

# recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# recv_socket.bind((udp_ip, udp_recv))
# recv_socket.setblocking(0)  # Set to non-blocking mode

send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    # def _scale_body_mass(self):
    #     try:
    #         body_id = self.model.body(self.target_body)
    #         original_mass = self.model.body_mass[body_id]
    #         self.model.body_mass[body_id] = original_mass * self.scale
    #         print(f"Scaled mass of '{self.target_body}' from {original_mass} to {self.model.body_mass[body_id]}")
    #     except Exception as e:
    #         print(f"Error scaling body mass: {e}")

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
            
            elv_angle_torq = self.data.qfrc_actuator[self.elv_angle_id] + self.data.qfrc_passive[self.elv_angle_id]
            shoulder_elv_torq = self.data.qfrc_actuator[self.shoulder_elv_id] + self.data.qfrc_passive[self.shoulder_elv_id]
            shoulder_rot_torq = self.data.qfrc_actuator[self.rot_id] + self.data.qfrc_passive[self.rot_id]
            elbow_torq = self.data.qfrc_actuator[self.elbow_id] + self.data.qfrc_passive[self.elbow_id]

            torq = [elv_angle_torq[0], shoulder_elv_torq[0], shoulder_rot_torq[0], elbow_torq[0]]
            # print(activations)
            # print(torq)
            return req_torques - torq

        constraints = {
            'type': 'eq',
            'fun': torque_constraint,
            'tol': 1e-3
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
        # print(f"Optimization took {(end_t - start_t) / 1e6} ms")

        if not res.success:
            print("Optimization failed:", res.message)

        elv_angle_torq = self.data.qfrc_actuator[self.elv_angle_id] + self.data.qfrc_passive[self.elv_angle_id] + self.data.qfrc_applied[self.elv_angle_id]
        shoulder_elv_torq = self.data.qfrc_actuator[self.shoulder_elv_id] + self.data.qfrc_passive[self.shoulder_elv_id] + self.data.qfrc_applied[self.shoulder_elv_id]
        shoulder_rot_torq = self.data.qfrc_actuator[self.rot_id] + self.data.qfrc_passive[self.rot_id] + self.data.qfrc_applied[self.rot_id]
        elbow_torq = self.data.qfrc_actuator[self.elbow_id] + self.data.qfrc_passive[self.elbow_id] + self.data.qfrc_applied[self.elbow_id]

        torq = [elv_angle_torq[0], shoulder_elv_torq[0], shoulder_rot_torq[0], elbow_torq[0]]

        return res.x, res.fun, res.success, end_t - start_t, torq
    
    def stiffness(self):
        K_muscle = []
        alpha = 23.4
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

        jac_1 = np.array(jac_1)
        dJdT = []
        # print(jac_1)

        for id in self.joint_id_list:
            jac_t1 = []
            dJ = []
            mujoco.mj_setState(model.model, model.data, state1, spec)
            self.data.qpos[id] += 0.001
            mujoco.mj_forward(self.model, self.data)
            jac = self.data.ten_J

            for i in range(self.model.na):
                if i in self.muscle_ids:
                    jac_t1.append([])
                    for j in range(self.model.nq):
                        if j in self.joint_id_list:
                            jac_t1[-1].append(jac[i+4, j])
            
            jac_t1 = np.array(jac_t1)
        
            dJ = (jac_t1.T - jac_1.T)/0.001

            dJdT.append(dJ)

        dJdT = np.array(dJdT)
        k_diffterm = dJdT @ muscle_forces
        # k_diffterm_col = dJ @ muscle_forces
        # k_diffterm = [k_diffterm_col for _ in range(len(self.joint_id_list))]
        # k_diffterm = np.array(k_diffterm)
        # k_diffterm = k_diffterm.T

        k_joints = jac_1.T @ np.diag(K_muscle) @ jac_1
        k_joints = k_joints #+ k_diffterm
        return k_joints # , jac_1, muscle_forces, K_muscle
    
    def linearize(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.model, spec)
        pre_calc_state = np.empty(size, np.float64)
        mujoco.mj_getState(self.model, self.data, pre_calc_state, spec)

        n_steps = 2
        act1 = 0.5
        act2 = 0.2

        for m_id in self.muscle_ids:
            self.data.ctrl[m_id] = act1

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        a1 = np.array([self.data.act[m_id] for m_id in self.muscle_ids])
        f1 = np.array([self.data.actuator_force[m_id] for m_id in self.muscle_ids])

        mujoco.mj_setState(self.model, self.data, pre_calc_state, spec)

        for m_id in self.muscle_ids:
            self.data.ctrl[m_id] = act2

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        a2 = np.array([self.data.act[m_id] for m_id in self.muscle_ids])
        f2 = np.array([self.data.actuator_force[m_id] for m_id in self.muscle_ids])

        m = (f2 - f1)/(a2 - a1)
        M = np.diag(m)
        c = f1 - M @ a1
        return M,c

    def optimize_fast(self, req_torques):
        # provide req_torques in the order: elv_angle, shoulder_elv, rot, elbow 
        # elv_angle is the angle of the elevation plane, and shoulder_elv is the angle in the elevation plane    
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.model, spec)
        state0 = np.empty(size, np.float64)
        mujoco.mj_getState(self.model, self.data, state0, spec)

        mujoco.mj_step1(model.model, model.data)
        # mujoco.mj_step(model.model, model.data)

        passive_torques = np.array([self.data.qfrc_passive[id][0] for id in self.joint_id_list])
        # print(f"Passive torques: {passive_torques}")

        jac = self.data.ten_J
        jac_1 = []

        for i in range(self.model.na):
            if i in self.muscle_ids:
                jac_1.append([])
                for j in range(self.model.nq):
                    if j in self.joint_id_list:
                        jac_1[-1].append(jac[i + 4, j])       

        jac_1 = np.array(jac_1)

        M, c = self.linearize()
        
        A = jac_1.T @ M
        C = jac_1.T @ c

        initial_guess = [self.act[i] for i in self.muscle_ids] #set initial guess to activations from previous step 

        mujoco.mj_setState(self.model, self.data, state0, spec)

        def cost_fn(activations):
            act_sum = 0
            for a in activations:
                act_sum += a**2
            return act_sum

        # def constraint_fun(activations):
        #     return A @ activations + C - req_torques
        
        def constraint_jac(activations):
            return A
        
        def objective_grad(activations):
            return 2 * activations

        constraints = [{
            'type': 'eq',
            'fun': lambda activations: A @ activations + C - req_torques,
            'jac': lambda activations: A
            # 'tol': 1e-3
        }]

        # constraints = [
        #     {'type': 'eq', 'fun': lambda activations: np.dot(A[i,:], activations) - req_torques[i] + C[i], 'jac': lambda activations: A[i,:]} for i in range(len(req_torques))
        # ]

        bounds = [(0, 1)] * len(self.muscle_ids)

        start_t = time.time_ns()
        res = scipy.optimize.minimize(
            fun=cost_fn,
            x0=initial_guess,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'disp': False, 'maxiter': 100},
            jac= lambda activations: 2 * activations
        )
        end_t = time.time_ns()
        # print(f"Optimization took {(end_t - start_t) / 1e6} ms")

        if not res.success:
            print("Optimization failed:", res.message)
        
        # for i, con in enumerate(constraints):
        #     residual = con['fun'](res.x)
        #     print(f"Constraint {i} ({con['type']}): residual = {residual + req_torques}")

        activations = res.x
        
        mujoco.mj_setState(self.model, self.data, state0, spec)

        ctrl = self.data.ctrl
            
        for idx, act in zip(self.muscle_ids, activations):
            ctrl[idx] = act

        self.data.ctrl = ctrl

        for _ in range(5):
            mujoco.mj_forward(self.model, self.data)

        elv_angle_torq = self.data.qfrc_actuator[self.elv_angle_id] #+ self.data.qfrc_passive[self.elv_angle_id] + self.data.qfrc_applied[self.elv_angle_id]
        shoulder_elv_torq = self.data.qfrc_actuator[self.shoulder_elv_id] #+ self.data.qfrc_passive[self.shoulder_elv_id] + self.data.qfrc_applied[self.shoulder_elv_id]
        shoulder_rot_torq = self.data.qfrc_actuator[self.rot_id] #+ self.data.qfrc_passive[self.rot_id] + self.data.qfrc_applied[self.rot_id]
        elbow_torq = self.data.qfrc_actuator[self.elbow_id] #+ self.data.qfrc_passive[self.elbow_id] + self.data.qfrc_applied[self.elbow_id]

        torq = [elv_angle_torq[0], shoulder_elv_torq[0], shoulder_rot_torq[0], elbow_torq[0]]

        return res.x, res.fun, res.success, end_t - start_t, torq

model_path = "myo_sim/arm/myoarm.xml" 
model = MuscleModel(
    scale=1.0,
    wrist_on=False,
    model_path=model_path,
)


emg_ids = {'bicep': 0, 'tricep': 1, 'wrist_flex': 2, 'wrist_ext': 3, 'delc': 4, 'pec': 5, 'dels': 6}

fail_count = 0
time_count = 0
error = 0

class UDPReceiver:
    def __init__(self, ip='localhost', port=12348, buffer_size=1024):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)  # Non-blocking socket
        
        self.buffer_size = buffer_size
        self.latest_data = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                with self.lock:
                    self.latest_data = data
            except BlockingIOError:
                # No data available right now
                pass

    def get_latest_data(self):
        with self.lock:
            return self.latest_data


# Initialize the UDP receiver
udp_receiver = UDPReceiver(ip=udp_ip, port=udp_recv, buffer_size=BUFFER_SIZE)
udp_receiver.start()

spec = mujoco.mjtState.mjSTATE_INTEGRATION
size = mujoco.mj_stateSize(model.model, spec)
state0 = np.empty(size, np.float64)
mujoco.mj_getState(model.model, model.data, state0, spec)
count = 0
print_iter = 0

bodies = [mujoco.mj_id2name(model.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.model.nbody)]
forearm_mass = 1.1942
hand_mass = 0.0182 + 0.7332

ulna_id = mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_BODY, "ulna")
radius_id = mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_BODY, "radius")
hand_id_start = radius_id + 1
model_fa_mass = model.model.body_mass[ulna_id] + model.model.body_mass[radius_id]
model_hand_mass = sum(model.model.body_mass[i] for i in range(hand_id_start, model.model.nbody))
fa_scale = forearm_mass / model_fa_mass
hand_scale = hand_mass / model_hand_mass

model.model.body_mass[ulna_id] *= fa_scale
model.model.body_inertia[ulna_id] *= fa_scale
model.model.body_mass[radius_id] *= fa_scale
model.model.body_inertia[radius_id] *= fa_scale
for i in range(hand_id_start, model.model.nbody):
    model.model.body_mass[i] *= hand_scale
    model.model.body_inertia[i] *= hand_scale

############################################################### MAIN LOOP ####################################################################
while True:
    count += 1
    if count == 1:
        print_iter = 1
        count = 0
    else:
        print_iter = 0
    mujoco.mj_setState(model.model, model.data, state0, spec)
    # data, addr = recv_socket.recvfrom(BUFFER_SIZE)
    data = udp_receiver.get_latest_data()
    if data is not None and len(data) == 22 * 8:
        values = struct.unpack('22d', data)
        torq = []
        shoulder_emg = values[0:4]
        arm_emg = values[4:8] #order: black, red, green, blue
        q_elbow = values[8] + np.pi/2
        q_shoulder = values[20:23]
        q_shoulder = [0.1,1.57,0.1]
        vel_elbow = values[12]
        vel_shoulder = values[23:26]
        vel_shoulder = [0,0,0]
        qpos = list(q_shoulder) + [q_elbow]
        qvel = list(vel_shoulder) + [vel_elbow]
        torq_el = values[16]
        for i in range(len(model.joint_names)):
            model.data.qpos[model.model.joint(model.joint_names[i]).qposadr] = qpos[i]
            model.data.qvel[model.model.joint(model.joint_names[i]).qposadr] = qvel[i]
            # torq.append(values[8+i])
    else:
        print("Received packet of invalid size")
        continue

    torq = [0.001,0.001,0.001,torq_el]
    torq = np.array(torq)
    mujoco.mj_forward(model.model, model.data) #joint positions set above, so run fwd to set sim state

    # spec = mujoco.mjtState.mjSTATE_INTEGRATION
    # size = mujoco.mj_stateSize(model.model, spec)
    state1 = np.empty(size, np.float64)
    mujoco.mj_getState(model.model, model.data, state1, spec)
    
    start = time.time_ns()
    activations, err, success, t, torq_out = model.optimize(req_torques=torq)
    end_t = time.time_ns()
    
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

    mujoco.mj_setState(model.model, model.data, state1, spec)
    ctrl = model.data.ctrl
    
    for idx, act in zip(model.muscle_ids, activations):
        ctrl[idx] = act

    blend = 1.0
    bicep_emg = (arm_emg[0] - 200)/4000
    tricep_emg = (arm_emg[1] - 200)/4000

    if bicep_emg < 0:
        bicep_emg = 0
    if tricep_emg < 0:
        tricep_emg = 0
    if tricep_emg > 1:
        tricep_emg = 1
    if bicep_emg > 1:
        bicep_emg = 1

    if print_iter:
        print(f"Received torques: {torq}")
        print(f"Full Optimization took {(end_t - start) / 1e6} ms")
        print(f"Torques: {torq_out}")

    # print("Bicep activation: ", bicep_emg)
    # print("Tricep activation: ", tricep_emg)
    for name in ["BIClong", "BICshort", "BRA", "BRD"]:
        a = ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]
        if print_iter:
            print(f"{name} activation: {a}")
        ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = blend*a + (1-blend)*bicep_emg
    
    for name in ["TRIlong","TRIlat", "TRImed"]:
        a = ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]
        if print_iter:
            print(f"{name} activation: {a}")
        ctrl[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = blend*a + (1-blend)*tricep_emg
    
    model.data.ctrl = ctrl
    for _ in range(2):
        mujoco.mj_step(model.model, model.data)

    elbow_pos = model.data.qpos[model.elbow_id]
    elbow_vel = model.data.qvel[model.elbow_id]
    if print_iter:
        print(f"Elbow position: {elbow_pos}")
        print(f"Elbow velocity: {model.data.qvel[model.elbow_id]}")
        print(f"Elbow torque: {model.data.qfrc_actuator[model.elbow_id]} + {model.data.qfrc_passive[model.elbow_id]} = {model.data.qfrc_actuator[model.elbow_id] + model.data.qfrc_passive[model.elbow_id]}")
        
    k = np.abs(model.stiffness())

    elbow_pos = float(elbow_pos)
    k = float(k[3,3])
    error = float(error)
    # print(f"Elbow stiffness: {k}")

    send_data = struct.pack('3d', elbow_vel, k, error)
    send_socket.sendto(send_data, (udp_ip, udp_send))

