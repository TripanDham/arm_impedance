import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt

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
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.model, spec)
        state0 = np.empty(size, np.float64)
        mujoco.mj_getState(self.model, self.data, state0, spec)

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
            
            mujoco.mj_setState(self.model, self.data, state0, spec)
            # mujoco.mj_step(self.model, self.data)

            # for i in range(len(self.joint_names)):
            #     self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])] = qpos[i]
            #     self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])] = qvel[i]
            
            ctrl = self.data.ctrl
            
            for idx, act in zip(self.opt_muscle_ids, activations):
                ctrl[idx] = act

            self.data.ctrl = ctrl
            for _ in range(1):
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
            options={'disp': False, 'maxiter': 50, 'tol': 1e-2}
        )
        end_t = time.time_ns()
        print(f"Optimization took {(end_t - start_t) / 1e6} ms")

        if not res.success:
            print("Optimization failed:", res.message)

        return res.x, res.fun, res.success, end_t - start_t

model_path = "myo_sim/arm/myoarm.xml" 
model = MuscleModel(
    scale=1.0,
    wrist_on=False,
    model_path=model_path,
    target_body="humerus" # just to change weight later on
)

df_all = pd.read_pickle('data.pkl')  

multiple_region_flg = True
plt_flg = False

df_all['region'] = (df_all['time'] == 0).cumsum()
# Filter data for region 1
df_region_1 = df_all[df_all['region'] == 401]

i = 1
j = df_all['region'].max()
# j = 200
# Get all regions from i to j
df_selected = df_all[df_all['region'].between(i, j)]
# Initialize empty list to hold adjusted dataframes
df_list = []
# Start cumulative time from 0
time_offset = 0
# Loop through each region
for region_id in range(i, j + 1):
    df_region = df_selected[df_selected['region'] == region_id].copy()
    # Shift time by cumulative offset
    df_region['time'] = df_region['time'] + time_offset
    # Update time offset by DURATION, not final time value
    duration = df_region['time'].iloc[-1] - df_region['time'].iloc[0]
    time_offset += duration + (df_region['time'].diff().dropna().min())  # small delta to avoid overlap
    # Store adjusted region
    df_list.append(df_region)

    print("Movement Number: ", region_id)

# Initialize tracking lists
commanded_torques_list = []
obtained_torques_list = []
optimization_times_ms = []
optimization_success = []

# with mujoco.viewer.launch_passive(model.model, model.data) as viewer:    
for index, row in df_list[5].iterrows():
    torq = []
    for name in model.joint_names:
        model.data.qpos[model.model.joint(name).qposadr] = np.deg2rad(row[f'a_{name}'])
        model.data.qvel[model.model.joint(name).qposadr] = np.deg2rad(row[f'vel_{name}'])
        
        torq.append(row[f'm_{name}'])

    torq = np.array(torq)
    mujoco.mj_forward(model.model, model.data)
    # mujoco.mj_step(model.model, model.data)
    activations, err, success, t = model.optimize(req_torques=torq)

    print("Torque = ", torq)
    print("Torque error = ", err)

    elv_plane_torq = model.data.qfrc_actuator[model.elv_plane_id] + model.data.qfrc_passive[model.elv_plane_id] + model.data.qfrc_applied[model.elv_plane_id]
    elv_angle_torq = model.data.qfrc_actuator[model.elv_angle_id] + model.data.qfrc_passive[model.elv_angle_id] + model.data.qfrc_applied[model.elv_angle_id]
    shoulder_rot_torq = model.data.qfrc_actuator[model.rot_id] + model.data.qfrc_passive[model.rot_id] + model.data.qfrc_applied[model.rot_id]
    elbow_torq = model.data.qfrc_actuator[model.elbow_id] + model.data.qfrc_passive[model.elbow_id] + model.data.qfrc_applied[model.elbow_id]

    obt_torq = [elv_plane_torq[0], elv_angle_torq[0], shoulder_rot_torq[0], elbow_torq[0]]
    
    commanded_torques_list.append(torq)
    obtained_torques_list.append(obt_torq)
    optimization_success.append(success)
    optimization_times_ms.append(t/1e6)

    # viewer.sync()    

# Convert lists to numpy arrays
commanded_torques_arr = np.array(commanded_torques_list)
obtained_torques_arr = np.array(obtained_torques_list)
times_ms = np.array(optimization_times_ms)
success_arr = np.array(optimization_success).astype(int)

num_joints = len(model.joint_names)
time_axis = np.arange(len(times_ms))

# Set up subplot grid: one row for each joint, plus one for time, one for success
total_rows = num_joints + 2
fig, axes = plt.subplots(total_rows, 1, figsize=(12, 3 * total_rows), sharex=True)

# Plot torque comparisons per joint
for i, joint in enumerate(model.joint_names):
    axes[i].plot(time_axis, commanded_torques_arr[:, i], label=f"Commanded {joint}", color='tab:blue')
    axes[i].plot(time_axis, obtained_torques_arr[:, i], label=f"Obtained {joint}", color='tab:orange', linestyle='--')
    axes[i].set_ylabel("Torque (Nm)")
    axes[i].legend(loc="upper right")
    axes[i].grid(True)

# Plot optimization time
axes[num_joints].plot(time_axis, times_ms, label="Optimization Time (ms)", color='tab:green')
axes[num_joints].set_ylabel("Time (ms)")
axes[num_joints].set_title("Optimization Time per Frame")
axes[num_joints].set_ylim(0,120)
axes[num_joints].legend(loc="upper right")
axes[num_joints].grid(True)

# Plot success
axes[num_joints + 1].plot(time_axis, success_arr, label="Success", color='tab:red', marker='o')
axes[num_joints + 1].set_ylabel("Success")
axes[num_joints + 1].set_xlabel("Frame Index")
axes[num_joints + 1].set_title("Optimization Success per Frame")
axes[num_joints + 1].set_yticks([0, 1])
axes[num_joints + 1].grid(True)
axes[num_joints + 1].legend(loc="upper right")

# General layout settings
fig.suptitle("Torque Tracking and Optimization Diagnostics", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


#TODO: 
# Display and check
# Test EMG
# Add a failsafe - if opt fails - for more than certain number of steps - then break out
# is it useful to store the jacobian and hessian of the cost function from the current instant to save time from the next instant

# torq = np.array([2, -2, -1, 2])
# q = [ 0.17281, 0.168055, -0.109975, 1.11181]
# for i in range(len(model.joint_names)):
#     model.data.qpos[model.model.joint(model.joint_names[i]).qposadr] = q[i]
# qpos = [0.168055, 1.11181, 0.17281, -0.109975]
# qvel = [model.data.qvel[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_JOINT, i)] for i in model.joint_names]


# with mujoco.viewer.launch_passive(model.model, model.data) as viewer:
#     while viewer.is_running():
#         # model.data.act[model.muscle_ids] = np.ones_like(model.muscle_ids)
#         mujoco.mj_forward(model.model, model.data)
#         # print([model.data.qpos[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_JOINT, i)] for i in model.joint_names])
#         viewer.sync()
#         time.sleep(0.01)
