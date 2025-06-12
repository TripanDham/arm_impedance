import mujoco
import time
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("myo_sim/arm/myoarm.xml")
data = mujoco.MjData(model)

joint_names = [
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

q = [ 0.17281, 0.168055, -0.109975, 1.11181]

for i in range(len(joint_names)):
    id = model.joint(joint_names[i]).qposadr
    data.qpos[id] = q[i]
# Set multiple joint positions
# target_positions = {
#     "shoulder_elv": 0.17281,
#     "elv_angle": 0.168055,
#     "elbow_flexion": 1.11181,
#     "shoulder_rot": -0.1,
# }

# for name, val in target_positions.items():
#     idx = model.joint(name).qposadr
#     data.qpos[idx] = val


mujoco.mj_forward(model, data)
spec = mujoco.mjtState.mjSTATE_INTEGRATION
size = mujoco.mj_stateSize(model, spec)
state0 = np.empty(size, np.float64)
mujoco.mj_getState(model, data, state0, spec)


joint_dof_count = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

for j in range(model.njnt):
    joint_type = model.jnt_type[j]
    dof_count = joint_dof_count[joint_type]
    dof_start = model.jnt_dofadr[j]
    name = model.names[model.name_jntadr[j]]

    for i in range(dof_count):
        dof_id = dof_start + i
        print(f"DOF {dof_id} from joint '{name}' (type {joint_type}, DOF {i+1} of {dof_count})")
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     for _ in range(200):
#         data.act = np.ones(model.na)
#         mujoco.mj_step(model, data)
#         # print([model.data.qpos[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_JOINT, i)] for i in model.joint_names])
#         viewer.sync()
#         time.sleep(0.01)
    
#     mujoco.mj_setState(model, data, state0, spec)
#     while viewer.is_running():
#         data.act[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = 1
#         mujoco.mj_step(model, data)
#         # print([model.data.qpos[mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_JOINT, i)] for i in model.joint_names])
#         viewer.sync()
#         time.sleep(0.01)

