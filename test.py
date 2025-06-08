from myosuite.utils import gym

env = gym.make('myoChallengeRelocateP1-v0')
env.reset()
id = env.unwrapped.sim.model.actuator('DELT1').id
print(env.unwrapped.sim.model.actuator_lengthrange[id])
print("Data: ")
for x in dir(env.unwrapped.sim.data):
    print(x)
print()
print("Model: ")
for x in dir(env.unwrapped.sim.model):
    print(x)
print()
print(env.unwrapped.sim.model.jnt)

attributes = [
    "name",
    "name2id",
    "name_actuatoradr",
    "name_bodyadr",
    "name_camadr",
    "name_eqadr",
    "name_excludeadr",
    "name_flexadr",
    "name_geomadr",
    "name_hfieldadr",
    "name_jntadr",
    "name_keyadr",
    "name_lightadr",
    "name_matadr",
    "name_meshadr",
    "name_numericadr",
    "name_pairadr",
    "name_pluginadr",
    "name_sensoradr",
    "name_siteadr",
    "name_skinadr",
    "name_tendonadr",
    "name_texadr",
    "name_textadr",
    "name_tupleadr",
    "names", 
    "names_map", 
    "njnt",
    "nq"
]

model = env.unwrapped.sim.model
data = env.unwrapped.sim.data

for attr in attributes:
    if hasattr(model, attr):
        value = getattr(model, attr)
        print(f"{attr}:\n{value}\n")
    else:
        print(f"{attr} not found in model.\n")
joints = model.jnt()
print(joints)

for _ in range(10000):
    env.mj_render()
    new_qpos = env.unwrapped.sim.data.qpos.copy()
    new_qvel = env.unwrapped.sim.data.qvel.copy()
    new_qpos += 0.0001
    new_qvel[:] = 0.0
    env.unwrapped.sim.data.qpos[:] = new_qpos
    env.unwrapped.sim.data.qvel[:] = new_qvel
    env.sim.forward()
    # env.step(env.action_space.sample()) # take a random action
    print("Actuator Force: ", env.unwrapped.sim.data.actuator_force)
    print("Actuator Length", env.unwrapped.sim.data.actuator_length)
    # print("Tendon Length: ", env.unwrapped.sim.data.ten_length)
    print("Joint Stiffnesses: ", model.jnt_stiffness)
env.close()
