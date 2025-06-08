import time  # Optional: for slowing down and observing
from myosuite.utils import gym
import numpy as np

env = gym.make('myoChallengeRelocateP1-v0')
env.reset()
model = env.unwrapped.sim.model
data = env.unwrapped.sim.data

num_steps = 300
activation_strength = 0.5

num_muscles = model.nu

for muscle_idx in range(num_muscles):
    print(f"\nActivating muscle {muscle_idx + 1}/{num_muscles}: {model.id2name(muscle_idx, 'actuator')}")
    env.reset()

    for step in range(num_steps):
        # Set only the current muscle to active, rest to zero
        action = np.zeros(num_muscles)
        action[muscle_idx] = activation_strength

        env.step(action)
        env.mj_render()  # Render the simulation

env.close()
