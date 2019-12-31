import gc
import gym
import sys
import cv2
import ray
import numpy as np
from gym.envs.box2d.car_racing import *


ray.init()

@ray.remote
def rollout():
    env = CarRacing()
    observation = env.reset()

    done = False
    obs_act = []
    gc.collect()
    while not done:
        action = env.action_space.sample()
        obs_act.append((observation.astype(np.uint8), action))
        observation, _, done, _ = env.step(action)
    env.close()
    return obs_act


batch = sys.argv[1]
remote_ids = []
for i in range(4):
    remote_ids.append(rollout.remote())

print("batch-{} ray_ids collected..".format(batch))
result = ray.get(remote_ids)
filename = "../../../../../media/piyush/New Volume/dataset/batch-{}".format(batch)
np.save(filename, np.array(result))
print(filename + "saved..")
del result
del remote_ids
gc.collect()
print()

