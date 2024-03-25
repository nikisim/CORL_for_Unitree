import gymnasium as gym 
import numpy as np
import loco_mujoco
from loco_mujoco import LocoEnv


mdp = LocoEnv.make("UnitreeA1.simple.perfect")

dataset = mdp.create_dataset()
print(dataset['actions'])
