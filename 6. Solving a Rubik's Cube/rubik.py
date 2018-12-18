# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook



plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from rl.agents.q_agent import QAgent




class RubiksCube(object):
    def __init__(self):

        print(f"Initialized RubiksCube")



    def render(self):
        pass
