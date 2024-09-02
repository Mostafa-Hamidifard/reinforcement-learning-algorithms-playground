# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:50:49 2024

@author: Mostafa
"""

# %% importing important libs
import os
from time import time
import numpy as np

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# %% checking torch version
print(torch.__version__)
# !nvdia-smi
