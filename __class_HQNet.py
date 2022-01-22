import torch as t
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from __class_CNet import CNet
from __class_PQC import PQC

"""
HQNet: Hybrid quantum network. This is the full scheme for learning the symmetries
of a given quantum state. It combines a parametrized quantum circuit (PQC)
and a classical deep network (CNet), which learns to estimate the loss within 
the HQNet to regularize against finding known symmetries.
"""

