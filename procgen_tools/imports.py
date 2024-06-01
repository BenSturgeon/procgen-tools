from typing import List, Tuple, Dict, Union, Optional, Callable
import re
from collections import defaultdict
import pickle
import funcy as fn

import numpy as np
import pandas as pd
import torch as t
import math

import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm
from einops import *
from IPython.display import *
from ipywidgets import *
from ipywidgets import interact
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140

import circrl.hooks as cmh
import circrl.rollouts as cro

import procgen_tools.models as models
import procgen_tools.utils as utils
from procgen import ProcgenGym3Env

import os, sys
from glob import glob
from pathlib import Path

from ipywidgets import (
    Text,
)  # Import this later because otherwise Text gets cast as str?

RAND_REGION = 5
NUM_ACTIONS = 15
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False
# PATH_PREFIX = '../' if in_jupyter else ''


def load_model(rand_region: int = 5, num_actions: int = 15, use_small: bool = False):
    """Load a model from the trained_models folder. Returns the policy and the hook."""
    import os
    print(os.getcwd())
    model_stub = f"../../model_9501.0.pt"

    try:
        utils.cd_into_procgen_tools()
    except Exception:
        Path("procgen-tools").mkdir(parents=True, exist_ok=True)
        os.chdir("procgen-tools")

    policy = models.load_policy(model_stub, num_actions, t.device("cpu"))
    manager = cmh.HookManager(policy)
    # custom_hook_dict = {'conv_seqs.1.res_block0.conv0.weight': custom_hook_func}

    hook = manager._make_cache_hook('conv_seqs.1.res_block0.conv0.weight')
    return policy, hook


policy, hook = load_model(
    rand_region=RAND_REGION, num_actions=NUM_ACTIONS, use_small=False
)



# # Useful general variables
default_layer = 'conv_seqs.1.res_block0.conv0.weight'
# hook(module='conv_seqs.1.res_block0.conv0.weight', input=np.zeros((1, 3, 64, 64) ,dtype=np.float32))
# labels = list(hook.values_by_label.keys())  # all labels in the model
# if "_out" in labels:
#     labels.remove("_out")
