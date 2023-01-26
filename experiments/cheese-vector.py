# %%
# Imports
import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
import plotly as py
import plotly.graph_objects as go

# NOTE: this is Monte's RL hooking code (and other stuff will be added in the future)
# Install normally with: pip install circrl
import circrl.module_hook as cmh

# This is a hack, we should make our shared repo into a python package we can install locally with
# pip install -e
import sys
sys.path.append('../../procgen-tools')
import models

import gatherdata

# %%
# Load two levels
import pickle as pkl
from procgen import ProcgenGym3Env
import envs.maze as maze
import lovely_tensors as lt 
lt.monkey_patch()

venv = ProcgenGym3Env(
    num=2, env_name='maze', num_levels=1, start_level=0,
    distribution_mode='hard', num_threads=1, render_mode="rgb_array",
)
venv = maze.wrap_venv(venv)

with open('../mazes/2.pkl', 'rb') as f:
    state_bytes = pkl.load(f) 
venv.env.callmethod('set_state', state_bytes)

# %%
# Load model
modelpath = '../../models/maze_I/model_rand_region_5.pth'
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

action_size = 15 # lol
env = gatherdata.create_venv()

# Load model
policy = load_policy(modelpath, action_size, device=device)

# Hook the network and run this observation through a custom predict-like function
hook = cmh.ModuleHook(policy)

# Custom probe function to evaluate the policy network
def forward_func_policy(network, inp):
    hidden = network.embedder(inp)
    return network.fc_policy(hidden)

# Get initial observation, and show maze rendering
obs = env.reset().astype(np.float32)  # Not sure why the env is returning a float64 object?
render = env.render(mode='rgb_array')
# px.imshow(render, title='Rendering').show()

# Do an initial run of this observation through the network
hook.probe_with_input(obs, func=forward_func_policy)

# Show the labels of all the intermediate activations
print(hook.values_by_label.keys())

# Visualize a random intermediate activation, and the logits
label = 'embedder.fc_out'
value = hook.get_value_by_label(label)
action_logits = hook.get_value_by_label('fc_policy_out').squeeze()
# px.imshow(value[0,...], title=label).show()

# Demonstrate ablating some values to zero, show impact on action logits
# (Just ablate the first channel of the above activation as a test)
mask = np.zeros_like(value, dtype=bool)
mask[0,...] = True

activ_diff = value - value_patched

patches = {label: cmh.PatchDef(
    t.from_numpy(mask),
    t.from_numpy(np.array([0.], dtype=np.float32)))}
# Run the patched probe
hook.probe_with_input(obs,  func=forward_func_policy, patches=patches)
value_patched = hook.get_value_by_label(label)
action_logits_patched = hook.get_value_by_label('fc_policy_out').squeeze()
# Plot results
action_meanings = env.env.combos
fig = go.Figure()
fig.add_trace(go.Scatter(y=action_logits, name='original'))
fig.add_trace(go.Scatter(y=action_logits_patched, name='patched'))
fig.update_layout(title="Action logits")
fig.update_xaxes(tickvals=np.arange(len(action_logits)), ticktext=action_meanings)
fig.show()

# %%