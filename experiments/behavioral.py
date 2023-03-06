# %%
%reload_ext autoreload
%autoreload 2

# %%
try:
    import procgen_tools
except ImportError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup(download_data=False) # create directory structure and download data 

# %%
from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze, vfield

# %% Generate vfields for randomly generated seeds
AX_SIZE = 4

# Control show_components with a checkbox
checkbox = widgets.Checkbox(value=False, description='Show action components')

rows = 1
cols = 2

fig_out = widgets.Output()
text_out = widgets.Output()
display(text_out)
display(fig_out)

def generate_plots(max_size : int = 18, min_size : int = 3, cols : int = 2, rows : int = 1):
    """ Generate rows*cols plots for random seeds with inner grid size at most max_size and at least min_size. """
    assert 3 <= min_size <= max_size <= maze.WORLD_DIM, 'Invalid min/max size'

    # Indicate that the plots are being generated    
    with text_out:
        print(f'Generating {rows*cols} plots...')

    fig, axs = plt.subplots(rows, cols, figsize=(AX_SIZE*cols, AX_SIZE*rows))
    for idx, ax in enumerate(axs.flatten()): 
        seed = np.random.randint(0, 100000)
        while maze.get_inner_grid_from_seed(seed=seed).shape[0] > max_size or maze.get_inner_grid_from_seed(seed=seed).shape[0] < min_size:
            seed = np.random.randint(0, 100000)
        venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
        vf = vfield.vector_field(venv, policy=hook.network)
        vfield.plot_vf(vf, ax=ax, show_components=checkbox.value, render_padding = False)
        ax.set_title(f'Seed: {seed:,}')
        ax.axis('off')  

    # Indicate that the plots are done being generated
    text_out.clear_output()
    fig_out.clear_output(wait=True)
    with fig_out:     
        plt.show()


# Make slider for max inner grid size
slider = widgets.IntSlider(min=3, max=25, step=1, value=18, description='Max grid size')
display(slider)

# Make a button to generate new plots
button = widgets.Button(description='Generate new plots')
button.on_click(lambda _: generate_plots(max_size = slider.value))
display(HBox([button, checkbox]))

generate_plots(max_size = slider.value)
# %% Make a totally empty venv and then visualize it
for fill_type in (maze.EMPTY, maze.CHEESE):
    venv = maze.get_filled_venv(fill_type=fill_type)
    img = visualization.visualize_venv(venv, mode='human', idx=0, show_plot=True, render_padding=True, render_mouse=True)

# %%

