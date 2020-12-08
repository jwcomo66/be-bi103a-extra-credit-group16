# Colab setup ------------------
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade iqplot datashader bebi103 watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../data/"
# ------------------------------
try:
    import multiprocess
except:
    import multiprocessing as multiprocess

import warnings
    
import numpy as np
import pandas as pd

import bebi103
import iqplot
import scipy
import scipy.stats as st
import holoviews as hv
import holoviews.operation.datashader
hv.extension('bokeh')

import bokeh
bokeh.io.output_notebook()
bebi103.hv.set_defaults()

import numpy.random
rg = numpy.random.default_rng()
import random
import numba
import panel as pn
from scipy.stats import gamma


# Code From hw 2.2
def ecdf_vals(data):
    ''' Takes in a Numpy array or Pandas Series and will return 
        x and y values for plotting an ecdf'''
    data = np.sort(data)
    return np.asarray([[val, (i+1)/(len(data))] for i, val in enumerate(data) 
                       if i == len(data)-1 or val != data[i+1]])

def plot_hw2():
	df =  pd.read_csv("data/gardner_time_to_catastrophe_dic_tidy.csv")
	label = df.loc[df["labeled"], "time to catastrophe (s)"]
	no_label = df.loc[~df["labeled"], "time to catastrophe (s)"]
	p = bokeh.plotting.figure(
    width=700,
    height=500,
    x_axis_label="Catastrophe Time. (s)",
    y_axis_label="Cumulative Distribution",
    title="eCDF of Catastrophe times"
	)

	# prepare list of colors

	colors = ["red", "blue"]
	legend_name = ["labeled", "non-labeled"]
	for i, lab in enumerate([ecdf_vals(label), ecdf_vals(no_label)]):
	    p.circle(
	        x=lab[:,0],
	        y=lab[:,1],
	        color=colors[i],
	        legend_label= legend_name[i], 
	        conf_int = True
	    )

	p1 = iqplot.stripbox(
	    data=df,
	    x_axis_label = 'Time (s)',
	    y_axis_label = 'Labeled?',
	    width=500,
	    height=500,
	    q='time to catastrophe (s)',
	    cats = 'labeled',
	    jitter=True,
	    top_level = "box",
	    whisker_kwargs=dict(line_color='#00000f', line_width=.8),
	    box_kwargs=dict(line_color='#00000f', line_width=.8),
	    median_kwargs=dict(line_color='#00000f', line_width=.8)
	)


	p.legend.location = 'bottom_right'
	p.legend.click_policy = "hide"
	bokeh.io.show(p1)
	bokeh.io.show(p)
	return



# Code from HW 6.1





