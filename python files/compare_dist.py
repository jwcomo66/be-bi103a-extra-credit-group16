# Compare Gamma Distribution to Exponential Distribution when it comes to 
# the catastrophe times of microtubules
from functions import *
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

# Download and tidy the data
df =  pd.read_csv("data/gardner_mt_catastrophe_only_tubulin.csv", header=9)
df12 = pd.DataFrame(df['12 uM'])
df12.columns = ['time']
df7 = pd.DataFrame(df['7 uM'])
df9 = pd.DataFrame(df['9 uM'])
df10 = pd.DataFrame(df['10 uM'])
df14 = pd.DataFrame(df['14 uM'])
con = [12,7,9,10,14]
frames = [df12, df7, df9, df10, df14]

for i, df in enumerate(frames):
    # get name 
    name = df.columns.tolist()
    df.columns = ['time']
    df["concentration"] = con[i]

df = pd.concat(frames)
df = df.dropna()
df = df.sort_values('concentration')
df.head()


def compare_ecdfs(mle_exp, gamma_mle):
	size = 150
	beta_1 = mle_exp[0]
	beta_2 = mle_exp[1]
	b_1 = rg.exponential(1/beta_1, size)
	b_2 = rg.exponential(1/beta_2, size)
	t_c =  b_1 + b_2

	p = iqplot.ecdf(
	    data= pd.DataFrame(df.loc[df['concentration'] == 12, 'time']),
	    q = 'time',
	    kind='colored',
	)

	p.legend.location = 'top_left'
	t = np.linspace(0, 250000, 1500)/(1/beta_1)
	if beta_2 - beta_1 != 0:
	    # The normalized intensity
	    c = (beta_1 * beta_2)/(beta_2 - beta_1)
	    cdf = c*((1-np.exp(-beta_1*t))/beta_1 - (1-np.exp(-beta_2*t))/beta_2)

	    p.line(
	        x=t,
	        y=cdf,
	        line_width=2,
	        color = "red"
	    )
	p.line(
	        x=t,
	        y = gamma.cdf(t, gamma_mle[0], scale = 1/gamma_mle[1]), 
	        line_width = 2,
	        color = 'yellow'
	    )

	bokeh.io.show(p)
	return 

def show_ecdfs(concentration):

	data = pd.DataFrame(df.loc[df['concentration'] == concentration, 'time'])
	data = data["time"].values
	gamma_mle = mle_iid_gamma(data)
	mle_exp = mle_iid_exp((data))
	compare_ecdfs(mle_exp, gamma_mle)
	return 


