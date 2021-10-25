# Shared dependencies:

import imageio
import scipy.stats
import scipy.special

import numpy as np
import scipy as sp
import pandas as pd
from numpy import array as arr
import tqdm

import os
import sys
import io

import shutil
import time
import glob
from natsort import natsorted
from pathlib import Path

import re
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits

from scipy.stats import sem
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.special import erf

from skimage import restoration
from skimage.feature import peak_local_max
from skimage import img_as_float

from PIL import Image

import pickle

#####
# Dependencies for HDF file wrapper class, ExpClass
import h5py as h5
from colorama import Fore, Style
#####

from astropy.modeling.models import Voigt1D
