import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import linregress
import os

# Load the FITS file
file_path = 'data/nihao_uhd_simulation_g8.26e11_xyz_positions_and_oxygen_ao.fits'
with fits.open(file_path) as hdul:
    data = hdul[1].data

# Extract the x, y, z coordinates and A(O)
x = data['x']
y = data['y']
z = data['z']
A_O = data['A_O']

print(data)