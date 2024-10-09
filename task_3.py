import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import os

# Loading FITS file
file_path = 'data/nihao_uhd_simulation_g8.26e11_xyz_positions_and_oxygen_ao.fits'
with fits.open(file_path) as hdul:
    data = hdul[1].data

# Extracting coordinates
x = data['x']
y = data['y']
z = data['z']
A_O = data['A_O']

# Calculating RGal
RGal = np.sqrt(x**2 + y**2 + z**2) / 1000  # Converting to kpc

# Filtering for RGal >= 25 kpc
mask = RGal < 25
RGal = RGal[mask]
A_O = A_O[mask]

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Parameters for figures
plt.rcParams['agg.path.chunksize'] = 2000  # Increase the path chunksize
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# (a) Logarithmic density plot of RGal vs A(O)
axs[0].scatter(RGal, A_O, alpha=0.5, label='Data Points')
axs[0].set_xlabel('$R_{Gal}$ (kpc)')
axs[0].set_ylabel('$A(O)$')
axs[0].set_title('Logarithmic Density Plot: $R_{Gal}$ vs. $A(O)$')

# Performing linear regression
slope, intercept, r_value, p_value, std_err = linregress(RGal, A_O)
line_fit = slope * RGal + intercept
axs[0].plot(RGal, line_fit, color='red', label='Linear Fit')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid()

# (b) Calculating residuals
residuals = A_O - line_fit

# Residual plot
axs[1].scatter(RGal, residuals, alpha=0.5)
axs[1].axhline(0, color='red', linestyle='--', label='Zero Residual')
axs[1].set_xlabel('$R_{Gal}$ (kpc)')
axs[1].set_ylabel(r'$\Delta A(O)$ (Residuals)')  # Use raw string to avoid escape issues
axs[1].set_title('Residuals of the Linear Fit: $R_{Gal}$ vs. $\Delta A(O)$')  # Use raw string
axs[1].legend()
axs[1].grid()

# Saving figures
plt.tight_layout()
plt.savefig('figures/rgal_vs_metallicity.png')
plt.show()

# Fitting
def linear_func(RGal, slope, intercept):
    return slope * RGal + intercept

optimal, variance = curve_fit(linear_func, RGal, A_O)

# Extracting slope and intercept
slope, intercept = optimal
slope_err, intercept_err = np.sqrt(np.diag(variance))

# Printing results
print(f"Slope: {slope:.4f} ± {slope_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_err:.4f}")

# Plotting
plt.scatter(RGal, A_O, label='Data Points', alpha=0.5)
plt.plot(RGal, linear_func(RGal, *optimal), color='red', label='Linear Fit')
plt.xlabel('$R_{Gal}$ (kpc)')
plt.ylabel('$A(O)$')
plt.title('Linear Fit of $A(O)$ vs. $R_{Gal}$')
plt.legend()
plt.grid()
plt.show()

# Calculating using linear model
A_O_fitted = linear_func(RGal, *optimal)

# Calculating residuals (observed - fitted)
residuals = A_O - A_O_fitted

# Calculating RMS error
rmse = np.sqrt(mean_squared_error(A_O, A_O_fitted))

# Calculating R^2
r_squared = r2_score(A_O, A_O_fitted)

# Printing
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r_squared:.4f}")

# Analyze where residuals are larger
large_residual_mask = np.abs(residuals) > (2 * np.std(residuals))  # Define large residuals
RGal_large_residuals = RGal[large_residual_mask]
A_O_large_residuals = A_O[large_residual_mask]
residuals_large = residuals[large_residual_mask]

print(f"Number of points with large residuals: {len(RGal_large_residuals)}")

# Defining no. of bins
bins = 100

# Creating 3-panel figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# (a) 2D-histogram of median simulated A(O)
h1 = axs[0].hist2d(x, y, bins=bins, weights=A_O, cmap='plasma')
axs[0].set_title('2D-Histogram of Median Simulated $A(O)$')
axs[0].set_xlabel('$x$ (kpc)')
axs[0].set_ylabel('$y$ (kpc)')
plt.colorbar(h1[3], ax=axs[0])

# (b) 2D-histogram of median fitted A(O)
h2 = axs[1].hist2d(x, y, bins=bins, weights=A_O_fitted, cmap='plasma')
axs[1].set_title('2D-Histogram of Median Fitted $A(O)$')
axs[1].set_xlabel('$x$ (kpc)')
axs[1].set_ylabel('$y$ (kpc)')
plt.colorbar(h2[3], ax=axs[1])

# (c) 2D-histogram of median residuals ΔA(O)
h3 = axs[2].hist2d(x, y, bins=bins, weights=residuals, cmap='seismic')
axs[2].set_title('2D-Histogram of Median Residuals $\\Delta A(O)$')
axs[2].set_xlabel('$x$ (kpc)')
axs[2].set_ylabel('$y$ (kpc)')
plt.colorbar(h3[3], ax=axs[2])

plt.tight_layout()
plt.savefig('figures/2d_histograms_residuals.png')
plt.show()

