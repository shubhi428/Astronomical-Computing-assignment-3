import os 
import matplotlib.pyplot as plt
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia

# ADQL query for fetching stars brighter than G = 14 within 1 degree of Messier 67
query = """
SELECT gaia.*, tmass.j_m, tmass.ks_m
FROM gaiadr3.gaia_source AS gaia
JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch
ON gaia.source_id = xmatch.source_id
JOIN gaiadr1.tmass_original_valid AS tmass
ON tmass.designation = xmatch.original_ext_source_id
WHERE CONTAINS(
  POINT('ICRS', gaia.ra, gaia.dec),
  CIRCLE('ICRS', 132.825, 11.8, 1)) = 1
AND gaia.phot_g_mean_mag < 14
"""

# Launching query to get results
job = Gaia.launch_job(query)
results = job.get_results()

# Printing available columns for debugging
print("Available columns in Gaia results:", results.colnames)

# Converting to NumPy array
ra = results['ra'].data
dec = results['dec'].data
parallax = results['parallax'].data

# Checking if 'ph_qual' exists (This was the error that I was receiving so checked it specifically)
if 'ph_qual' in results.colnames:
    ph_qual = results['ph_qual'].data
else:
    print("'ph_qual' not found in results. Proceeding without it.")
    ph_qual = None  # or any other default value you want to assign

bp_rp = results['bp_rp'].data
phot_g_mean_mag = results['phot_g_mean_mag'].data

# Filtering for 'ra' or 'dec' is NaN
valid_indices = np.isfinite(ra) & np.isfinite(dec)
ra = ra[valid_indices]
dec = dec[valid_indices]
parallax = parallax[valid_indices]

if ph_qual is not None:  # Checking specifically if ph_qual was retrieved successfully
    ph_qual = ph_qual[valid_indices]

bp_rp = bp_rp[valid_indices]
phot_g_mean_mag = phot_g_mean_mag[valid_indices]

# Defining SkyCoord objects with valid values
gaia_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

print("Gaia coordinates successfully created.") # printing for checking each step

# Accessing specific columns using vizier catalogue
vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Kmag'])

# For 1 degree of M67 (coordinates: RA=132.825, Dec=11.8125)
result_2mass = vizier.query_region(SkyCoord(ra=132.825 * u.deg, dec=11.8125 * u.deg), 
                                   radius=1 * u.deg, catalog='II/246')

# Checking columns of result
df_2mass = result_2mass[0]
print("Columns in 2MASS result:", df_2mass.columns)

# Extracting 2MASS data
tmass_ra = df_2mass['RAJ2000'].data
tmass_dec = df_2mass['DEJ2000'].data
tmass_j_m = df_2mass['Jmag'].data
tmass_ks_m = df_2mass['Kmag'].data

# Defining 2MASS coordinates
tmass_coords = SkyCoord(ra=tmass_ra * u.deg, dec=tmass_dec * u.deg)

# Crossmatch within 1 arcsecond 
idx, d2d, _ = gaia_coords.match_to_catalog_sky(tmass_coords)
sep_constraint = d2d < 1 * u.arcsec

# Crossmatched results
crossmatched_gaia = np.array([
    ra[idx[sep_constraint]],
    dec[idx[sep_constraint]],
    parallax[idx[sep_constraint]],
    ph_qual[idx[sep_constraint]] if ph_qual is not None else np.full(np.sum(sep_constraint), np.nan),  # Handle missing ph_qual
    bp_rp[idx[sep_constraint]],
    phot_g_mean_mag[idx[sep_constraint]],
])

crossmatched_tmass = np.array([
    tmass_j_m[idx[sep_constraint]],
    tmass_ks_m[idx[sep_constraint]],
])

# Displaying initial stars
initial_count = len(ra)
print(f'Initial number of stars: {initial_count}')

# Identifying stars with bad photometry
if 'ph_qual' in df_2mass.colnames:
    bad_photometry_mask = df_2mass['ph_qual'] != 'AAA'
    bad_photometry_stars = df_2mass[bad_photometry_mask]
    
    # Printing the number of stars with bad photometry(couldn't find any)
    print(f"Number of stars with bad 2MASS photometry: {len(bad_photometry_stars)}")
    print("Bad photometry stars:")
    print(bad_photometry_stars)
else:
    print("'ph_qual' column not found in 2MASS results. Proceeding without this check.")

# Identifying stars with non-positive parallaxes
non_positive_parallax_count = np.sum(crossmatched_gaia[2] <= 0)
print(f'Stars with non-positive parallaxes: {non_positive_parallax_count}')

# Quality cuts: keep only stars with positive parallaxes
positive_parallax_mask = crossmatched_gaia[2] > 0
filtered_indices = np.where(positive_parallax_mask)

filtered_count = len(filtered_indices[0])
print(f'Number of stars after quality cuts: {filtered_count}')

# Calculating absolute G magnitude
abs_G = crossmatched_gaia[5][filtered_indices] - 5 * (np.log10(1000 / crossmatched_gaia[2][filtered_indices]) + 1)

# Plotting
plt.figure(figsize=(12, 6))

# Panel (a): Color-Magnitude Diagram (CMD)
plt.subplot(1, 2, 1)
plt.scatter(crossmatched_gaia[4][filtered_indices], abs_G, color='blue', marker='o', s=10)
plt.xlabel('BP - RP')
plt.ylabel('Absolute G Magnitude')
plt.title('Color-Magnitude Diagram')
plt.gca().invert_yaxis()  # Inverting y-axis for magnitude

# Panel (b): 2MASS J-Ks vs. Apparent K diagram
plt.subplot(1, 2, 2)
plt.scatter(crossmatched_tmass[0][filtered_indices] - crossmatched_tmass[1][filtered_indices], 
            crossmatched_tmass[1][filtered_indices], color='red', marker='o', s=10)
plt.xlabel('J - K_s')
plt.ylabel('Apparent K Magnitude')
plt.title('2MASS J-Ks vs. Apparent K Magnitude Diagram')

# Ensuring the figures directory exists
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)  # Create the figures directory if it does not exist

# Saving figure
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'cmds_M67.png'), dpi=200)
plt.show()

# Recommendation for potential proposal
print("Focus fiber allocation on areas with high star density in CMD and 2MASS diagrams for efficient data gathering. This can be checked using the plot that has been generated with the use of filtered data.")
