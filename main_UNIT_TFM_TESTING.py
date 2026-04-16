#!/usr/bin/env python3
"""
===============================================================================
     GALAXY-HALO ANALYSIS PIPELINE: HOD CALIBRATION FROM SAM CATALOGUES
===============================================================================

Author:         Joaquín Delgado Amar, Violeta González Pérez
Supervisor:     Violeta González Pérez
Institution:    Universidad Autónoma de Madrid (UAM)
Created:        07 Jul 2025
Last Updated:   26 Aug 2025

-------------------------------------------------------------------------------
PURPOSE
-------------------------------------------------------------------------------

Analyse galaxy and halo catalogues from semi-analytical models to measure
the key ingredients of the galaxy-halo connection and export them as 
calibrated inputs for HOD-based mock generation.

The pipeline measures:
  • Halo Occupation Distributions (centrals and satellites).
  • Galactic conformity parameters (mass-dependent and global K1, K2).
  • Radial satellite profiles (fitted with extended NFW profile (Reyes-Pedraza, 2024)).
  • Velocity distributions of satellites:
       - Radial velocities: 3-Gaussian mixture (Reyes-Pedraza, 2024)
       - Tangential velocities: power-law with exponential cutoff (Reyes-Pedraza, 2024)
  • Two-point correlation functions (real and redshift space, via Corrfunc).
  • Kaiser analysis of redshift-space distortions.
  • Shuffling tests (halo/galaxy reshuffling for conformity baselines).

All results are stored in HDF5 files (`h2s_output.h5` and shuffled equivalent) 
and diagnostic plots, ensuring full reproducibility.

-------------------------------------------------------------------------------
INPUTS
-------------------------------------------------------------------------------

1) Galaxy catalogue (HDF5 or text): positions, velocities, halo associations, 
   fluxes (H_alpha).
2) Halo catalogue (HDF5 or text): IDs, masses, positions, velocities.

Optional:
   • Shuffled halo and galaxy catalogues (auto-generated if enabled).

-------------------------------------------------------------------------------
OUTPUTS
-------------------------------------------------------------------------------

• Calibrated parameters:
    - Extended NFW: (N0, alpha, beta, kappa, r0)
    - Radial velocities: (A_i, mu_i, sigma_i)
    - Tangential velocities: (v0, epsilon, omega, delta)
    - Conformity: K1, K2 (per mass bin and global)

• Master HDF5 files:
    - `h2s_output.h5` (original catalogues)
    - `h2s_output_shuffled.h5` (if DO_SHUFFLING=True)

• Figures (PNG): HODs, conformity, radial/velocity profiles, 2PCFs, Kaiser tests.

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------

NumPy, SciPy, Matplotlib, h5py, Corrfunc, and project modules:
  - src.h2s_io            : I/O and filtering
  - src.h2s_shuffle       : shuffling routines
  - src.h2s_conformity    : conformity estimators
  - src.h2s_profile_r     : radial profiles
  - src.h2s_profile_vel   : velocity profiles
  - src.h2s_corr          : correlation functions
  - src.h2s_plots         : plotting utilities

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

1) Set the control flags and input file paths at the top of this script.
2) Run from the project root directory:

>>> python main.py

Outputs will be stored in `output/` and intermediate data in `data/`.

-------------------------------------------------------------------------------
CONTACT
-------------------------------------------------------------------------------

joaquin.delgado@estudiante.uam.es
violetagp@protonmail.com

References: Avila et al. (2020); Vos Ginés et al. (2024); Reyes Pedraza (2024).
===============================================================================
"""

import os
import h5py
import numpy as np

# ===================================================================
# ============== 1. PARAMETER DEFINITION SECTION ====================
# ===================================================================

# Control parameters
TESTING = True              # If True, runs in testing mode (faster, less data), for Galaxies and Halos
VERBOSE = True               # If True, prints detailed information during execution
DO_PLOTS = True              # If True, generates plots for the results

# Cosmological parameters
OMEGA_M = 0.3089             # Matter density parameter
OMEGA_L = 0.6911             # Dark energy density parameter
h = 0.6774                   # Hubble constant (H0 = 100 * h km/s/Mpc)
Z_SNAP = 1.321               # Snapshot redshift

# Simulation parameters
BOXSIZE = 1000.0             # Box size in Mpc/h
FLUX_MIN = 1.325e-16       # Minimum Halpha flux for ELGs [erg/s/cm^2]
DO_FLUX_CUT = True           # If True, applies a flux cut to the galaxy sample

# Input and output file paths
BASE_DIR = "/home/tfm-violeta/Victor/HODfit2sim_victorrojalo"        # Base directory for the project
#BASE_DIR = "/home2/guillermo/HODfit2sim"        # Base directory for the project
RESULTS_DIR = os.path.join("/home/tfm-violeta/Victor/HODfit2sim_victorrojalo", "output")    # Directory for results  
DATA_DIR = os.path.join("/home2/guillermo/TFM_JOAQUIN", "data")        # Directory for input data files
os.makedirs(RESULTS_DIR, exist_ok=True)          # Create results directory if it doesn't exist
#os.makedirs(DATA_DIR, exist_ok=True)             # Create data directory if it doesn't exist

# Input files
GALAXY_FILE = os.path.join(DATA_DIR, "UNITSIM1_model_z1.321_ELGs.h5")        # Input galaxy file
HALO_FILE = os.path.join(DATA_DIR, "Halos_tree_DOC_PID_Vmax_all_Mass.txt")   # Input halo file
#print (HALO_FILE);exit()
# Input file formats
INPUT_GALAXY_FORMAT = "h5"                       # Input Halaxy file format ('h5', 'txt')
INPUT_HALO_FORMAT = "txt"                        # Input Halo file format ('h5', 'txt')     

# Galaxy file keys
if INPUT_GALAXY_FORMAT == "h5":
    GALAXY_MAIN_ID_KEY = "MainHaloID"            # Key for main halo ID 
    GALAXY_HOST_ID_KEY = "HostHaloID"            # Key for host halo ID 
    GALAXY_MAIN_MASS_KEY = "MainMhalo"           # Key for main halo mass
    GALAXY_XPOS_KEY = "Xpos"                     # Key for x position 
    GALAXY_YPOS_KEY = "Ypos"                     # Key for y position 
    GALAXY_ZPOS_KEY = "Zpos"                     # Key for z position 
    GALAXY_XVEL_KEY = "Xvel"                     # Key for x velocity 
    GALAXY_YVEL_KEY = "Yvel"                     # Key for y velocity
    GALAXY_ZVEL_KEY = "Zvel"                     # Key for z velocity 
    GALAXY_LOG_HALPHA_KEY = "logFHalpha_att"     # Key for Halpha flux 

if INPUT_GALAXY_FORMAT == "txt":
    GALAXY_MAIN_ID_KEY = 1                       # Key for main halo ID
    GALAXY_HOST_ID_KEY = 2                       # Key for host halo ID 
    GALAXY_MAIN_MASS_KEY = 3                     # Key for main halo mass 
    GALAXY_XPOS_KEY = 4                          # Key for x position 
    GALAXY_YPOS_KEY = 5                          # Key for y position
    GALAXY_ZPOS_KEY = 6                          # Key for z position 
    GALAXY_XVEL_KEY = 7                          # Key for x velocity 
    GALAXY_YVEL_KEY = 8                          # Key for y velocity 
    GALAXY_ZVEL_KEY = 9                          # Key for z velocity 
    GALAXY_LOG_HALPHA_KEY = 10                   # Key for Halpha flux 

# Halo file keys
if INPUT_HALO_FORMAT == "h5":
    HALO_XPOS_KEY = "Xpox"      # Key for x position
    HALO_YPOS_KEY = "Ypos"      # Key for y position
    HALO_ZPOS_KEY = "Zpos"      # Key for z position 
    HALO_MASS_KEY = "Mass"      # Key for halo mass
    HALO_XVEL_KEY = "Xvel"      # Key for x velocity
    HALO_YVEL_KEY = "Yvel"      # Key for y velocity
    HALO_ZVEL_KEY = "Zvel"      # Key for z velocity
    HALO_ID_KEY = "ID"          # Key for halo ID
    HALO_PID_KEY = "PID"        # Key for halo PID

if INPUT_HALO_FORMAT == "txt":
    HALO_XPOS_KEY = 3           # Key for x position
    HALO_YPOS_KEY = 4           # Key for y position 
    HALO_ZPOS_KEY = 5           # Key for z position
    HALO_MASS_KEY = 17          # Key for halo mass
    HALO_XVEL_KEY = 6           # Key for x velocity
    HALO_YVEL_KEY = 7           # Key for y velocity 
    HALO_ZVEL_KEY = 8           # Key for z velocity
    HALO_ID_KEY = 1             # Key for halo ID
    HALO_PID_KEY = 13           # Key for halo PID

# ===================================================================
# ========= 2. Two Point Correlation Function. Real Space ===========
# ===================================================================

DO_CORRELATION_FUNC_REAL_SPACE = True  # If True, computes the 2PCF in real space
NBINS_2PCF_REAL = 80                    # Number of bins
RMAX_2PCF_REAL = 140.0                  # Maximum distance in Mpc/h
RMIN_2PCF_REAL = 1.4e-3                 # Minimum distance in Mpc/h
NTHREADS_2PCF_REAL = 4                  # Number of threads for correlation function computation
DO_2PCF_REAL_RATIO = True               # If True, computes the ratio of 2PCF between original and shuffled catalogs
XLIM_2PCF_RATIO = (0.01, 100.0)         # X-axis limits for 2PCF ratio plot (None for automatic)
YLIM_2PCF_RATIO = (0.5, 1.4)            # Y-axis limits for 2PCF ratio plot (None for automatic)

# ====================================================================
# ======== 3. Two Point Correlation Function. Redshift Space =========
# ====================================================================

DO_CORRELATION_FUNC_REDSHIFT_SPACE = True  # If True, computes the 2PCF in redshift space
NBINS_2PCF_REDSHIFT = 80                   # Number of bins
RMAX_2PCF_REDSHIFT = 140.0                 # Maximum distance in Mpc/h
RMIN_2PCF_REDSHIFT = 1.4e-3                # Minimum distance in Mpc/h
NTHREADS_2PCF_REDSHIFT = 4                 # Number of threads for correlation function computation
LOS_AXIS = 'z'                             # Line of sight axis for redshift space (default is 'z')

# ====================================================================
# ================= 4. Conformity Parameters =========================
# ====================================================================
# If DO_PLOTS, plots the Halo Mass Function (HMF) and Halo Occupation Distribution (HOD).

DO_CONFORMITY = True                      # If True, computes conformity parameters
M_MIN_CONFORMITY = 10.5                   # Minimum halo mass for conformity analysis (logarithmic scale)
M_MAX_CONFORMITY = 14.5                   # Maximum halo mass for conformity analysis (logarithmic scale)
N_BINS_CONFORMITY = 70                    # Number of bins for halo mass in conformity analysis
BIN_WIDTH_CONFORMITY = 0.057              # Width of each bin in logarithmic scale
DO_HMF_COMPARISON = True                  # If True, compares halo mass functions (HMF) between original and shuffled catalogs
BINNING_HMF= np.linspace(10.5, 14.5, 71)  # Mass binning for HMF comparison in logarithmic scale

# ====================================================================
# ===================== 5. Radial Profile ============================
# ====================================================================

DO_RADIAL_PROFILE = True                   # If True, computes the radial profile
BINNING_RADIAL = np.linspace(0, 1.5, 151)  # Binning for radial profile in Mpc/h
DO_FIT_RADIAL_PROFILE = True               # If True, fits the radial profile  
INITAL_GUESS_RADIAL = None                 # Initial guess for fitting parameters (if needed)

# ====================================================================
# ================= 8. Radial Velocity Profile =======================
# ====================================================================

DO_VR_PROFILE = True                         # If True, computes the radial velocity profile
BINNING_VR = np.linspace(-1000, 1000, 201)   # Binning for radial velocity profile in km/s
DO_FIT_VR_PROFILE = True                     # If True, fits the radial velocity profile
MANUAL_PARAMS_VR = None                      # Manual parameters for fitting (if needed)

# ====================================================================
# ================= 9. Tangential Velocity Profile ===================
# ===================================================================

DO_VTAN_PROFILE = True                    # If True, computes the tangential velocity profile
BINNING_VTAN = np.linspace(0, 1000, 201)  # Binning for tangential velocity profile in km/s
DO_FIT_VTAN_PROFILE = True                # If True, fits the tangential velocity profile
MANUAL_PARAMS_VTAN = None                 # Manual parameters for fitting (if needed)

# ====================================================================
# ================= 10. Kaiser Analysis ==============================
# ====================================================================

DO_KAISER_ANALYSIS = True            # If True, performs Kaiser analysis
BIAS = 1.86                          # Bias parameter for ELGs
GAMMA = 0.55                         # Gamma parameter for Kaiser analysis
DO_KAISER_RATIO = True               # If True, computes Kaiser ratio between redshift and real space
XLIM_KAISER = (0.1,100)              # X-axis limits for Kaiser ratio plot (None for automatic)
YLIM_KAISER = (0.75,1.5)             # Y-axis limits for Kaiser ratio plot (None for automatic)

# ====================================================================
# ===================== 11. Shuffling ================================
# ====================================================================

DO_SHUFFLING = True               # If True, shuffles the galaxy catalogue
N_BINS_SHUFFLING = 70              # Number of bins for halo mass in shuffling
M_MIN_SHUFFLING = 10.5             # Minimum halo mass for shuffling (logarithmic scale)
M_MAX_SHUFFLING = 14.5             # Maximum halo mass for shuffling (logarithmic scale)



def main():

    # ================== Apply flux cut =================
    if DO_FLUX_CUT:
        print("Applying flux cut...")
        from src.h2s_io import filter_log_flux
        FILTERED_GALAXIES = os.path.join(BASE_DIR, f"filtered_ELGs_{FLUX_MIN}.{INPUT_GALAXY_FORMAT}") 
        filter_log_flux(infile=GALAXY_FILE, fmin=FLUX_MIN, outfile_name=FILTERED_GALAXIES, 
                        output_format=INPUT_GALAXY_FORMAT, param_name=GALAXY_LOG_HALPHA_KEY, verbose=VERBOSE, testing=TESTING)
                         
                        
        
    # ================= Galaxy Shuffling ================

    if DO_SHUFFLING:
        print("Splitting Halo Catalogue by mass...")
        from src.h2s_io import split_halo_catalog_by_mass
        HALO_MASS_BINS = os.path.join(BASE_DIR, "halo_mass_bins.h5") 
        split_halo_catalog_by_mass(halo_file=HALO_FILE, mass_column=HALO_MASS_KEY, logmass = False,
                                      output_h5=HALO_MASS_BINS, min_logmass=M_MIN_SHUFFLING, max_logmass=M_MAX_SHUFFLING,
                                      n_bins=N_BINS_SHUFFLING, columns={"id": HALO_ID_KEY, "X": HALO_XPOS_KEY,
                                      "Y": HALO_YPOS_KEY, "Z": HALO_ZPOS_KEY,"vx": HALO_XVEL_KEY, "vy": HALO_YVEL_KEY,
                                                                        "vz": HALO_ZVEL_KEY, "pid": HALO_PID_KEY,"Mass": HALO_MASS_KEY}, verbose=VERBOSE, testing=TESTING)
        
        print("Shuffling halo catalogue...")
        from src.h2s_shuffle import shuffle_parent_halos
        SHUFFLED_HALO_FILE = os.path.join(BASE_DIR, "halo_mass_bins_shuffled.h5") 
        shuffle_parent_halos(input_hdf5=HALO_MASS_BINS, output_hdf5 = SHUFFLED_HALO_FILE, verbose=VERBOSE,
                              rng_seed=None)
        
        print("Shuffling galaxy catalogue...")
        from src.h2s_shuffle import shuffle_galaxy_catalog_binned
        from src.h2s_profile_r import boundary_correction
        if DO_FLUX_CUT:
            SHUFFLED_GALAXY_FILE = os.path.join(BASE_DIR, f"filtered_ELGs_{FLUX_MIN}_shuffled.h5")   
        else:
            SHUFFLED_GALAXY_FILE = os.path.join(BASE_DIR, "filtered_ELGs_shuffled.h5")    
        shuffle_galaxy_catalog_binned(galaxy_file=FILTERED_GALAXIES, halo_shuffled_file=SHUFFLED_HALO_FILE, output_file=SHUFFLED_GALAXY_FILE,
                                      boxsize=BOXSIZE, bins=N_BINS_SHUFFLING, boundary_correction=boundary_correction,
                                      verbose=VERBOSE)

    # =========== Calculate 2PCF. Real Space ============

    if DO_CORRELATION_FUNC_REAL_SPACE:
        print("Calculating correlation function in real space...")
        from src.h2s_corr import compute_correlation_corrfunc, export_positions

        if DO_FLUX_CUT:
            POSITIONS_GALAXIES = os.path.join(BASE_DIR, f"positions_{FLUX_MIN}.txt")  
            GALAXY_2PCF_REAL = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_{FLUX_MIN}.txt")
            POSITIONS_GALAXIES_SHUFFLED = os.path.join(BASE_DIR, f"positions_shuffled_{FLUX_MIN}.txt")  
            GALAXY_2PCF_REAL_SHUFFLED = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_shuffled_{FLUX_MIN}.txt")
        else:
            POSITIONS_GALAXIES = os.path.join(BASE_DIR, "positions.txt")   
            GALAXY_2PCF_REAL = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_.txt")
            POSITIONS_GALAXIES_SHUFFLED = os.path.join(BASE_DIR, "positions_shuffled.txt")  
            GALAXY_2PCF_REAL_SHUFFLED = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_shuffled.txt")

        if DO_SHUFFLING:
            export_positions(SHUFFLED_GALAXY_FILE, POSITIONS_GALAXIES_SHUFFLED,
                            xkey=GALAXY_XPOS_KEY, ykey=GALAXY_YPOS_KEY, zkey=GALAXY_ZPOS_KEY, verbose=VERBOSE)
            
            compute_correlation_corrfunc(POSITIONS_GALAXIES_SHUFFLED, GALAXY_2PCF_REAL_SHUFFLED, boxsize=BOXSIZE,
                                         n_bins= NBINS_2PCF_REAL, rmax = RMAX_2PCF_REAL, rmin=RMIN_2PCF_REAL, n_threads=NTHREADS_2PCF_REAL,
                                         verbose=VERBOSE)
        
        export_positions(FILTERED_GALAXIES if DO_FLUX_CUT else GALAXY_FILE, POSITIONS_GALAXIES,
                            xkey=GALAXY_XPOS_KEY, ykey=GALAXY_YPOS_KEY, zkey=GALAXY_ZPOS_KEY, verbose=VERBOSE)
            
        compute_correlation_corrfunc(POSITIONS_GALAXIES, GALAXY_2PCF_REAL, boxsize=BOXSIZE, n_bins= NBINS_2PCF_REAL,
                                     rmax = RMAX_2PCF_REAL, rmin=RMIN_2PCF_REAL, n_threads=NTHREADS_2PCF_REAL,
                                     verbose=VERBOSE)
        
        if DO_PLOTS:
            from src.h2s_plots import plot_correlation_function, plot_2pcf_ratio
            if DO_FLUX_CUT:
                PLOT_2PCF_REAL = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_{FLUX_MIN}.png")
                PLOT_2PCF_REAL_SHUFFLED = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_shuffled_{FLUX_MIN}.png")
            else:
                PLOT_2PCF_REAL = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_.png")
                PLOT_2PCF_REAL_SHUFFLED = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_shuffled.png")
            plot_correlation_function(GALAXY_2PCF_REAL, PLOT_2PCF_REAL, loglog=True, show=True,
                                      r_index=0, xi_index=1, err_index=2)
            if DO_SHUFFLING:
                plot_correlation_function(GALAXY_2PCF_REAL_SHUFFLED, PLOT_2PCF_REAL_SHUFFLED, loglog=True, show=True,
                                          r_index=0, xi_index=1, err_index=2)
                if DO_2PCF_REAL_RATIO:
                    if DO_FLUX_CUT:
                        PLOT_2PCF_RATIO = os.path.join(RESULTS_DIR, f"2PCF_ratio_{FLUX_MIN}.png")
                    else:
                        PLOT_2PCF_RATIO = os.path.join(RESULTS_DIR, "2PCF_ratio.png")
                    plot_2pcf_ratio(file_normal=GALAXY_2PCF_REAL, file_shuffled=GALAXY_2PCF_REAL_SHUFFLED,
                                    output_png=PLOT_2PCF_RATIO, xlim=XLIM_2PCF_RATIO, ylim=YLIM_2PCF_RATIO, show=True)
                
            
    # ========= Calculate 2PCF. Redshift Space =========

    if DO_CORRELATION_FUNC_REDSHIFT_SPACE:
        print("Calculating correlation function in redshift space...")
        from src.h2s_corr import export_positions_redshift_space, compute_correlation_corrfunc

        if DO_FLUX_CUT:
            POSITIONS_GALAXIES_RS = os.path.join(BASE_DIR, f"positions_redshift_{FLUX_MIN}.txt") 
            GALAXY_2PCF_REDSHIFT = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_redshift_{FLUX_MIN}.txt")
            POSITIONS_GALAXIES_RS_SHUFFLED = os.path.join(BASE_DIR, f"positions_redshift_shuffled_{FLUX_MIN}.txt") 
            GALAXY_2PCF_REDSHIFT_SHUFFLED = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_redshift_shuffled_{FLUX_MIN}.txt")
        else:
            POSITIONS_GALAXIES_RS = os.path.join(BASE_DIR, "positions_redshift.txt") 
            GALAXY_2PCF_REDSHIFT = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_redshift.txt")
            POSITIONS_GALAXIES_RS_SHUFFLED = os.path.join(BASE_DIR, "positions_redshift_shuffled.txt")  
            GALAXY_2PCF_REDSHIFT_SHUFFLED = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_redshift_shuffled.txt")

        if DO_SHUFFLING:
            export_positions_redshift_space(infile=SHUFFLED_GALAXY_FILE,
                                           outfile= POSITIONS_GALAXIES_RS_SHUFFLED, z_snap=Z_SNAP, Omega_L=OMEGA_L,
                                           Omega_m=OMEGA_M, h=h, xkey=GALAXY_XPOS_KEY, ykey=GALAXY_YPOS_KEY, 
                                           zkey=GALAXY_ZPOS_KEY, xvel_key=GALAXY_XVEL_KEY,yvel_key=GALAXY_YVEL_KEY,
                                           zvel_key=GALAXY_ZVEL_KEY, los_axis=LOS_AXIS, verbose=VERBOSE)
            
            compute_correlation_corrfunc(POSITIONS_GALAXIES_RS_SHUFFLED, GALAXY_2PCF_REDSHIFT_SHUFFLED, boxsize=BOXSIZE, 
                                     n_bins= NBINS_2PCF_REDSHIFT,rmax = RMAX_2PCF_REDSHIFT, rmin=RMIN_2PCF_REDSHIFT,
                                     n_threads=NTHREADS_2PCF_REDSHIFT, verbose=VERBOSE)
        
            
        export_positions_redshift_space(infile=FILTERED_GALAXIES if DO_FLUX_CUT else GALAXY_FILE,
                                           outfile= POSITIONS_GALAXIES_RS, z_snap=Z_SNAP, Omega_L=OMEGA_L,
                                          Omega_m=OMEGA_M, h=h, xkey=GALAXY_XPOS_KEY, ykey=GALAXY_YPOS_KEY, 
                                          zkey=GALAXY_ZPOS_KEY, xvel_key=GALAXY_XVEL_KEY,yvel_key=GALAXY_YVEL_KEY,
                                          zvel_key=GALAXY_ZVEL_KEY, los_axis=LOS_AXIS, verbose=VERBOSE)

        compute_correlation_corrfunc(POSITIONS_GALAXIES_RS, GALAXY_2PCF_REDSHIFT, boxsize=BOXSIZE, 
                                     n_bins= NBINS_2PCF_REDSHIFT,rmax = RMAX_2PCF_REDSHIFT, rmin=RMIN_2PCF_REDSHIFT,
                                     n_threads=NTHREADS_2PCF_REDSHIFT, verbose=VERBOSE)
        
        if DO_PLOTS:
            from src.h2s_plots import plot_correlation_function
            if DO_FLUX_CUT:
                PLOT_2PCF_REDSHIFT = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_redshift_{FLUX_MIN}.png")
                PLOT_2PCF_REDSHIFT_SHUFFLED = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_redshift_shuffled_{FLUX_MIN}.png")
            else:
                PLOT_2PCF_REDSHIFT = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_redshift.png")
                PLOT_2PCF_REDSHIFT_SHUFFLED = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_redshift_shuffled.png")
            plot_correlation_function(GALAXY_2PCF_REDSHIFT, PLOT_2PCF_REDSHIFT, loglog=True, show=True,
                                      r_index=0, xi_index=1, err_index=2)       
            if DO_SHUFFLING:
                plot_correlation_function(GALAXY_2PCF_REDSHIFT_SHUFFLED, PLOT_2PCF_REDSHIFT_SHUFFLED, loglog=True, show=True,
                                          r_index=0, xi_index=1, err_index=2)

    # ========= Calculate Conformity Parameters =========
    # If DO_PLOTS, plots the Halo Mass Function (HMF) and Halo Occupation Distribution (HOD).

    if DO_CONFORMITY:
        print("Calculating conformity parameters...")
        from src.h2s_conformity import compute_conformity_parameters, compute_conformity_parameters_shuffled

        if DO_FLUX_CUT:
            CONFORMITY = os.path.join(RESULTS_DIR, f"conformity_{FLUX_MIN}.h5")
            CONFORMITY_SHUFFLED = os.path.join(RESULTS_DIR, f"conformity_shuffled_{FLUX_MIN}.h5")
        else:
            CONFORMITY = os.path.join(RESULTS_DIR, "conformity.h5")
            CONFORMITY_SHUFFLED = os.path.join(RESULTS_DIR, "conformity_shuffled.h5")

        compute_conformity_parameters(galaxy_file=FILTERED_GALAXIES if DO_FLUX_CUT else GALAXY_FILE,
                                      halo_file=HALO_FILE, M_min= M_MIN_CONFORMITY,
                                      M_max= M_MAX_CONFORMITY, N_bins=N_BINS_CONFORMITY,
                                      galaxy_format=INPUT_GALAXY_FORMAT, halo_format= INPUT_HALO_FORMAT,
                                      output_file=CONFORMITY, halo_pid=HALO_PID_KEY, halo_mass=HALO_MASS_KEY,
                                      verbose=VERBOSE, gal_host_id= GALAXY_HOST_ID_KEY,
                                      gal_main_id= GALAXY_MAIN_ID_KEY, gal_main_mass= GALAXY_MAIN_MASS_KEY, testing=TESTING)
        
        if DO_SHUFFLING:
            compute_conformity_parameters_shuffled(galaxy_file=SHUFFLED_GALAXY_FILE,
                                                   halo_file=HALO_FILE, M_min=M_MIN_CONFORMITY,
                                                   M_max=M_MAX_CONFORMITY, N_bins=N_BINS_CONFORMITY,
                                                   galaxy_format=INPUT_GALAXY_FORMAT, halo_format=INPUT_HALO_FORMAT,
                                                   output_file=CONFORMITY_SHUFFLED, halo_pid=HALO_PID_KEY, halo_mass=HALO_MASS_KEY,
                                                   verbose=VERBOSE, gal_is_central="is_central",
                                                   gal_main_id=GALAXY_MAIN_ID_KEY, gal_main_mass=GALAXY_MAIN_MASS_KEY, testing=TESTING)

        if DO_PLOTS:
            from src.h2s_plots import plot_hmf, plot_hod
            if DO_FLUX_CUT:
                PLOT_HMF = os.path.join(RESULTS_DIR, f"HMF_{FLUX_MIN}.png")
                PLOT_HOD = os.path.join(RESULTS_DIR, f"HOD_Occupation_{FLUX_MIN}.png")
            else:
                PLOT_HMF = os.path.join(RESULTS_DIR, "HMF.png")
                PLOT_HOD = os.path.join(RESULTS_DIR, "HOD_Occupation.png")

            plot_hmf(h5file=CONFORMITY, output_path=PLOT_HMF, box_size=BOXSIZE,
                     logM_min=M_MIN_CONFORMITY, logM_max=M_MAX_CONFORMITY, n_bins=N_BINS_CONFORMITY, show=True)
            plot_hod(h5file=CONFORMITY, output_path=PLOT_HOD, show=True)

            if DO_HMF_COMPARISON:
                print("Comparing HMF between original and shuffled catalogs...")
                from src.h2s_plots import plot_hmf_comparison
                if DO_FLUX_CUT:
                    PLOT_HMF_COMPARISON = os.path.join(RESULTS_DIR, f"HMF_comparison_{FLUX_MIN}.png")
                else:
                    PLOT_HMF_COMPARISON = os.path.join(RESULTS_DIR, "HMF_comparison.png")
                plot_hmf_comparison(conformity_file=CONFORMITY, halo_bins_file=SHUFFLED_HALO_FILE, bins=BINNING_HMF,
                                    output_png=PLOT_HMF_COMPARISON, boxsize=BOXSIZE, show=True, loglog=True)


    #=============== Calculate Radial Profile ===============
    
    if DO_RADIAL_PROFILE:
        print("Calculating radial profile...")
        from src.h2s_profile_r import compute_radial_profile, compute_radial_profile_shuffled

        if DO_FLUX_CUT:
            RADIAL_PROFILE = os.path.join(RESULTS_DIR, f"radial_profile_{FLUX_MIN}.h5")
            RADIAL_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, f"radial_profile_shuffled_{FLUX_MIN}.h5")
        else:
            RADIAL_PROFILE = os.path.join(RESULTS_DIR, "radial_profile.h5")
            RADIAL_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, "radial_profile_shuffled.h5")

        compute_radial_profile(galaxy_file=FILTERED_GALAXIES if DO_FLUX_CUT else GALAXY_FILE,
                               halo_file=HALO_FILE, output_file=RADIAL_PROFILE, verbose=VERBOSE,
                               boxsize=BOXSIZE, bins=BINNING_RADIAL, halo_format=INPUT_HALO_FORMAT,
                               halo_id_key=HALO_ID_KEY, halo_pid_key=HALO_PID_KEY, halo_x_key=HALO_XPOS_KEY,
                               halo_y_key=HALO_YPOS_KEY, halo_z_key=HALO_ZPOS_KEY,
                               galaxy_host_key=GALAXY_HOST_ID_KEY, galaxy_id_key=GALAXY_MAIN_ID_KEY,
                               galaxy_x_key=GALAXY_XPOS_KEY, galaxy_y_key=GALAXY_YPOS_KEY,
                               galaxy_z_key=GALAXY_ZPOS_KEY, testing=TESTING)
        
        if DO_SHUFFLING:
            compute_radial_profile_shuffled(galaxy_file=SHUFFLED_GALAXY_FILE, halo_file=HALO_FILE,
                                              output_file=RADIAL_PROFILE_SHUFFLED, verbose=VERBOSE,
                                              boxsize=BOXSIZE, bins=BINNING_RADIAL, halo_format=INPUT_HALO_FORMAT,
                                              halo_id_key=HALO_ID_KEY, halo_pid_key=HALO_PID_KEY, halo_x_key=HALO_XPOS_KEY,
                                              halo_y_key=HALO_YPOS_KEY, halo_z_key=HALO_ZPOS_KEY,
                                              galaxy_is_central_key='is_central', galaxy_id_key=GALAXY_MAIN_ID_KEY,
                                              galaxy_x_key=GALAXY_XPOS_KEY, galaxy_y_key=GALAXY_YPOS_KEY,
                                              galaxy_z_key=GALAXY_ZPOS_KEY, testing=TESTING)

        if DO_PLOTS:
            from src.h2s_plots import plot_radial_profile
            if DO_FLUX_CUT:
                PLOT_RADIAL_PROFILE = os.path.join(RESULTS_DIR, f"radial_profile_{FLUX_MIN}.png")
            else:
                PLOT_RADIAL_PROFILE = os.path.join(RESULTS_DIR, "radial_profile.png")
            plot_radial_profile(profile_file=RADIAL_PROFILE, output_png=PLOT_RADIAL_PROFILE,
                                loglog=True, show=True)
        
        if DO_FIT_RADIAL_PROFILE:
            print("Fitting radial profile...")
            from src.h2s_profile_r import fit_radial_profile
            if DO_FLUX_CUT:
                FIT_RADIAL_PROFILE_PARAMS = os.path.join(RESULTS_DIR, f"radial_profile_fit_params{FLUX_MIN}.txt")
                FIT_RADIAL_PROFILE_PARAMS_SHUFFLED = os.path.join(RESULTS_DIR, f"radial_profile_fit_params_shuffled{FLUX_MIN}.txt")
            else:
                FIT_RADIAL_PROFILE_PARAMS = os.path.join(RESULTS_DIR, "radial_profile_fit_params.txt")
                FIT_RADIAL_PROFILE_PARAMS_SHUFFLED = os.path.join(RESULTS_DIR, "radial_profile_fit_params_shuffled.txt")
            fit_radial_profile(profile_file=RADIAL_PROFILE, output_params_file=FIT_RADIAL_PROFILE_PARAMS,
                               initial_guess= INITAL_GUESS_RADIAL, bounds=None, verbose=VERBOSE)
            if DO_SHUFFLING:
                fit_radial_profile(profile_file=RADIAL_PROFILE_SHUFFLED, output_params_file=FIT_RADIAL_PROFILE_PARAMS_SHUFFLED,
                                   initial_guess= INITAL_GUESS_RADIAL, bounds=None, verbose=VERBOSE)
            if DO_PLOTS:
                from src.h2s_plots import plot_radial_profile_fit
                if DO_FLUX_CUT:
                    FIT_RADIAL_PROFILE = os.path.join(RESULTS_DIR, f"radial_profile_fit_{FLUX_MIN}.png")
                else:
                    FIT_RADIAL_PROFILE = os.path.join(RESULTS_DIR, "radial_profile_fit.png")
                
                params_radial = np.loadtxt(FIT_RADIAL_PROFILE_PARAMS, delimiter=",", skiprows=1).flatten()
                params_radial_shuffled = np.loadtxt(FIT_RADIAL_PROFILE_PARAMS_SHUFFLED, delimiter=",", skiprows=1).flatten()
                plot_radial_profile_fit(profile_file=RADIAL_PROFILE, output_png=FIT_RADIAL_PROFILE,
                                        params = params_radial, show=True, loglog=True)

    # ========== Calculate Radial Velocity Profile ===========

    if DO_VR_PROFILE:
        print("Calculating radial velocity profile...")
        from src.h2s_profile_vel import compute_vr_profile, compute_vr_profile_shuffled

        if DO_FLUX_CUT:
            VR_PROFILE = os.path.join(RESULTS_DIR, f"vr_profile_{FLUX_MIN}.h5")
            VR_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, f"vr_profile_shuffled_{FLUX_MIN}.h5")
        else:
            VR_PROFILE = os.path.join(RESULTS_DIR, "vr_profile.h5")
            VR_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, "vr_profile_shuffled.h5")

        compute_vr_profile(galaxy_file=FILTERED_GALAXIES if DO_FLUX_CUT else GALAXY_FILE,
                            halo_file=HALO_FILE, output_file=VR_PROFILE,verbose=VERBOSE, boxsize=BOXSIZE, 
                            bins=BINNING_VR, halo_format=INPUT_HALO_FORMAT, halo_id_key=HALO_ID_KEY,
                            halo_pid_key=HALO_PID_KEY, halo_x_key=HALO_XPOS_KEY, halo_y_key=HALO_YPOS_KEY,
                            halo_z_key=HALO_ZPOS_KEY, halo_vx_key=HALO_XVEL_KEY, halo_vy_key=HALO_YVEL_KEY,
                            halo_vz_key=HALO_ZVEL_KEY, galaxy_host_id_key=GALAXY_HOST_ID_KEY,
                            galaxy_main_id_key=GALAXY_MAIN_ID_KEY, galaxy_x_key=GALAXY_XPOS_KEY,
                            galaxy_y_key=GALAXY_YPOS_KEY, galaxy_z_key=GALAXY_ZPOS_KEY,
                            galaxy_vx_key=GALAXY_XVEL_KEY, galaxy_vy_key=GALAXY_YVEL_KEY,
                            galaxy_vz_key=GALAXY_ZVEL_KEY,testing=TESTING)
        
        if DO_SHUFFLING:
            compute_vr_profile_shuffled(galaxy_file=SHUFFLED_GALAXY_FILE,
                                         halo_file=HALO_FILE, output_file=VR_PROFILE_SHUFFLED, verbose=VERBOSE,
                                         boxsize=BOXSIZE, bins=BINNING_VR, halo_format=INPUT_HALO_FORMAT,
                                         halo_id_key=HALO_ID_KEY, halo_pid_key=HALO_PID_KEY,
                                         halo_x_key=HALO_XPOS_KEY, halo_y_key=HALO_YPOS_KEY,
                                         halo_z_key=HALO_ZPOS_KEY, halo_vx_key=HALO_XVEL_KEY,
                                         halo_vy_key=HALO_YVEL_KEY, halo_vz_key=HALO_ZVEL_KEY,
                                         galaxy_is_central_key="is_central",
                                         galaxy_main_id_key=GALAXY_MAIN_ID_KEY, galaxy_x_key=GALAXY_XPOS_KEY,
                                         galaxy_y_key=GALAXY_YPOS_KEY, galaxy_z_key=GALAXY_ZPOS_KEY,
                                         galaxy_vx_key=GALAXY_XVEL_KEY, galaxy_vy_key=GALAXY_YVEL_KEY,
                                         galaxy_vz_key=GALAXY_ZVEL_KEY,testing=TESTING)

        if DO_PLOTS:
            from src.h2s_plots import plot_vr_distribution
            if DO_FLUX_CUT:
                PLOT_VR_PROFILE = os.path.join(RESULTS_DIR, f"vr_profile_{FLUX_MIN}.png")
            else:
                PLOT_VR_PROFILE = os.path.join(RESULTS_DIR, "vr_profile.png")
            plot_vr_distribution(vr_profile_file=VR_PROFILE, output_png=PLOT_VR_PROFILE, loglog=False, show=True)
        
        if DO_FIT_VR_PROFILE:
            print("Fitting radial velocity profile...")
            from src.h2s_profile_vel import fit_vr_profile
            if DO_FLUX_CUT:
                FIT_VR_PROFILE = os.path.join(RESULTS_DIR, f"vr_profile_fit_{FLUX_MIN}.png")
                FIT_VR_PROFILE_PARAMS = os.path.join(RESULTS_DIR, f"vr_profile_fit_params_{FLUX_MIN}.txt")
                FIT_VR_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, f"vr_profile_fit_shuffled_{FLUX_MIN}.png")
                FIT_VR_PROFILE_PARAMS_SHUFFLED = os.path.join(RESULTS_DIR, f"vr_profile_fit_params_shuffled_{FLUX_MIN}.txt")
            else:
                FIT_VR_PROFILE_PARAMS = os.path.join(RESULTS_DIR, "vr_profile_fit_params.txt")
                FIT_VR_PROFILE = os.path.join(RESULTS_DIR, "vr_profile_fit.png")
                FIT_VR_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, "vr_profile_fit_shuffled.png")
                FIT_VR_PROFILE_PARAMS_SHUFFLED = os.path.join(RESULTS_DIR, "vr_profile_fit_params_shuffled.txt")

            fit_vr_profile(vr_profile_file=VR_PROFILE, plot=True, output_png=FIT_VR_PROFILE,
                           loglog=False, manual_params=MANUAL_PARAMS_VR, output_params_file=FIT_VR_PROFILE_PARAMS)
            if DO_SHUFFLING:
                fit_vr_profile(vr_profile_file=VR_PROFILE_SHUFFLED, plot=True, output_png=FIT_VR_PROFILE_SHUFFLED,
                               loglog=False, manual_params=MANUAL_PARAMS_VR, output_params_file=FIT_VR_PROFILE_PARAMS_SHUFFLED)

    # ========== Calculate Tangential Velocity Profile ===========

    if DO_VTAN_PROFILE:
        print("Calculating tangential velocity profile...")
        from src.h2s_profile_vel import compute_vtan_profile, compute_vtan_profile_shuffled
        if DO_FLUX_CUT:
            VTAN_PROFILE = os.path.join(RESULTS_DIR, f"vtan_profile_{FLUX_MIN}.h5")
            VTAN_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, f"vtan_profile_shuffled_{FLUX_MIN}.h5")
        else:
            VTAN_PROFILE = os.path.join(RESULTS_DIR, "vtan_profile.h5")
            VTAN_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, "vtan_profile_shuffled.h5")

        compute_vtan_profile(galaxy_file=FILTERED_GALAXIES if DO_FLUX_CUT else GALAXY_FILE,
                             halo_file=HALO_FILE, output_file=VTAN_PROFILE, verbose=VERBOSE, boxsize=BOXSIZE,
                             bins=BINNING_VTAN, halo_format=INPUT_HALO_FORMAT, halo_id_key=HALO_ID_KEY,
                             halo_pid_key=HALO_PID_KEY, halo_x_key=HALO_XPOS_KEY, halo_y_key=HALO_YPOS_KEY,
                             halo_z_key=HALO_ZPOS_KEY, halo_vx_key=HALO_XVEL_KEY, halo_vy_key=HALO_YVEL_KEY,
                             halo_vz_key=HALO_ZVEL_KEY, galaxy_host_id_key=GALAXY_HOST_ID_KEY,
                             galaxy_main_id_key=GALAXY_MAIN_ID_KEY, galaxy_x_key=GALAXY_XPOS_KEY,
                             galaxy_y_key=GALAXY_YPOS_KEY, galaxy_z_key=GALAXY_ZPOS_KEY,
                             galaxy_vx_key=GALAXY_XVEL_KEY, galaxy_vy_key=GALAXY_YVEL_KEY,
                             galaxy_vz_key=GALAXY_ZVEL_KEY, testing=TESTING)
        
        if DO_SHUFFLING:
            compute_vtan_profile_shuffled(galaxy_file=SHUFFLED_GALAXY_FILE,
                                           halo_file=HALO_FILE, output_file=VTAN_PROFILE_SHUFFLED, verbose=VERBOSE,
                                           boxsize=BOXSIZE, bins=BINNING_VTAN, halo_format=INPUT_HALO_FORMAT,
                                           halo_id_key=HALO_ID_KEY, halo_pid_key=HALO_PID_KEY,
                                           halo_x_key=HALO_XPOS_KEY, halo_y_key=HALO_YPOS_KEY,
                                           halo_z_key=HALO_ZPOS_KEY, halo_vx_key=HALO_XVEL_KEY,
                                           halo_vy_key=HALO_YVEL_KEY, halo_vz_key=HALO_ZVEL_KEY,
                                           galaxy_is_central_key="is_central",
                                           galaxy_main_id_key=GALAXY_MAIN_ID_KEY, galaxy_x_key=GALAXY_XPOS_KEY,
                                           galaxy_y_key=GALAXY_YPOS_KEY, galaxy_z_key=GALAXY_ZPOS_KEY,
                                           galaxy_vx_key=GALAXY_XVEL_KEY, galaxy_vy_key=GALAXY_YVEL_KEY,
                                           galaxy_vz_key=GALAXY_ZVEL_KEY,testing=TESTING)

        if DO_PLOTS:
            from src.h2s_plots import plot_vtan_distribution
            if DO_FLUX_CUT:
                PLOT_VTAN_PROFILE = os.path.join(RESULTS_DIR, f"vtan_profile_{FLUX_MIN}.png")
            else:
                PLOT_VTAN_PROFILE = os.path.join(RESULTS_DIR, "vtan_profile.png")
            plot_vtan_distribution(vtan_profile_file=VTAN_PROFILE, output_png=PLOT_VTAN_PROFILE, loglog=False, show=True)
        
        if DO_FIT_VTAN_PROFILE:
            print("Fitting tangential velocity profile...")
            from src.h2s_profile_vel import fit_vtheta_profile
            if DO_FLUX_CUT:
                FIT_VTAN_PROFILE = os.path.join(RESULTS_DIR, f"vtan_profile_fit_{FLUX_MIN}.png")
                FIT_VTAN_PROFILE_PARAMS = os.path.join(RESULTS_DIR, f"vtan_profile_fit_params_{FLUX_MIN}.txt")
                FIT_VTAN_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, f"vtan_profile_fit_shuffled_{FLUX_MIN}.png")
                FIT_VTAN_PROFILE_PARAMS_SHUFFLED = os.path.join(RESULTS_DIR, f"vtan_profile_fit_params_shuffled_{FLUX_MIN}.txt")
            else:
                FIT_VTAN_PROFILE = os.path.join(RESULTS_DIR, "vtan_profile_fit.png")
                FIT_VTAN_PROFILE_PARAMS = os.path.join(RESULTS_DIR, "vtan_profile_fit_params.txt")
                FIT_VTAN_PROFILE_SHUFFLED = os.path.join(RESULTS_DIR, "vtan_profile_fit_shuffled.png")
                FIT_VTAN_PROFILE_PARAMS_SHUFFLED = os.path.join(RESULTS_DIR, "vtan_profile_fit_params_shuffled.txt")
            fit_vtheta_profile(vtheta_profile_file=VTAN_PROFILE, plot=True, output_png=FIT_VTAN_PROFILE, output_params_file= FIT_VTAN_PROFILE_PARAMS,
                                loglog=False, manual_params= MANUAL_PARAMS_VTAN)
            if DO_SHUFFLING:
                fit_vtheta_profile(vtheta_profile_file=VTAN_PROFILE_SHUFFLED, plot=True, output_png=FIT_VTAN_PROFILE_SHUFFLED,
                                   loglog=False, manual_params=MANUAL_PARAMS_VTAN, output_params_file=FIT_VTAN_PROFILE_PARAMS_SHUFFLED)

    # ================= Kaiser Analysis =================

    if DO_KAISER_ANALYSIS:
        print("Performing Kaiser analysis...")
        from src.h2s_plots import plot_kaiser_comparison
        from src.h2s_io import kaiser_factor

        fkaiser = kaiser_factor(omega_m=OMEGA_M, b=BIAS, gamma=GAMMA)
        print(f"Kaiser factor: {fkaiser:.4f}")

        if DO_PLOTS:
            print("Plotting Kaiser comparison...")

            if DO_FLUX_CUT:
                REAL_FILE = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_{FLUX_MIN}.txt")
                REDSHIFT_FILE = os.path.join(RESULTS_DIR, f"2PCF_Corrfunc_redshift_{FLUX_MIN}.txt")
                KAISER_PNG = os.path.join(RESULTS_DIR, f"kaiser_comparison_{FLUX_MIN}.png")
            else:
                REAL_FILE = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_.txt")
                REDSHIFT_FILE = os.path.join(RESULTS_DIR, "2PCF_Corrfunc_redshift.txt")
                KAISER_PNG = os.path.join(RESULTS_DIR, "kaiser_comparison.png")

        
            plot_kaiser_comparison(real_file=REAL_FILE, redshift_file=REDSHIFT_FILE, omega_m=OMEGA_M, xlim = None, ylim=None,
                                    b=BIAS, gamma=GAMMA, output_png=KAISER_PNG, show=True, loglog=True)
        
        if DO_KAISER_RATIO:
            print("Calculating Kaiser ratio...")
            from src.h2s_plots import plot_kaiser_ratio
            if DO_FLUX_CUT:
                KAISER_RATIO = os.path.join(RESULTS_DIR, f"kaiser_ratio_{FLUX_MIN}.png")
            else:
                KAISER_RATIO = os.path.join(RESULTS_DIR, "kaiser_ratio.png")
            plot_kaiser_ratio(real_file=REAL_FILE, redshift_file=REDSHIFT_FILE, output_png=KAISER_RATIO,
                                 omega_m=OMEGA_M, bias=BIAS, gamma=GAMMA, show=True, xlim=XLIM_KAISER, ylim=YLIM_KAISER)

        
    # ============ Write master output file: h2s_output.h5 ============

    # Path for the master output file
    MASTER_OUTPUT = os.path.join(RESULTS_DIR, "h2s_output.h5")

    # Load per-bin data from conformity, radial profile, VR, and VTAN files
    with h5py.File(CONFORMITY, "r") as f_conf, \
         h5py.File(RADIAL_PROFILE, "r") as f_radial, \
         h5py.File(VR_PROFILE, "r") as f_vr, \
         h5py.File(VTAN_PROFILE, "r") as f_vtan:

       # ---- GLOBAL CONSTANTS/HEADER ----
       # Conformity global parameters
        k1_global = f_conf["data/global/k1_global"][()]
        k2_global = f_conf["data/global/k2_global"][()]
    
        # Cosmology & run config
        z_snap = Z_SNAP
        omega_m = OMEGA_M
        omega_l = OMEGA_L
        h_param = h
        boxsize = BOXSIZE

        # Fit parameters: must have been assigned after running the fits!
        
        params_vtan = np.loadtxt(FIT_VTAN_PROFILE_PARAMS, delimiter=",", skiprows=1).flatten()
        params_vr = np.loadtxt(FIT_VR_PROFILE_PARAMS, delimiter=",", skiprows=1).flatten()
        alpha, beta, r0, N0, kappa = params_radial
        vtan_y0, vtan_alpha, vtan_beta, vtan_kappa = params_vtan
        (vr_A1, vr_mu1, vr_sigma1, 
         vr_A2, vr_mu2, vr_sigma2, 
         vr_A3, vr_mu3, vr_sigma3) = params_vr

        # ---- DATA BLOCKS: Load per-bin data from each file ----
        # Conformity
        conf_bins = f_conf["data/bins"]
        M_min = conf_bins["M_min_bin"][:]
        M_max = conf_bins["M_max_bin"][:]
        N_halo = conf_bins["N_Halos"][:]
        k1 = conf_bins["k1"][:]
        k2 = conf_bins["k2"][:]
        Nsat = conf_bins["N_S"][:]    # satellites per bin
        Ncen = conf_bins["N_C"][:]    # centrals per bin

        # Radial profile
        r_centers = f_radial['radial_bins'][:]
        Nsat_r = f_radial['counts'][:]
        dr = np.diff(r_centers)
        edges = np.concatenate(([r_centers[0] - dr[0]/2], r_centers[:-1] + dr/2, [r_centers[-1] + dr[-1]/2]))
        r_min = edges[:-1]
        r_max = edges[1:]

        # VR profile
        vr_centers = f_vr["velocity_bins"][:]
        dvr = np.diff(vr_centers)
        vr_edges = np.concatenate(([vr_centers[0] - dvr[0]/2], vr_centers[:-1] + dvr/2, [vr_centers[-1] + dvr[-1]/2]))
        Nsat_vr = f_vr["density"][:]
        vr_min = vr_edges[:-1]
        vr_max = vr_edges[1:]

        # VTAN profile
        vtan_centers = f_vtan["velocity_bins"][:]
        dvtan = np.diff(vtan_centers)
        vtan_edges = np.concatenate(([vtan_centers[0] - dvtan[0]/2], vtan_centers[:-1] + dvtan/2, [vtan_centers[-1] + dvtan[-1]/2]))
        Nsat_vtan = f_vtan["density"][:]
        vtan_min = vtan_edges[:-1]
        vtan_max = vtan_edges[1:]

    # --------- Write master output HDF5 ---------
    with h5py.File(MASTER_OUTPUT, "w") as f:
        # HEADER group with attributes
        header = f.create_group("header")
        header.attrs["K1_global"] = k1_global
        header.attrs["K2_global"] = k2_global
        header.attrs["z_snap"] = z_snap
        header.attrs["omega_m"] = omega_m
        header.attrs["omega_l"] = omega_l
        header.attrs["h"] = h_param
        header.attrs["boxsize"] = boxsize
        header.attrs["alpha"] = alpha
        header.attrs["beta"] = beta
        header.attrs["kappa"] = kappa
        header.attrs["N0"] = N0
        header.attrs["r0"] = r0
        header.attrs["vtan_y0"] = vtan_y0
        header.attrs["vtan_alpha"] = vtan_alpha
        header.attrs["vtan_beta"] = vtan_beta
        header.attrs["vtan_kappa"] = vtan_kappa
        header.attrs["vr_A1"] = vr_A1
        header.attrs["vr_mu1"] = vr_mu1
        header.attrs["vr_sigma1"] = vr_sigma1
        header.attrs["vr_A2"] = vr_A2
        header.attrs["vr_mu2"] = vr_mu2
        header.attrs["vr_sigma2"] = vr_sigma2
        header.attrs["vr_A3"] = vr_A3
        header.attrs["vr_mu3"] = vr_mu3
        header.attrs["vr_sigma3"] = vr_sigma3

        # DATA group with all bin-wise arrays
        data = f.create_group("data")
        # Conformity
        data.create_dataset("M_min", data=M_min)
        data.create_dataset("M_max", data=M_max)
        data.create_dataset("N_halo", data=N_halo)
        data.create_dataset("k1", data=k1)
        data.create_dataset("k2", data=k2)
        data.create_dataset("Nsat", data=Nsat)
        data.create_dataset("Ncen", data=Ncen)
        # Radial profile
        data.create_dataset("r_min", data=r_min)
        data.create_dataset("r_max", data=r_max)
        data.create_dataset("Nsat_r", data=Nsat_r)
        # VR profile
        data.create_dataset("vr_min", data=vr_min)
        data.create_dataset("vr_max", data=vr_max)
        data.create_dataset("Nsat_vr", data=Nsat_vr)
        # VTAN profile
        data.create_dataset("vtan_min", data=vtan_min)
        data.create_dataset("vtan_max", data=vtan_max)
        data.create_dataset("Nsat_vtan", data=Nsat_vtan)

    print(f"\n[INFO] Master output file written: {MASTER_OUTPUT}\n")
    if DO_SHUFFLING:
        # Path for the master output file
        MASTER_OUTPUT_SHUFFLED = os.path.join(RESULTS_DIR, "h2s_output_shuffled.h5")

        # Load per-bin data from conformity, radial profile, VR, and VTAN files
        with h5py.File(CONFORMITY_SHUFFLED, "r") as f_conf, \
             h5py.File(RADIAL_PROFILE_SHUFFLED, "r") as f_radial, \
             h5py.File(VR_PROFILE_SHUFFLED, "r") as f_vr, \
             h5py.File(VTAN_PROFILE_SHUFFLED, "r") as f_vtan:

           # ---- GLOBAL CONSTANTS/HEADER ----
           # Conformity global parameters
            k1_global = f_conf["data/global/k1_global"][()]
            k2_global = f_conf["data/global/k2_global"][()]

            # Cosmology & run config
            z_snap = Z_SNAP
            omega_m = OMEGA_M
            omega_l = OMEGA_L
            h_param = h
            boxsize = BOXSIZE

            # Fit parameters: must have been assigned after running the fits!

            params_vtan_shuffled = np.loadtxt(FIT_VTAN_PROFILE_PARAMS_SHUFFLED, delimiter=",", skiprows=1).flatten()
            params_vr_shuffled = np.loadtxt(FIT_VR_PROFILE_PARAMS_SHUFFLED, delimiter=",", skiprows=1).flatten()
            alpha, beta, r0, N0, kappa = params_radial_shuffled
            vtan_y0, vtan_alpha, vtan_beta, vtan_kappa = params_vtan_shuffled
            (vr_A1, vr_mu1, vr_sigma1, 
             vr_A2, vr_mu2, vr_sigma2, 
             vr_A3, vr_mu3, vr_sigma3) = params_vr_shuffled

            # ---- DATA BLOCKS: Load per-bin data from each file ----
            # Conformity
            conf_bins = f_conf["data/bins"]
            M_min = conf_bins["M_min_bin"][:]
            M_max = conf_bins["M_max_bin"][:]
            N_halo = conf_bins["N_Halos"][:]
            k1 = conf_bins["k1"][:]
            k2 = conf_bins["k2"][:]
            Nsat = conf_bins["N_S"][:]    # satellites per bin
            Ncen = conf_bins["N_C"][:]    # centrals per bin

            # Radial profile
            r_centers = f_radial['radial_bins'][:]
            Nsat_r = f_radial['counts'][:]
            dr = np.diff(r_centers)
            edges = np.concatenate(([r_centers[0] - dr[0]/2], r_centers[:-1] + dr/2, [r_centers[-1] + dr[-1]/2]))
            r_min = edges[:-1]
            r_max = edges[1:]

            # VR profile
            vr_centers = f_vr["velocity_bins"][:]
            dvr = np.diff(vr_centers)
            vr_edges = np.concatenate(([vr_centers[0] - dvr[0]/2], vr_centers[:-1] + dvr/2, [vr_centers[-1] + dvr[-1]/2]))
            Nsat_vr = f_vr["density"][:]
            vr_min = vr_edges[:-1]
            vr_max = vr_edges[1:]

            # VTAN profile
            vtan_centers = f_vtan["velocity_bins"][:]
            dvtan = np.diff(vtan_centers)
            vtan_edges = np.concatenate(([vtan_centers[0] - dvtan[0]/2], vtan_centers[:-1] + dvtan/2, [vtan_centers[-1] + dvtan[-1]/2]))
            Nsat_vtan = f_vtan["density"][:]
            vtan_min = vtan_edges[:-1]
            vtan_max = vtan_edges[1:]

        # --------- Write master output HDF5 ---------
        with h5py.File(MASTER_OUTPUT_SHUFFLED, "w") as f:
            # HEADER group with attributes
            header = f.create_group("header")
            header.attrs["K1_global"] = k1_global
            header.attrs["K2_global"] = k2_global
            header.attrs["z_snap"] = z_snap
            header.attrs["omega_m"] = omega_m
            header.attrs["omega_l"] = omega_l
            header.attrs["h"] = h_param
            header.attrs["boxsize"] = boxsize
            header.attrs["alpha"] = alpha
            header.attrs["beta"] = beta
            header.attrs["kappa"] = kappa
            header.attrs["N0"] = N0
            header.attrs["r0"] = r0
            header.attrs["vtan_y0"] = vtan_y0
            header.attrs["vtan_alpha"] = vtan_alpha
            header.attrs["vtan_beta"] = vtan_beta
            header.attrs["vtan_kappa"] = vtan_kappa
            header.attrs["vr_A1"] = vr_A1
            header.attrs["vr_mu1"] = vr_mu1
            header.attrs["vr_sigma1"] = vr_sigma1
            header.attrs["vr_A2"] = vr_A2
            header.attrs["vr_mu2"] = vr_mu2
            header.attrs["vr_sigma2"] = vr_sigma2
            header.attrs["vr_A3"] = vr_A3
            header.attrs["vr_mu3"] = vr_mu3
            header.attrs["vr_sigma3"] = vr_sigma3

            # DATA group with all bin-wise arrays
            data = f.create_group("data")
            # Conformity
            data.create_dataset("M_min", data=M_min)
            data.create_dataset("M_max", data=M_max)
            data.create_dataset("N_halo", data=N_halo)
            data.create_dataset("k1", data=k1)
            data.create_dataset("k2", data=k2)
            data.create_dataset("Nsat", data=Nsat)
            data.create_dataset("Ncen", data=Ncen)
            # Radial profile
            data.create_dataset("r_min", data=r_min)
            data.create_dataset("r_max", data=r_max)
            data.create_dataset("Nsat_r", data=Nsat_r)
            # VR profile
            data.create_dataset("vr_min", data=vr_min)
            data.create_dataset("vr_max", data=vr_max)
            data.create_dataset("Nsat_vr", data=Nsat_vr)
            # VTAN profile
            data.create_dataset("vtan_min", data=vtan_min)
            data.create_dataset("vtan_max", data=vtan_max)
            data.create_dataset("Nsat_vtan", data=Nsat_vtan)

        print(f"\n[INFO] Master output file for shuffled catalog written: {MASTER_OUTPUT_SHUFFLED}\n")

if __name__ == "__main__":
    main()
