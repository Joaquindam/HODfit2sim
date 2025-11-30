import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from Corrfunc.theory.xi import xi

def export_positions(
    infile,
    outfile,
    xkey='Xpos',
    ykey='Ypos',
    zkey='Zpos',
    verbose=True
):
    """
    Exports galaxy positions (X, Y, Z) from an HDF5 file into a plain .txt file
    compatible with CUTE (no header, three columns: x y z).

    Parameters:
    -----------
    infile : str
        Path to the input HDF5 file with filtered galaxies.
    outfile : str
        Path to the output .txt file to generate.
    xkey, ykey, zkey : str
        Names of the datasets for positions in the HDF5 file.
    verbose : bool
        If True, prints progress messages.

    Returns:
    --------
    outfile : str
        Path to the saved output file.
    """

    if verbose:
        print(f"Reading positions from: {infile}")

    with h5py.File(infile, "r") as f:
        for key in [xkey, ykey, zkey]:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {infile}")
        
        x = f[xkey][:]
        y = f[ykey][:]
        z = f[zkey][:]

    positions = np.vstack((x, y, z)).T  # shape: (N, 3)

    if verbose:
        print(f"Saving {positions.shape[0]} positions to: {outfile}")

    np.savetxt(outfile, positions, fmt="%.6e")

    if verbose:
        print("Export complete.")

    return outfile

def compute_correlation_corrfunc(
    positions_file,
    output_file,
    boxsize,
    rmin,
    rmax,
    n_bins,
    log_binning=True,
    n_threads=4,
    verbose=True
):
    """
    Computes the two-point correlation function ξ(r) using Corrfunc,
    and returns analytical Poisson errors per bin.

    Parameters:
    -----------
    positions_file : str
        Path to a .txt file with three columns (x, y, z) of galaxy positions.
    output_file : str
        Path where the output .txt (three columns: r_center, xi, error) will be saved.
    boxsize : float
        Size of the simulation box (Mpc/h).
    rmin : float
        Minimum separation (Mpc/h) to consider.
    rmax : float
        Maximum separation (Mpc/h) to consider.
    n_bins : int
        Number of logarithmic bins between rmin and rmax.
    n_threads : int, optional
        Number of threads to use in Corrfunc (default: 4).
    verbose : bool, optional
        If True, prints progress messages (default: True).

    Returns:
    --------
    output_file : str
        Path to the saved .txt file with columns [r_center, xi(r), error(r)].
    r_centers, xi_vals, errors : arrays
        Arrays with the bin centers, xi(r), and error per bin.
    """
    if verbose:
        print(f"Loading positions from: {positions_file}")

    # Load positions (assumed to be a plain text file with 3 columns)
    data = np.loadtxt(positions_file)
    x_data, y_data, z_data = data[:, 0], data[:, 1], data[:, 2]

    if verbose:
        print("Generating bins...")
        if log_binning == True:
            rbins = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)
            r_centers = 0.5 * (rbins[:-1] + rbins[1:])
        else:
            rbins = np.linspace(rmin, rmax, n_bins + 1)
            r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    if verbose:
        print("Computing ξ(r) with Corrfunc...")
    results = xi(
        boxsize=boxsize,
        nthreads=n_threads,
        binfile=rbins,
        X=x_data, Y=y_data, Z=z_data
    )

    # Extract xi values and npairs (pair counts)
    xi_vals = np.array([b['xi'] for b in results])
    npairs = np.array([b['npairs'] for b in results])

    # Analytical Poisson errors: sigma_xi = (1 + xi) / sqrt(Npairs)
    # Avoid division by zero:
    errors = np.zeros_like(xi_vals)
    mask = npairs > 0
    errors[mask] = (1.0 + xi_vals[mask]) / np.sqrt(npairs[mask])
    errors[~mask] = 0.0

    # Stack r_centers, xi values, and errors into three columns
    output_data = np.column_stack((r_centers, xi_vals, errors))

    if verbose:
        print(f"Saving correlation to: {output_file}")
    header = "#r_center[Mpc/h], #xi(r), #err_analytical"
    np.savetxt(output_file, output_data, delimiter=",", header=header, comments='')

    if verbose:
        print("Correlation computation complete.")
    return output_file, r_centers, xi_vals, errors

def export_positions_redshift_space(
    infile,
    outfile,
    z_snap=1.321,
    Omega_m=0.3089,
    Omega_L=0.6911,
    h=0.6774,
    xkey='Xpos',
    ykey='Ypos',
    zkey='Zpos',
    xvel_key='Xvel',
    yvel_key='Yvel',
    zvel_key='Zvel',
    los_axis='z',
    verbose=True
):
    """
    Export galaxy positions in redshift-space (peculiar velocity correction along 'los_axis').
    Uses cosmological parameters from Planck 2015 (default), redshift z_snap.

    Parameters
    ----------
    infile : str
        Path to input HDF5 file (filtered galaxies).
    outfile : str
        Output .txt file for redshift-space positions.
    z_snap : float
        Snapshot redshift (default 1.321).
    Omega_m : float
        Matter density parameter (default 0.3089).
    Omega_L : float
        Lambda density parameter (default 0.6911).
    h : float
        Little h (default 0.6774), so H0 = 100*h km/s/Mpc.
    xkey, ykey, zkey : str
        Dataset keys for positions.
    xvel_key, yvel_key, zvel_key : str
        Dataset keys for velocities (km/s).
    los_axis : str
        Axis to use as line of sight ('x', 'y', or 'z').
    verbose : bool
        Print status if True.

    Returns
    -------
    outfile : str
        Path to output .txt file (comma separated, no header).
    """
    import h5py
    import numpy as np
    import os

    # Hubble parameter at this redshift
    H0 = 100.0 * h  # km/s/Mpc
    Hz = H0 * np.sqrt(Omega_m * (1 + z_snap) ** 3 + Omega_L)
    a = 1.0 / (1.0 + z_snap)

    if verbose:
        print(f"Cosmology: Omega_m = {Omega_m}, Omega_L = {Omega_L}, h = {h:.4f}")
        print(f"z = {z_snap:.3f}, a = {a:.4f}, H(z) = {Hz:.4f} km/s/Mpc")

    with h5py.File(infile, "r") as f:
        x = f[xkey][:]
        y = f[ykey][:]
        z = f[zkey][:]
        vx = f[xvel_key][:]
        vy = f[yvel_key][:]
        vz = f[zvel_key][:]

    # Shift line-of-sight positions
    if los_axis == 'x':
        s_los = x + (vx) / (Hz)
        positions = np.vstack([s_los, y, z]).T
    elif los_axis == 'y':
        s_los = y + (vy) / (Hz)
        positions = np.vstack([x, s_los, z]).T
    else:  # 'z'
        s_los = z + (vz) / (Hz)
        positions = np.vstack([x, y, s_los]).T

    # Ensure output directory exists
    outdir = os.path.dirname(outfile)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    np.savetxt(outfile, positions, fmt="%.6e", delimiter=" ")

    if verbose:
        print(f"Redshift-space positions exported to {outfile}")

    return outfile
