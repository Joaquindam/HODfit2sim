import numpy as np
import h5py
import os
from src.h2s_io import get_mask, check_file
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import src.h2s_const as const

def boundary_correction(xin, box, groups=False):
    """
    Applies periodic boundary conditions to an 1D array of coordinates.
    """
    xout = np.copy(xin)

    if groups:
        lbox2 = box / 2.0
        xout[xin < -lbox2] += box
        xout[xin >= lbox2] -= box
    else:
        xout[xin < 0] += box
        xout[xin >= box] -= box

    return xout

def get_diffpos(x1, y1, z1, x2, y2, z2, box=None):
    """
    Calculates the difference in positions between two sets of coordinates,
    """
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    if box is not None:
        dx = boundary_correction(dx, box, groups=True)
        dy = boundary_correction(dy, box, groups=True)
        dz = boundary_correction(dz, box, groups=True)

    return dx, dy, dz

def get_r(x1, y1, z1, x2, y2, z2, box=None):
    """
    Calculates euclidean distance between two points, asuming periodic boundaries if `box`.
    """
    dx, dy, dz = get_diffpos(x1, y1, z1, x2, y2, z2, box)
    return np.sqrt(dx**2 + dy**2 + dz**2)

def fit_radial_profile(
    profile_file,
    output_params_file=None,
    initial_guess=None,
    bounds=None,
    verbose=True
):
    """
    Fits an analytic function to the radial profile stored in an HDF5 file,
    taking into account Poisson errors (sqrt(counts)).

    Analytic form:
        N(r) = N0 * (r/r0)^alpha * (1 + (r/r0)^beta)^kappa

    Parameters
    ----------
    profile_file : str
        Path to the HDF5 file containing:
            - radial_bins : bin centers
            - counts : counts per bin
    output_params_file : str or None, optional
        Path to save fitted parameters (CSV). If None, no file is saved.
    initial_guess : list[float] or tuple, optional
        Initial guess [alpha, beta, r0, N0, kappa].
    bounds : tuple, optional
        Bounds as ([lower], [upper]). If None, no bounds.
    verbose : bool, optional
        Print details if True.

    Returns
    -------
    popt : ndarray
        Best-fit parameters.
    pcov : ndarray
        Covariance matrix.
    """

    with h5py.File(profile_file, "r") as f:
        r = f["radial_bins"][:]
        counts = f["counts"][:]

    # Keep only bins with positive counts
    positive_mask = counts > 0
    r_fit = r[positive_mask]
    counts_fit = counts[positive_mask]

    if r_fit.size == 0:
        raise RuntimeError("No positive counts; cannot fit.")

    # Poisson errors
    sigma_counts = np.sqrt(counts_fit)

    # Analytic function
    def analytic_counts(r_vals, alpha, beta, r0, N0, kappa):
        return N0 * (r_vals / r0)**alpha * (1.0 + (r_vals / r0)**beta)**kappa

    # Default initial guess
    if initial_guess is None:
        p0 = [1.23, 3.19, 0.34, 3928.273, -2.1]
    else:
        p0 = initial_guess

    # Default bounds
    if bounds is None:
        bounds = (-np.inf, np.inf)

    if verbose:
        print("Fitting with Poisson errors (sigma=sqrt(counts))...")
        print(f"Initial guess: {p0}")
        print(f"Bounds: {bounds}")

    # Perform fit with Poisson errors
    popt, pcov = curve_fit(
        analytic_counts,
        r_fit,
        counts_fit,
        p0=p0,
        sigma=sigma_counts,  
        absolute_sigma=True,  
        bounds=bounds,
        maxfev=10000
    )

    if verbose:
        print("Fit complete.")
        param_names = ["alpha", "beta", "r0", "N0", "kappa"]
        for name, val in zip(param_names, popt):
            print(f"{name} = {val:.4f}")

    if output_params_file is not None:
        header = "alpha,beta,r0,N0,kappa"
        np.savetxt(output_params_file, popt.reshape(1, -1),
                   delimiter=",", header=header, comments="")
        if verbose:
            print(f"Parameters saved to: {output_params_file}")

    return popt, pcov

def fit_radial_profile_log(
    profile_file,
    output_params_file=None,
    initial_guess=None,
    bounds=None,
    verbose=True
):
    """
    Fit analytic model to radial profile using log(counts) as target (log-log fit).
    """
    # Load data
    with h5py.File(profile_file, "r") as f:
        r = f["radial_bins"][:]
        counts = f["counts"][:]
    mask = (counts > 0)
    r_fit = r[mask]
    counts_fit = counts[mask]

    # Analytic model (must be positive!)
    def analytic_counts(r_vals, alpha, beta, r0, N0, kappa):
        return N0 * (r_vals / r0) ** alpha * (1.0 + (r_vals / r0) ** beta) ** kappa

    # Fit to log(counts)
    def log_analytic_counts(r_vals, alpha, beta, r0, N0, kappa):
        model = analytic_counts(r_vals, alpha, beta, r0, N0, kappa)
        return np.log(model)

    ydata = np.log(counts_fit)

    # Initial guess 
    if initial_guess is None:
        alpha0 = 1.23
        beta0 = 3.19
        r0_0 = 0.34
        N0_0 = 3928.273
        kappa0 = -2.1
        p0 = [alpha0, beta0, r0_0, N0_0, kappa0]
    else:
        p0 = list(initial_guess)

    # Bounds
    if bounds is None:
        lower_bounds = [-np.inf]*5
        upper_bounds = [ np.inf]*5
    else:
        lower_bounds, upper_bounds = bounds

    # Fit
    popt, pcov = curve_fit(
        log_analytic_counts, r_fit, ydata,
        p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=100000
    )

    if verbose:
        print("Fit complete.")
        print(f"Fitted parameters (counts-based):\n"
              f"  alpha = {popt[0]:.4f}\n"
              f"  beta  = {popt[1]:.4f}\n"
              f"  r0    = {popt[2]:.4f}\n"
              f"  N0    = {popt[3]:.4e}\n"
              f"  kappa = {popt[4]:.4f}")

    if output_params_file is not None:
        header = "alpha, beta, r0, N0, kappa"
        np.savetxt(output_params_file, np.array([popt]), delimiter=",", header=header, comments="")

    return popt, pcov

def compute_radial_profile(
    galaxy_file,
    halo_file,
    output_file,
    boxsize=1000.0,
    bins=np.linspace(0, 1.5, 151),
    halo_format="txt",
    halo_id_key=1,
    halo_pid_key=13,
    halo_x_key=3,
    halo_y_key=4,
    halo_z_key=5,
    galaxy_id_key="MainHaloID",
    galaxy_host_key="HostHaloID",
    galaxy_x_key="Xpos",
    galaxy_y_key="Ypos",
    galaxy_z_key="Zpos",
    testing=False,
    rockstar_format=False,
    verbose=True
):
    """
    Computes the radial profile of satellite galaxies using true halo positions (parent halos only).
    Satellites are matched to their parent halos (pid==-1) by ID.

    Parameters
    ----------
    galaxy_file : str
        Path to the HDF5 file containing galaxy data.
    halo_file : str
        Path to the halos catalog (TXT or HDF5, currently assumes TXT).
    output_file : str
        HDF5 file where the radial profile will be saved.
    boxsize : float
        Simulation box size.
    bins : array_like
        Bins for radial profile.
    halo_format : str
        Format of the halo catalog ('txt' or 'hdf5').
    halo_id_key, halo_pid_key, halo_x_key, halo_y_key, halo_z_key : int or str
        Columns/fields for halo ID, parent ID, and positions.
    galaxy_*_key : str
        Names of the datasets for galaxy IDs and positions.
    verbose : bool
        If True, print progress.
    """
    check_file(galaxy_file, verbose=True)
    check_file(halo_file, verbose=True)

    # 1. Load parent halos and build halo_pos_dict
    if halo_format == "txt":
        halos = np.loadtxt(halo_file)
        # filter parent halos (pid == -1)
        if testing:
            halos = halos[:const.testlimit_halos]
            parent_mask = halos[:, halo_pid_key] == -1
        else:
            parent_mask = halos[:, halo_pid_key] == -1
        parent_ids = halos[parent_mask, halo_id_key]
        parent_x = halos[parent_mask, halo_x_key]
        parent_y = halos[parent_mask, halo_y_key]
        parent_z = halos[parent_mask, halo_z_key]
    else:
        raise NotImplementedError("Only txt halos implemented for now.")

    halo_pos_dict = {hid: (x, y, z) for hid, x, y, z in zip(parent_ids, parent_x, parent_y, parent_z)}

    if verbose:
        print(f"Loaded {len(halo_pos_dict)} parent halos from {halo_file}.")

    # 2. Load galaxies (assumes HDF5)
    with h5py.File(galaxy_file, "r") as f:
        host_id = f[galaxy_host_key][:]
        main_id = f[galaxy_id_key][:]
        xpos = f[galaxy_x_key][:]
        ypos = f[galaxy_y_key][:]
        zpos = f[galaxy_z_key][:]

    # 3. Identify satellites
    is_sat = host_id != main_id
    xs = xpos[is_sat]
    ys = ypos[is_sat]
    zs = zpos[is_sat]
    halo_ids = main_id[is_sat]  # For each satellite, its parent halo ID

    # 4. Match satellite galaxies to their halo positions
    xc, yc, zc = [], [], []
    missing = 0

    for hid in halo_ids:
        if hid in halo_pos_dict:
            xh, yh, zh = halo_pos_dict[hid]
        else:
            xh, yh, zh = np.nan, np.nan, np.nan
            missing += 1
        xc.append(xh)
        yc.append(yh)
        zc.append(zh)

    xc, yc, zc = np.array(xc), np.array(yc), np.array(zc)

    if missing > 0:
        print(f"  {missing} satellites have no matching parent halo. Distances set to NaN.")

    # 5. Compute distances
    distances = get_r(xs, ys, zs, xc, yc, zc, box=boxsize)
    valid = np.isfinite(distances) & (distances >= 0)
    distances = distances[valid]

    # 6. Build histogram
    hist, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dbins = bin_edges[-1] - bin_edges[-2]

    if verbose:
        print(f" Radial profile: {len(hist)} computed bins.")
        print(f" Results written to {output_file}")

    with h5py.File(output_file, "w") as fout:
        fout.create_dataset("radial_bins", data=bin_centers)
        fout.create_dataset("counts", data=hist)
        fout.attrs["boxsize"] = boxsize
        fout.attrs["n_bins"] = len(hist)
        fout.attrs["n_satellites"] = len(distances)

    if verbose:
        print(" Radial profile stored successfully.")

def compute_radial_profile_shuffled(
    galaxy_file,
    halo_file,
    output_file,
    boxsize=1000.0,
    bins=np.linspace(0, 1.5, 151),
    halo_format="txt",
    halo_id_key=1,
    halo_pid_key=13,
    halo_x_key=3,
    halo_y_key=4,
    halo_z_key=5,
    galaxy_id_key="MainHaloID",
    galaxy_is_central_key="is_central",
    galaxy_x_key="Xpos",
    galaxy_y_key="Ypos",
    galaxy_z_key="Zpos",
    testing=False,
    rockstar_format=False,
    verbose=True
):
    """
    Computes the radial profile of satellite galaxies using true halo positions (parent halos only).
    Satellites are matched to their parent halos (pid==-1) by ID.

    Parameters
    ----------
    galaxy_file : str
        Path to the HDF5 file containing galaxy data.
    halo_file : str
        Path to the halos catalog (TXT or HDF5, currently assumes TXT).
    output_file : str
        HDF5 file where the radial profile will be saved.
    boxsize : float
        Simulation box size.
    bins : array_like
        Bins for radial profile.
    halo_format : str
        Format of the halo catalog ('txt' or 'hdf5').
    halo_id_key, halo_pid_key, halo_x_key, halo_y_key, halo_z_key : int or str
        Columns/fields for halo ID, parent ID, and positions.
    galaxy_*_key : str
        Names of the datasets for galaxy IDs and positions.
    verbose : bool
        If True, print progress.
    """
    check_file(galaxy_file, verbose=True)
    check_file(halo_file, verbose=True)

    # 1. Load parent halos and build halo_pos_dict
    if halo_format == "txt":
        halos = np.loadtxt(halo_file)
        # filter parent halos (pid == -1)
        if testing:
            halos = halos[:const.testlimit_halos]
            parent_mask = halos[:, halo_pid_key] == -1
        else:
            parent_mask = halos[:, halo_pid_key] == -1
        parent_ids = halos[parent_mask, halo_id_key]
        parent_x = halos[parent_mask, halo_x_key]
        parent_y = halos[parent_mask, halo_y_key]
        parent_z = halos[parent_mask, halo_z_key]
    else:
        raise NotImplementedError("Only txt halos implemented for now.")

    halo_pos_dict = {hid: (x, y, z) for hid, x, y, z in zip(parent_ids, parent_x, parent_y, parent_z)}

    if verbose:
        print(f"Loaded {len(halo_pos_dict)} parent halos from {halo_file}.")

    # 2. Load galaxies (assumes HDF5)
    with h5py.File(galaxy_file, "r") as f:
        is_central = f[galaxy_is_central_key][:]
        main_id = f[galaxy_id_key][:]
        xpos = f[galaxy_x_key][:]
        ypos = f[galaxy_y_key][:]
        zpos = f[galaxy_z_key][:]

    # 3. Identify satellites
    is_sat = (is_central == 0)
    xs = xpos[is_sat]
    ys = ypos[is_sat]
    zs = zpos[is_sat]
    halo_ids = main_id[is_sat]  # For each satellite, its parent halo ID

    # 4. Match satellite galaxies to their halo positions
    xc, yc, zc = [], [], []
    missing = 0

    for hid in halo_ids:
        if hid in halo_pos_dict:
            xh, yh, zh = halo_pos_dict[hid]
        else:
            xh, yh, zh = np.nan, np.nan, np.nan
            missing += 1
        xc.append(xh)
        yc.append(yh)
        zc.append(zh)

    xc, yc, zc = np.array(xc), np.array(yc), np.array(zc)

    if missing > 0:
        print(f"  {missing} satellites have no matching parent halo. Distances set to NaN.")

    # 5. Compute distances
    distances = get_r(xs, ys, zs, xc, yc, zc, box=boxsize)
    valid = np.isfinite(distances) & (distances >= 0)
    distances = distances[valid]

    # 6. Build histogram
    hist, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dbins = bin_edges[-1] - bin_edges[-2]

    if verbose:
        print(f" Radial profile: {len(hist)} computed bins.")
        print(f" Results written to {output_file}")

    with h5py.File(output_file, "w") as fout:
        fout.create_dataset("radial_bins", data=bin_centers)
        fout.create_dataset("counts", data=hist)
        fout.attrs["boxsize"] = boxsize
        fout.attrs["n_bins"] = len(hist)
        fout.attrs["n_satellites"] = len(distances)

    if verbose:
        print(" Radial profile stored successfully.")
