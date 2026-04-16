import numpy as np
import h5py
import os
from src.h2s_io import get_mask, check_file, get_central
from src.h2s_profile_r import get_r, get_diffpos
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import src.h2s_const as const

def get_vr(x1,y1,z1,x2,y2,z2,vx1,vy1,vz1,vx2,vy2,vz2,box=None):
    """
    Calculate the relative radial velocity of an object
    
    Parameters
    ----------
    x1,y1,z1 : (array of) floats
       Coordinattes of object 1 (halo or central). Array for simulations.
    x2,y2,z2 : (array of) floats
       Coordinattes of object 2 (subhalo or satellite). Array for simulations.
    vx1,vy1,vz1 : (array of) floats
       Velocity of object 1 (halo or central). Array for simulations.
    vx2,vy2,vz2 : (array of) floats
       Velocity of object 2 (subhalo or satellite). Array for simulations.
    box : float
       If a simulation, side of the simulation box

    Returns
    -------
    vr : float
       Radial relative velocity
    """

    r = get_r(x1,y1,z1,x2,y2,z2,box)

    dx,dy,dz = get_diffpos(x1,y1,z1,x2,y2,z2,box)
    dvx,dvy,dvz = get_diffpos(vx1,vy1,vz1,vx2,vy2,vz2)

    if hasattr(r, "__len__"):
        vr = np.empty(len(r)); vr[:] = np.nan    
        ind = np.where(r>0)
        if (np.shape(ind)[1]>0):
            vr[ind] = (dx[ind]*dvx[ind] + dy[ind]*dvy[ind] + dz[ind]*dvz[ind])/r[ind]
    else:
        vr = np.nan
        if (abs(r)>0): vr = (dx*dvx + dy*dvy + dz*dvz)/r

    return vr


def get_vtheta(x1,y1,z1,x2,y2,z2,vx1,vy1,vz1,vx2,vy2,vz2,box=None):
    """
    Calculate the relative tangential velocity in the x-y plane (v theta).
    
    Parameters
    ----------
    x1,y1,z1 : (array of) floats
       Coordinattes of object 1 (halo or central). Array for simulations.
    x2,y2,z2 : (array of) floats
       Coordinattes of object 2 (subhalo or satellite). Array for simulations.
    vx1,vy1,vz1 : (array of) floats
       Velocity of object 1 (halo or central). Array for simulations.
    vx2,vy2,vz2 : (array of) floats
       Velocity of object 2 (subhalo or satellite). Array for simulations.
    box : float
       If a simulation, side of the simulation box

    Returns
    -------
    vtheta : float
       Relative tangential velocity in the x-y plane (v theta)
    """

    dx,dy,dz = get_diffpos(x1,y1,z1,x2,y2,z2,box)
    dvx,dvy,dvz = get_diffpos(vx1,vy1,vz1,vx2,vy2,vz2)

    den = np.sqrt(dx*dx + dy*dy) 

    if hasattr(den, "__len__"):
        vtheta = np.empty(len(den)); vtheta[:] = np.nan    
        ind = np.where(den>0)
        if (np.shape(ind)[1]>0):
            vtheta[ind] =  (dx[ind]*dvy[ind] - dy[ind]*dvx[ind])/den[ind]
    else:
        vtheta = np.nan
        if (abs(den)>0): vtheta =  (dx*dvy - dy*dvx)/den
        
    return vtheta


def get_vphi(x1,y1,z1,x2,y2,z2,vx1,vy1,vz1,vx2,vy2,vz2,box=None):
    """
    Calculate the relative tangential velocity (perpendicular to the x-y plane) of an object
    
    Parameters
    ----------
    x1,y1,z1 : (array of) floats
       Coordinattes of object 1 (halo or central). Array for simulations.
    x2,y2,z2 : (array of) floats
       Coordinattes of object 2 (subhalo or satellite). Array for simulations.
    vx1,vy1,vz1 : (array of) floats
       Velocity of object 1 (halo or central). Array for simulations.
    vx2,vy2,vz2 : (array of) floats
       Velocity of object 2 (subhalo or satellite). Array for simulations.
    box : float
       If a simulation, side of the simulation box

    Returns
    -------
    vphi : float
       Relative tangential velocity perpendicular to the x-y plane (v phi)
    """
    r = get_r(x1,y1,z1,x2,y2,z2,box)

    dx,dy,dz = get_diffpos(x1,y1,z1,x2,y2,z2,box)
    dvx,dvy,dvz = get_diffpos(vx1,vy1,vz1,vx2,vy2,vz2)

    num = dz*(dx*dvx + dy*dvy) - dvz*(dx*dx + dy*dy)
    den = r*r*np.sqrt(dx*dx + dy*dy)

    if hasattr(r, "__len__"):
        vphi = np.empty(len(r)); vphi[:] = np.nan    
        ind = np.where(den>0)
        if (np.shape(ind)[1]>0):
            vphi[ind] = num[ind]/den[ind]
    else:
        vphi = np.nan
        if (abs(den)>0): vphi = num/den
        
    return vphi

def compute_vr_profile(
    galaxy_file,
    halo_file,
    output_file,
    boxsize=1000.0,
    bins=np.linspace(-1000, 1000, 201),
    halo_format="txt",
    halo_id_key=1,
    halo_pid_key=13,
    halo_x_key=4,
    halo_y_key=5,
    halo_z_key=6,
    halo_vx_key=7,
    halo_vy_key=8,
    halo_vz_key=9,
    centralformats="sage",
    galaxy_main_id_key="MainHaloID",
    galaxy_host_id_key="HostHaloID",
    galaxy_x_key="Xpos",
    galaxy_y_key="Ypos",
    galaxy_z_key="Zpos",
    galaxy_vx_key="Xvel",
    galaxy_vy_key="Yvel",
    galaxy_vz_key="Zvel",
    testing=False,
    rockstar_format=False,
    verbose=True
):
    """
    Compute the radial-velocity distribution (density) of satellite galaxies
    relative to their parent halo positions and velocities.

    Parameters:
    -----------
    galaxy_file : str
        Path to the HDF5 file containing galaxy data.
    halo_file : str
        Path to the halos catalog (TXT or HDF5, currently assumes TXT).
    output_file : str
        Path for output HDF5 file.
    boxsize : float
        Simulation box size (for periodic boundaries).
    bins : array_like
        Edges of the velocity bins.
    halo_format : str
        Format of the halo file ('txt' or 'hdf5').
    halo_id_key, halo_pid_key, halo_x_key, halo_y_key, halo_z_key, halo_vx_key, halo_vy_key, halo_vz_key : int or str
        Columns/keys for halo ID, parent ID, positions, and velocities.
    galaxy_*_key : str
        Names of the datasets for galaxy IDs, positions, and velocities.
    verbose : bool
        If True, print progress.
    """
    check_file(galaxy_file, verbose=verbose)
    check_file(halo_file, verbose=verbose)

    # 1. Load parent halos and build halo_dict
    if halo_format == "txt":
        halos = np.loadtxt(halo_file, skiprows=1, comments='#')
        if testing:
            halos = halos[:const.testlimit_halos]
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        else:
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        parent_ids = halos[mask_host, halo_id_key]
        parent_x = halos[mask_host, halo_x_key]
        parent_y = halos[mask_host, halo_y_key]
        parent_z = halos[mask_host, halo_z_key]
        parent_vx = halos[mask_host, halo_vx_key]
        parent_vy = halos[mask_host, halo_vy_key]
        parent_vz = halos[mask_host, halo_vz_key]
    else:
        raise NotImplementedError("Only txt halos implemented for now.")

    halo_dict = {hid: (x, y, z, vx, vy, vz)
                 for hid, x, y, z, vx, vy, vz in zip(parent_ids, parent_x, parent_y, parent_z, parent_vx, parent_vy, parent_vz)}

    if verbose:
        print(f"Loaded {len(halo_dict)} parent halos from {halo_file}.")

    # 2. Load galaxies
    with h5py.File(galaxy_file, "r") as f:
        host_id = f[galaxy_host_id_key][:]
        main_id = f[galaxy_main_id_key][:]
        xpos = f[galaxy_x_key][:]
        ypos = f[galaxy_y_key][:]
        zpos = f[galaxy_z_key][:]
        vx = f[galaxy_vx_key][:]
        vy = f[galaxy_vy_key][:]
        vz = f[galaxy_vz_key][:]

    # 3. Identify satellites
    #is_sat = host_id != main_id
    is_c = get_central(host_id, main_id, centralformats)
    is_sat = ~is_c

    if centralformats == 'sage':
        parent_id = main_id
    elif centralformats == 'galform':
        parent_id = host_id

    
    xs = xpos[is_sat]
    ys = ypos[is_sat]
    zs = zpos[is_sat]
    vxs = vx[is_sat]
    vys = vy[is_sat]
    vzs = vz[is_sat]
    halo_ids = parent_id[is_sat]

    n_sat = len(xs)
    xh, yh, zh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    vxh, vyh, vzh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    missing = 0

    for i, hid in enumerate(halo_ids):
        if hid in halo_dict:
            xh[i], yh[i], zh[i], vxh[i], vyh[i], vzh[i] = halo_dict[hid]
        else:
            missing += 1

    if missing > 0 and verbose:
        print(f"  {missing} satellites have no matching parent halo. Setting to NaN.")

    # 4. Compute radial velocities
    if verbose:
        print("Computing radial velocities (v_r) for satellites w.r.t. parent halo...")
    vr_all = get_vr(
        xh, yh, zh, xs, ys, zs,
        vxh, vyh, vzh, vxs, vys, vzs,
        box=boxsize
    )

    # Keep only finite values
    mask = np.isfinite(vr_all)
    vr = vr_all[mask]
    n_used = len(vr)

    if verbose:
        print(f"Number of valid satellite v_r: {n_used}")

    # Compute histogram: counts per bin, then density = counts / bin_width
    hist, bin_edges = np.histogram(vr, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    density = hist / bin_widths

    # Save to HDF5
    if verbose:
        print(f"Writing v_r profile density to {output_file}")
    with h5py.File(output_file, "w") as fout:
        fout.create_dataset("velocity_bins", data=bin_centers)
        fout.create_dataset("density", data=density)
        fout.attrs["n_bins"] = len(bin_centers)
        fout.attrs["n_galaxies"] = len(vr)

    if verbose:
        print("v_r profile stored successfully.")
    return output_file

def compute_vr_profile_shuffled(
    galaxy_file,
    halo_file,
    output_file,
    boxsize=1000.0,
    bins=np.linspace(-1000, 1000, 201),
    halo_format="txt",
    halo_id_key=1,
    halo_pid_key=13,
    halo_x_key=4,
    halo_y_key=5,
    halo_z_key=6,
    halo_vx_key=7,
    halo_vy_key=8,
    halo_vz_key=9,
    centralformats="sage",
    galaxy_main_id_key="MainHaloID",
    galaxy_host_id_key="HostHaloID",
    galaxy_is_central_key="is_central",
    galaxy_x_key="Xpos",
    galaxy_y_key="Ypos",
    galaxy_z_key="Zpos",
    galaxy_vx_key="Xvel",
    galaxy_vy_key="Yvel",
    galaxy_vz_key="Zvel",
    testing=False,
    rockstar_format=False,
    verbose=True
):
    """
    Compute the radial-velocity distribution (density) of satellite galaxies
    relative to their parent halo positions and velocities.

    Parameters:
    -----------
    galaxy_file : str
        Path to the HDF5 file containing galaxy data.
    halo_file : str
        Path to the halos catalog (TXT or HDF5, currently assumes TXT).
    output_file : str
        Path for output HDF5 file.
    boxsize : float
        Simulation box size (for periodic boundaries).
    bins : array_like
        Edges of the velocity bins.
    halo_format : str
        Format of the halo file ('txt' or 'hdf5').
    halo_id_key, halo_pid_key, halo_x_key, halo_y_key, halo_z_key, halo_vx_key, halo_vy_key, halo_vz_key : int or str
        Columns/keys for halo ID, parent ID, positions, and velocities.
    galaxy_*_key : str
        Names of the datasets for galaxy IDs, positions, and velocities.
    verbose : bool
        If True, print progress.
    """
    check_file(galaxy_file, verbose=verbose)
    check_file(halo_file, verbose=verbose)

    # 1. Load parent halos and build halo_dict
    if halo_format == "txt":
        halos = np.loadtxt(halo_file, skiprows=1, comments='#')
        if testing:
            halos = halos[:const.testlimit_halos]
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        else:
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        parent_ids = halos[mask_host, halo_id_key]
        parent_x = halos[mask_host, halo_x_key]
        parent_y = halos[mask_host, halo_y_key]
        parent_z = halos[mask_host, halo_z_key]
        parent_vx = halos[mask_host, halo_vx_key]
        parent_vy = halos[mask_host, halo_vy_key]
        parent_vz = halos[mask_host, halo_vz_key]
    else:
        raise NotImplementedError("Only txt halos implemented for now.")

    halo_dict = {hid: (x, y, z, vx, vy, vz)
                 for hid, x, y, z, vx, vy, vz in zip(parent_ids, parent_x, parent_y, parent_z, parent_vx, parent_vy, parent_vz)}

    if verbose:
        print(f"Loaded {len(halo_dict)} parent halos from {halo_file}.")

    # 2. Load galaxies
    with h5py.File(galaxy_file, "r") as f:
        is_central = f[galaxy_is_central_key][:]
        main_id = f[galaxy_main_id_key][:]
        host_id = f[galaxy_host_id_key][:]
        xpos = f[galaxy_x_key][:]
        ypos = f[galaxy_y_key][:]
        zpos = f[galaxy_z_key][:]
        vx = f[galaxy_vx_key][:]
        vy = f[galaxy_vy_key][:]
        vz = f[galaxy_vz_key][:]

    # 3. Identify satellites
    is_sat = (is_central == 0)


    if centralformats == 'sage':
        parent_id = main_id
    elif centralformats == 'galform':
        parent_id = host_id
    
    xs = xpos[is_sat]
    ys = ypos[is_sat]
    zs = zpos[is_sat]
    vxs = vx[is_sat]
    vys = vy[is_sat]
    vzs = vz[is_sat]
    halo_ids = parent_id[is_sat]

    n_sat = len(xs)
    xh, yh, zh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    vxh, vyh, vzh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    missing = 0

    for i, hid in enumerate(halo_ids):
        if hid in halo_dict:
            xh[i], yh[i], zh[i], vxh[i], vyh[i], vzh[i] = halo_dict[hid]
        else:
            missing += 1

    if missing > 0 and verbose:
        print(f"  {missing} satellites have no matching parent halo. Setting to NaN.")

    # 4. Compute radial velocities
    if verbose:
        print("Computing radial velocities (v_r) for satellites w.r.t. parent halo...")
    vr_all = get_vr(
        xh, yh, zh, xs, ys, zs,
        vxh, vyh, vzh, vxs, vys, vzs,
        box=boxsize
    )

    # Keep only finite values
    mask = np.isfinite(vr_all)
    vr = vr_all[mask]
    n_used = len(vr)

    if verbose:
        print(f"Number of valid satellite v_r: {n_used}")

    # Compute histogram: counts per bin, then density = counts / bin_width
    hist, bin_edges = np.histogram(vr, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    density = hist / bin_widths

    # Save to HDF5
    if verbose:
        print(f"Writing v_r profile density to {output_file}")
    with h5py.File(output_file, "w") as fout:
        fout.create_dataset("velocity_bins", data=bin_centers)
        fout.create_dataset("density", data=density)
        fout.attrs["n_bins"] = len(bin_centers)
        fout.attrs["n_galaxies"] = len(vr)

    if verbose:
        print("v_r profile stored successfully.")
    return output_file

def compute_vtan_profile(
    galaxy_file,
    halo_file,
    output_file,
    boxsize=1000.0,
    bins=np.linspace(0, 1000, 201),
    halo_format="txt",
    halo_id_key=1,
    halo_pid_key=13,
    halo_x_key=3,
    halo_y_key=4,
    halo_z_key=5,
    halo_vx_key=6,
    halo_vy_key=7,
    halo_vz_key=8,
    galaxy_main_id_key="MainHaloID",
    galaxy_host_id_key="HostHaloID",
    galaxy_x_key="Xpos",
    galaxy_y_key="Ypos",
    galaxy_z_key="Zpos",
    galaxy_vx_key="Xvel",
    galaxy_vy_key="Yvel",
    galaxy_vz_key="Zvel",
    centralformats="sage",
    testing=False,
    rockstar_format=False,
    verbose=True
):
    """
    Compute the tangential velocity (v_tan) profile of satellite galaxies with respect to their parent halos.
    Uses |v_tan| = sqrt(|v|^2 - v_r^2), where:
      - |v| is the relative speed between satellite and halo,
      - v_r is the radial component (already calculated using get_vr).

    Parameters:
    -----------
    galaxy_file : str
        Path to the HDF5 file containing galaxy data.
    halo_file : str
        Path to the halos catalog (TXT or HDF5, currently assumes TXT).
    output_file : str
        Path for output HDF5 file.
    boxsize : float
        Simulation box size (for periodic boundaries, applied only to positions).
    bins : array_like
        Edges of the velocity bins.
    halo_format : str
        Format of the halo file ('txt' or 'hdf5').
    col_id, col_pid, col_x, col_y, col_z, col_vx, col_vy, col_vz : int or str
        Columns/keys for halo ID, parent ID, positions, and velocities.
    galaxy_*_key : str
        Names of the datasets for galaxy IDs, positions, and velocities.
    verbose : bool
        If True, print progress.
    """
    check_file(galaxy_file, verbose=verbose)
    check_file(halo_file, verbose=verbose)

    # 1. Load parent halos and build halo_dict
    if halo_format == "txt":
        halos = np.loadtxt(halo_file, skiprows=1, comments='#')
        if testing:
            halos = halos[:const.testlimit_halos]
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        else:
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        parent_ids = halos[mask_host, halo_id_key]
        parent_x = halos[mask_host, halo_x_key]
        parent_y = halos[mask_host, halo_y_key]
        parent_z = halos[mask_host, halo_z_key]
        parent_vx = halos[mask_host, halo_vx_key]
        parent_vy = halos[mask_host, halo_vy_key]
        parent_vz = halos[mask_host, halo_vz_key]
    else:
        raise NotImplementedError("Only txt halos implemented for now.")

    halo_dict = {hid: (x, y, z, vx, vy, vz)
                 for hid, x, y, z, vx, vy, vz in zip(parent_ids, parent_x, parent_y, parent_z, parent_vx, parent_vy, parent_vz)}

    if verbose:
        print(f"Loaded {len(halo_dict)} parent halos from {halo_file}.")

    # 2. Load galaxies
    with h5py.File(galaxy_file, "r") as f:
        host_id = f[galaxy_host_id_key][:]
        main_id = f[galaxy_main_id_key][:]
        xpos = f[galaxy_x_key][:]
        ypos = f[galaxy_y_key][:]
        zpos = f[galaxy_z_key][:]
        vx = f[galaxy_vx_key][:]
        vy = f[galaxy_vy_key][:]
        vz = f[galaxy_vz_key][:]

    # 3. Identify satellites
    #is_sat = host_id != main_id
    is_c = get_central(host_id, main_id, centralformats)
    is_sat = ~is_c


    if centralformats == 'sage':
        parent_id = main_id
    elif centralformats == 'galform':
        parent_id = host_id

    
    xs = xpos[is_sat]
    ys = ypos[is_sat]
    zs = zpos[is_sat]
    vxs = vx[is_sat]
    vys = vy[is_sat]
    vzs = vz[is_sat]
    halo_ids = parent_id[is_sat]

    n_sat = len(xs)
    xh, yh, zh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    vxh, vyh, vzh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    missing = 0

    for i, hid in enumerate(halo_ids):
        if hid in halo_dict:
            xh[i], yh[i], zh[i], vxh[i], vyh[i], vzh[i] = halo_dict[hid]
        else:
            missing += 1

    if missing > 0 and verbose:
        print(f"  {missing} satellites have no matching parent halo. Setting to NaN.")

    # 4. Compute relative velocities
    dvx = vxs - vxh
    dvy = vys - vyh
    dvz = vzs - vzh
    # Relative velocity modulus |v| (no periodicity correction needed for velocities)
    vmod = np.sqrt(dvx**2 + dvy**2 + dvz**2)

    # 5. Compute v_r using get_vr (this will use box for positions, but NOT for velocities)
    vr_all = get_vr(
        xh, yh, zh, xs, ys, zs,
        vxh, vyh, vzh, vxs, vys, vzs,
        box=boxsize
    )

    # 6. Compute |v_tan| = sqrt(|v|^2 - v_r^2)
    vtan_all = np.sqrt(np.maximum(0, vmod**2 - vr_all**2))  # avoid negatives from float errors

    # Keep only finite values
    mask = np.isfinite(vtan_all)
    vtan = vtan_all[mask]
    n_used = len(vtan)

    if verbose:
        print(f"Number of valid satellite v_tan: {n_used}")

    # Compute histogram: counts per bin, then density = counts / bin_width
    hist, bin_edges = np.histogram(vtan, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    density = hist / bin_widths

    # Save to HDF5
    if verbose:
        print(f"Writing v_tan profile density to {output_file}")
    with h5py.File(output_file, "w") as fout:
        fout.create_dataset("velocity_bins", data=bin_centers)
        fout.create_dataset("density", data=density)
        fout.attrs["n_bins"] = len(bin_centers)
        fout.attrs["n_galaxies"] = n_used

    if verbose:
        print("v_tan profile stored successfully.")
    return output_file

def compute_vtan_profile_shuffled(
    galaxy_file,
    halo_file,
    output_file,
    boxsize=1000.0,
    bins=np.linspace(0, 1000, 201),
    halo_format="txt",
    halo_id_key=1,
    halo_pid_key=13,
    halo_x_key=3,
    halo_y_key=4,
    halo_z_key=5,
    halo_vx_key=6,
    halo_vy_key=7,
    halo_vz_key=8,
    galaxy_main_id_key="MainHaloID",
    galaxy_host_id_key="HostHaloID",
    galaxy_is_central_key="is_central",
    galaxy_x_key="Xpos",
    galaxy_y_key="Ypos",
    galaxy_z_key="Zpos",
    galaxy_vx_key="Xvel",
    galaxy_vy_key="Yvel",
    galaxy_vz_key="Zvel",
    centralformats="sage",
    testing=False,
    rockstar_format=False,
    verbose=True
):
    """
    Compute the tangential velocity (v_tan) profile of satellite galaxies with respect to their parent halos.
    Uses |v_tan| = sqrt(|v|^2 - v_r^2), where:
      - |v| is the relative speed between satellite and halo,
      - v_r is the radial component (already calculated using get_vr).

    Parameters:
    -----------
    galaxy_file : str
        Path to the HDF5 file containing galaxy data.
    halo_file : str
        Path to the halos catalog (TXT or HDF5, currently assumes TXT).
    output_file : str
        Path for output HDF5 file.
    boxsize : float
        Simulation box size (for periodic boundaries, applied only to positions).
    bins : array_like
        Edges of the velocity bins.
    halo_format : str
        Format of the halo file ('txt' or 'hdf5').
    col_id, col_pid, col_x, col_y, col_z, col_vx, col_vy, col_vz : int or str
        Columns/keys for halo ID, parent ID, positions, and velocities.
    galaxy_*_key : str
        Names of the datasets for galaxy IDs, positions, and velocities.
    verbose : bool
        If True, print progress.
    """
    check_file(galaxy_file, verbose=verbose)
    check_file(halo_file, verbose=verbose)

    # 1. Load parent halos and build halo_dict
    if halo_format == "txt":
        halos = np.loadtxt(halo_file, skiprows=1, comments='#')
        if testing:
            halos = halos[:const.testlimit_halos]
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        else:
            pids_array = halos[:, halo_pid_key]
            mask_host = get_mask(pids_array, rockstar_format)
        parent_ids = halos[mask_host, halo_id_key]
        parent_x = halos[mask_host, halo_x_key]
        parent_y = halos[mask_host, halo_y_key]
        parent_z = halos[mask_host, halo_z_key]
        parent_vx = halos[mask_host, halo_vx_key]
        parent_vy = halos[mask_host, halo_vy_key]
        parent_vz = halos[mask_host, halo_vz_key]
    else:
        raise NotImplementedError("Only txt halos implemented for now.")

    halo_dict = {hid: (x, y, z, vx, vy, vz)
                 for hid, x, y, z, vx, vy, vz in zip(parent_ids, parent_x, parent_y, parent_z, parent_vx, parent_vy, parent_vz)}

    if verbose:
        print(f"Loaded {len(halo_dict)} parent halos from {halo_file}.")

    # 2. Load galaxies
    with h5py.File(galaxy_file, "r") as f:
        is_central = f[galaxy_is_central_key][:]
        main_id = f[galaxy_main_id_key][:]
        host_id = f[galaxy_host_id_key][:]
        xpos = f[galaxy_x_key][:]
        ypos = f[galaxy_y_key][:]
        zpos = f[galaxy_z_key][:]
        vx = f[galaxy_vx_key][:]
        vy = f[galaxy_vy_key][:]
        vz = f[galaxy_vz_key][:]

    # 3. Identify satellites
    is_sat = (is_central == 0)


    if centralformats == 'sage':
        parent_id = main_id
    elif centralformats == 'galform':
        parent_id = host_id
    
    xs = xpos[is_sat]
    ys = ypos[is_sat]
    zs = zpos[is_sat]
    vxs = vx[is_sat]
    vys = vy[is_sat]
    vzs = vz[is_sat]
    halo_ids = parent_id[is_sat]

    n_sat = len(xs)
    xh, yh, zh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    vxh, vyh, vzh = np.full(n_sat, np.nan), np.full(n_sat, np.nan), np.full(n_sat, np.nan)
    missing = 0

    for i, hid in enumerate(halo_ids):
        if hid in halo_dict:
            xh[i], yh[i], zh[i], vxh[i], vyh[i], vzh[i] = halo_dict[hid]
        else:
            missing += 1

    if missing > 0 and verbose:
        print(f"  {missing} satellites have no matching parent halo. Setting to NaN.")

    # 4. Compute relative velocities
    dvx = vxs - vxh
    dvy = vys - vyh
    dvz = vzs - vzh
    # Relative velocity modulus |v| (no periodicity correction needed for velocities)
    vmod = np.sqrt(dvx**2 + dvy**2 + dvz**2)

    # 5. Compute v_r using get_vr (this will use box for positions, but NOT for velocities)
    vr_all = get_vr(
        xh, yh, zh, xs, ys, zs,
        vxh, vyh, vzh, vxs, vys, vzs,
        box=boxsize
    )

    # 6. Compute |v_tan| = sqrt(|v|^2 - v_r^2)
    vtan_all = np.sqrt(np.maximum(0, vmod**2 - vr_all**2))  # avoid negatives from float errors

    # Keep only finite values
    mask = np.isfinite(vtan_all)
    vtan = vtan_all[mask]
    n_used = len(vtan)

    if verbose:
        print(f"Number of valid satellite v_tan: {n_used}")

    # Compute histogram: counts per bin, then density = counts / bin_width
    hist, bin_edges = np.histogram(vtan, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    density = hist / bin_widths

    # Save to HDF5
    if verbose:
        print(f"Writing v_tan profile density to {output_file}")
    with h5py.File(output_file, "w") as fout:
        fout.create_dataset("velocity_bins", data=bin_centers)
        fout.create_dataset("density", data=density)
        fout.attrs["n_bins"] = len(bin_centers)
        fout.attrs["n_galaxies"] = n_used

    if verbose:
        print("v_tan profile stored successfully.")
    return output_file

def three_gaussians_norm(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    term1 = (A1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    term2 = (A2 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    term3 = (A3 / (np.sqrt(2 * np.pi) * sigma3)) * np.exp(-(x - mu3)**2 / (2 * sigma3**2))
    return term1 + term2 + term3

def fit_vr_profile(
    vr_profile_file,
    plot=True,
    output_png=None,
    loglog=False,
    manual_params=None,
    output_params_file=None
):
    """
    Fits the satellite radial-velocity profile with a sum of three *normalized* Gaussians.
    Optionally, overlays a user-provided manual fit.
    """
    with h5py.File(vr_profile_file, "r") as f:
        v_bins = f["velocity_bins"][:]
        density = f["density"][:]
    x = v_bins
    y = density

    # Use only nonzero y (to avoid log problems)
    mask = y > 0
    x = x[mask]
    y = y[mask]

    # Initial guess: amplitudes, means, sigmas
    guess = [-11954.702, -333.368, -127.831, -11489.693, 266.023236, -242.559427, -22660.605, -379, -312.5]
    try:
        popt, pcov = curve_fit(three_gaussians_norm, x, y, p0=guess, maxfev=10000)
    except RuntimeError:
        print("Curve fit did not converge!")
        return None

    if output_params_file is not None:
        header = "A1,mu1,sigma1,A2,mu2,sigma2,A3,mu3,sigma3"
        np.savetxt(output_params_file, np.array(popt).reshape(1,-1), delimiter=",", header=header, comments="")
        print(f"Parameters saved to: {output_params_file}")
        
    if plot:
        plt.figure(figsize=(7,5))
        if loglog:
            plt.xscale('log')
            plt.yscale('log')
        plt.plot(v_bins, density, 'b.', label='Data')
        plt.plot(x, three_gaussians_norm(x, *popt), 'r-', label='Fit (3 Norm. Gaussians)')
        if manual_params is not None:
            plt.plot(
                x, three_gaussians_norm(x, *manual_params),
                'g--', label=f'Manual fit)'
            )
        plt.xlabel(r"$v_r$ [$\mathrm{km/s}$]")
        plt.ylabel("Density [counts / bin width]")
        plt.title("Radial Velocity Profile (fit: 3 Norm. Gaussians)")
        plt.legend()
        if output_png:
            plt.tight_layout()
            plt.savefig(output_png, dpi=300)
            print(f"Plot saved to: {output_png}")
        plt.show()
        plt.close()
    print("Best-fit parameters (A1, mu1, sigma1, ...):", popt)
    if manual_params is not None:
        print("Manual parameters used (A1, mu1, sigma1, ...):", manual_params)
    return popt

def tangential_model(x, y0, alpha, kappa, beta):
    return y0 * x**alpha * np.exp(kappa * x**beta)

def fit_vtheta_profile(
    vtheta_profile_file,
    plot=True,
    output_png=None,
    loglog=False,
    manual_params=None,
    output_params_file=None
):
    """
    Fits the tangential velocity profile with:
        y = y0 * x^alpha * exp(-kappa * x^beta)
    Optionally, also plots a user-provided model (manual_params).
    """
    with h5py.File(vtheta_profile_file, "r") as f:
        v_bins = f["velocity_bins"][:]
        density = f["density"][:]
    x = v_bins
    y = density

    # Use only positive x and y
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    # Initial guess: (y0, alpha, kappa, beta)
    guess = [123, 0.8, 6.25e-4, 1.3]
    try:
        popt, pcov = curve_fit(tangential_model, x, y, p0=guess, maxfev=10000)
    except RuntimeError:
        print("Curve fit did not converge!")
        return None
    
    if output_params_file is not None:
        header = "y0,alpha,kappa,beta"
        np.savetxt(output_params_file, np.array(popt).reshape(1,-1), delimiter=",", header=header, comments="")
        print(f"Parameters saved to: {output_params_file}")

    if plot:
        plt.figure(figsize=(7,5))
        if loglog:
            plt.xscale('log')
            plt.yscale('log')
        plt.plot(v_bins, density, 'g.', label='Data')
        plt.plot(x, tangential_model(x, *popt), 'r-', label='Fit (auto)')
        if manual_params is not None:
            plt.plot(
                x, tangential_model(x, *manual_params),
                'b--', label=f'Analytical (params: {manual_params})'
            )
        plt.xlabel(r"$|v_\theta|$ [$\mathrm{km/s}$]")
        plt.ylabel("Density [counts / bin width]")
        plt.title("Tangential Velocity Profile (fit)")
        plt.legend()
        if output_png:
            plt.tight_layout()
            plt.savefig(output_png, dpi=300)
            print(f"Plot saved to: {output_png}")
        plt.show()
        plt.close()
    print("Best-fit parameters (y0, alpha, kappa, beta):", popt)
    if manual_params is not None:
        print("Manual parameters used (y0, alpha, kappa, beta):", manual_params)
    return popt
