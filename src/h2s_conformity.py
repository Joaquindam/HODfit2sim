import numpy as np
from src.h2s_io import get_selection, create_dir
import h5py
import os

def load_halos(halo_file, file_format, col_pid, col_mass, verbose=True): #quitar verbose=True
    
    """
    Loads the halo catalog and returns PID and mass arrays.

    Parameters
    ----------
    halo_file : str
        Path to the halo catalog file. Accepted formats: TXT (ASCII) or H5.
    file_format : {'txt', 'h5'}
        Format of the halo catalog. Use 'txt' for ASCII files or 'h5' for H5 datasets.
    col_pid : int or str
        If 'txt', the (0-based) column index for the PID field.
        If 'h5', the name of the dataset or field to use for PID.
    col_mass : int or str
        If 'txt', the (0-based) column index for the halo mass field.
        If 'h5', the name of the dataset or field to use for halo mass.

    Returns
    -------
    pid : numpy.ndarray
        Array of host PIDs, as loaded from the file.
    mass : numpy.ndarray
        Array of halo masses, as loaded from the file.

    Raises
    ------
    ValueError
        If the file format is not recognized or not supported.

    Notes
    -----
    For ASCII files, columns are assumed to be whitespace-separated and 0-based indexed.
    For H5 files, the function accesses datasets by their string names.
    """
    if verbose:  
        print(f"Loading halos from {halo_file} (format={file_format})...")
    if file_format == "txt":
        halos = np.loadtxt(halo_file)
        pid   = halos[:, col_pid]
        mass  = halos[:, col_mass]
        if verbose:  
            print(f"Loaded {halos.shape[0]} rows from TXT file.")
    elif file_format == "h5":
        with h5py.File(halo_file, "r") as f:
            pid  = f[col_pid][:]
            mass = f[col_mass][:]
        if verbose:  
            print(f"Loaded {len(pid)} halos from H5 file.")
    else:
        raise ValueError("halo_file format not recognised (use 'txt' or 'h5')")
    if verbose: 
        print("Halo data loaded successfully.\n")
        
    return pid, mass

def load_galaxies(galaxy_file, file_format, prop_host_id, prop_main_id, prop_mass):
    """
    Loads the galaxy catalog and returns HostHaloID, MainHaloID, and halo mass arrays.

    Parameters
    ----------
    galaxy_file : str
        Path to the galaxy catalog file. Accepted formats: TXT (ASCII) or H5.
    file_format : {'txt', 'h5'}
        Format of the galaxy catalog. Use 'txt' for ASCII files or 'h5' for H5 datasets.
    prop_host_id : int or str
        If 'txt', the (0-based) column index for HostHaloID.
        If 'h5', the name of the dataset or field to use for HostHaloID.
    prop_main_id : int or str
        If 'txt', the (0-based) column index for MainHaloID.
        If 'h5', the name of the dataset or field to use for MainHaloID.
    prop_mass : int or str
        If 'txt', the (0-based) column index for halo mass.
        If 'h5', the name of the dataset or field to use for halo mass.

    Returns
    -------
    host_id : numpy.ndarray
        Array of HostHaloID values, as loaded from the file.
    main_id : numpy.ndarray
        Array of MainHaloID values, as loaded from the file.
    mass : numpy.ndarray
        Array of halo masses, as loaded from the file.

    Raises
    ------
    ValueError
        If the file format is not recognized or not supported.

    Notes
    -----
    For ASCII files, columns are assumed to be whitespace-separated and 0-based indexed.
    For H5 files, the function accesses datasets by their string names.
    """
    if file_format == "txt":
        gal = np.loadtxt(galaxy_file)
        host_id = gal[:, prop_host_id]
        main_id = gal[:, prop_main_id]
        mass    = gal[:, prop_mass]
    elif file_format == "h5":
        with h5py.File(galaxy_file, "r") as f:
            host_id = f[prop_host_id][:]
            main_id = f[prop_main_id][:]
            mass    = f[prop_mass][:]
    else:
        raise ValueError("galaxy_file format not recognised (use 'txt' or 'h5')")
    return host_id, main_id, mass

def compute_conformity_parameters(
    halo_file,
    galaxy_file,
    M_min,
    M_max,
    N_bins,
    output_file,
    halo_format="txt",
    halo_pid = 13,       
    halo_mass = 17,      
    galaxy_format="h5",
    gal_host_id="HostHaloID", 
    gal_main_id="MainHaloID",
    gal_main_mass="MainMhalo",    
    verbose=True,
): 
                
    """
    Computes the satellite conformity parameters k1 and k2 as a function of halo mass.

    This function identifies central and satellite galaxies, separates satellites into
    those with/without a central galaxy in the same halo, bins all relevant quantities
    by halo mass, and computes conformity parameters per bin and globally. Results
    are saved in an output H5 file for further analysis.

    Parameters
    ----------
    halo_file : str
        Path to the input halo catalog (TXT or F5).
    galaxy_file : str
        Path to the input galaxy catalog (TXT or F5), previously filtered as needed.
    M_min : float
        Lower limit of halo mass (in log10 solar masses) for binning.
    M_max : float
        Upper limit of halo mass (in log10 solar masses) for binning.
    N_bins : int
        Number of mass bins to use for analysis.
    output_file : str
        Path for the H5 output file to store the results.
    halo_format : {'txt', 'h5'}, optional
        Format of the halo file. Default is 'txt'.
    halo_pid : int or str, optional
        Column index (if txt) or dataset name (if h5) for PID. Default is 13.
    halo_mass : int or str, optional
        Column index (if txt) or dataset name (if h5) for halo mass. Default is 17.
    galaxy_format : {'txt', 'h5'}, optional
        Format of the galaxy file. Default is 'h5'.
    gal_host_id : int or str, optional
        Column index (if txt) or dataset name (if h5) for HostHaloID.
    gal_main_id : int or str, optional
        Column index (if txt) or dataset name (if h5) for MainHaloID.
    gal_main_mass : int or str, optional
        Column index (if txt) or dataset name (if h5) for halo mass.
    verbose : bool, optional
        If True, prints progress and status messages.

    Returns
    -------
    None

    Outputs
    -------
    An H5 file is written to `output_file` containing:
        - Bin edges and counts for centrals, satellites, satellites with/without a central, and halos.
        - Per-bin k1 and k2 parameters.
        - Global k1 and k2 parameters.

    Raises
    ------
    ValueError
        If any file format or required field is not recognized.

    Notes
    -----
    Central galaxies are defined as those with HostHaloID == MainHaloID.
    Satellite galaxies are those with HostHaloID != MainHaloID.
    Satellites are further split into those with a central galaxy in their halo and those without.
    """

    # 1) Read & filter host halos (PID == -1 in col 13, logMvir from col 17)
    pid, mass = load_halos(halo_file, halo_format, halo_pid, halo_mass)
    mask_host = (pid == -1.0)
    logM_halos = np.log10(mass[mask_host])

    if verbose:
        print(f"Loaded {np.sum(mask_host)} host halos from {halo_file}")

    

    # 2) Read galaxies (no further flux cut!)
    host_id, main_id, mhalo = load_galaxies(galaxy_file, galaxy_format, gal_host_id, gal_main_id, gal_main_mass)
    mhalo = np.log10(mhalo)
    if verbose:
        print(f"Loaded {len(mhalo)} galaxies from {galaxy_file}")

    # 3) Classify centrals vs satellites
    is_c = (host_id == main_id)
    is_s = ~is_c

    C_ids = host_id[is_c]
    C_m   = mhalo[is_c]

    S_ids = main_id[is_s]
    S_m   = mhalo[is_s]

    sel_with    = np.isin(S_ids, C_ids)
    sel_without = ~sel_with
    S_mwc  = S_m[sel_with]
    S_mwoc = S_m[sel_without]

    if S_ids.size > 0:
        uniq_parents, sats_per_parent = np.unique(S_ids.astype(np.int64), return_counts=True)
        k_max = int(sats_per_parent.max())
        # H_full[k] = nº de halos con k satélites (incluiría k=0 si existiera, pero no hay ceros aquí)
        H_full = np.bincount(sats_per_parent, minlength=k_max + 1).astype(np.int64)
        # Guardamos solo k >= 1
        k_vals = np.arange(1, k_max + 1, dtype=np.int64)
        H_k = H_full[1:]
        total_halos_with_satellites = int(uniq_parents.size)
    else:
        k_vals = np.array([], dtype=np.int64)
        H_k = np.array([], dtype=np.int64)
        k_max = 0
        total_halos_with_satellites = 0

    # 4) Binning
    l_bin = (M_max - M_min) / N_bins

    M_min_bin, M_max_bin = [], []
    N_C, N_S, N_swc, N_swoc, N_H = [], [], [], [], []
    K1, K2 = [], []

    for i in range(N_bins):
        lo, hi = M_min + i*l_bin, M_min + (i+1)*l_bin

        n_c    = np.sum((C_m    >= lo)&(C_m    < hi))
        n_s    = np.sum((S_m    >= lo)&(S_m    < hi))
        n_swc  = np.sum((S_mwc  >= lo)&(S_mwc  < hi))
        n_swoc = np.sum((S_mwoc >= lo)&(S_mwoc < hi))
        n_h    = np.sum((logM_halos >= lo)&(logM_halos < hi))

        M_min_bin.append(lo)
        M_max_bin.append(hi)
        N_C.append(n_c)
        N_S.append(n_s)
        N_swc.append(n_swc)
        N_swoc.append(n_swoc)
        N_H.append(n_h)

        if n_swc != 0.0 and n_swoc != 0.0:
            k1 = (n_swc)*(1./n_c)*(1./n_s)*n_h
            k2 = (1 - k1 * (n_c/n_h))/(1 - (n_c/n_h))
            K1.append(float(k1))
            K2.append(float(k2))

        elif n_swc == 0.0 and n_swoc != 0.0:
            k1 = 0.0
            k2 = (n_swoc)*(1./(n_h-n_c))*(1./n_s)*n_h
            K1.append(k1)
            K2.append(k2)

        elif n_swc != 0.0 and n_swoc == 0.0:
            k1 = (n_swc)*(1./n_c)*(1./n_s)*n_h
            k2 = 0.0
            K1.append(k1)
            K2.append(k2)

        else:
            k1 = 0.0
            k2 = 0.0
            K1.append(k1)
            K2.append(k2)

        # per‐bin k1, k2
        #if n_c>0 and n_s>0 and n_h>0:
            #fracC = n_c / n_h
            #k1b   = (n_swc * n_h) / (n_c * n_s)
            #k2b   = (1 - k1b*fracC) / (1 - fracC) if (1 - fracC)!=0 else 0.0
        #else:
            #k1b = k2b = 0.0

        #K1.append(k1b)
        #K2.append(k2b)

    # 5) Global k1, k2
    N_C   = np.array(N_C)
    N_S   = np.array(N_S)
    N_swc = np.array(N_swc)
    N_swoc= np.array(N_swoc)
    N_H   = np.array(N_H)

    sum_c   = N_C.sum()
    sum_s   = N_S.sum()
    sum_swc = N_swc.sum()
    sum_h   = N_H.sum()

    Ncmv = np.array([ N_C[i]/N_H[i] if N_H[i]>0 else 0.0 for i in range(N_bins) ])

    if sum_s>0 and sum_h>0 and Ncmv.sum()>0:
        k1_global = sum(N_swc)/(sum(np.array(N_S) * np.array( Ncmv )))
        k2_global = sum(N_swoc)/(sum(np.array(N_S) * (1 - np.array(Ncmv))))
    else:
        k1_global = k2_global = 0.0

    if verbose:
        print(f"Global k1 = {k1_global:.4f}, k2 = {k2_global:.4f}")

     # ---------- NEW: by-mass-bin satellites-per-halo histograms (k >= 1) ----------
    # We need, for each host with >=1 satellite, its host mass (log10) and its satellite count.
    # Use the satellite list grouped by S_ids to get one record per host:
    bybin_k_list = []
    bybin_H_list = []
    bybin_counts_hosts = []  # number of hosts with >=1 satellite in the bin

    if S_ids.size > 0:
        # Sort satellites by parent ID so groups are contiguous
        order = np.argsort(S_ids)
        pid_sorted  = S_ids[order]
        mass_sorted = S_m[order]  # host mass per satellite (should be constant within a halo)

        # Group starts where parent ID changes
        grp_starts = np.r_[0, np.nonzero(pid_sorted[1:] != pid_sorted[:-1])[0] + 1]
        # One record per host with >=1 sat:
        sats_per_parent = np.diff(np.r_[grp_starts, pid_sorted.size]).astype(np.int64)   # K per host (>=1)
        host_mass_per_parent = mass_sorted[grp_starts]  # take the first mass in the group

        # Build histograms per mass bin
        edges = np.linspace(M_min, M_max, N_bins + 1)
        for i in range(N_bins):
            lo, hi = edges[i], edges[i+1]
            sel_bin = (host_mass_per_parent >= lo) & (host_mass_per_parent < hi)
            counts_in_bin = sats_per_parent[sel_bin]  # all >=1 by construction
            bybin_counts_hosts.append(int(counts_in_bin.size))

            if counts_in_bin.size > 0:
                k_max_i = int(counts_in_bin.max())
                H_full_i = np.bincount(counts_in_bin, minlength=k_max_i + 1).astype(np.int64)
                k_i = np.arange(1, k_max_i + 1, dtype=np.int64)
                H_i = H_full_i[1:]  # drop 0
            else:
                k_i = np.array([], dtype=np.int64)
                H_i = np.array([], dtype=np.int64)

            bybin_k_list.append(k_i)
            bybin_H_list.append(H_i)
    else:
        # No satellites at all → every bin is empty
        bybin_counts_hosts = [0 for _ in range(N_bins)]
        bybin_k_list = [np.array([], dtype=np.int64) for _ in range(N_bins)]
        bybin_H_list = [np.array([], dtype=np.int64) for _ in range(N_bins)]

    # 6) Save to H5
    create_dir(os.path.dirname(output_file))
    with h5py.File(output_file, "w") as out:
        grp = out.create_group("data")
        bins = grp.create_group("bins")
        bins.create_dataset("M_min_bin", data=np.array(M_min_bin))
        bins.create_dataset("M_max_bin", data=np.array(M_max_bin))
        bins.create_dataset("N_C",        data=N_C)
        bins.create_dataset("N_S",        data=N_S)
        bins.create_dataset("N_swc",      data=N_swc)
        bins.create_dataset("N_swoc",     data=N_swoc)
        bins.create_dataset("N_Halos",    data=N_H)
        bins.create_dataset("k1",         data=np.array(K1))
        bins.create_dataset("k2",         data=np.array(K2))

        glob = grp.create_group("global")
        glob.create_dataset("k1_global", data=k1_global)
        glob.create_dataset("k2_global", data=k2_global)

        sat_grp = glob.create_group("satellites_per_halo")
        sat_grp.create_dataset("k", data=k_vals)  # k = nº de satélites (1..k_max)
        sat_grp.create_dataset("N_halos_with_k_satellites", data=H_k)
        sat_grp.attrs["base"] = "Parent halos with >=1 satellite"
        sat_grp.attrs["k_min"] = 1
        sat_grp.attrs["k_max"] = int(k_max)
        sat_grp.attrs["total_halos_counted"] = int(total_halos_with_satellites)

        bybin_root = grp.create_group("by_bin")
        sph_root = bybin_root.create_group("satellites_per_halo")
        for i in range(N_bins):
            g = sph_root.create_group(f"bin_{i:03d}")
            g.attrs["M_min"] = float(M_min_bin[i])
            g.attrs["M_max"] = float(M_max_bin[i])
            g.attrs["num_halos_with_ge1_sat"] = int(bybin_counts_hosts[i])  # hosts with ≥1 sat in this bin
            g.create_dataset("k", data=bybin_k_list[i])  # 1..k_max_in_bin (may be empty)
            g.create_dataset("N_halos_with_k_satellites", data=bybin_H_list[i])  # same length as k

    if verbose:
        print("Results written to:", output_file)

def compute_conformity_parameters_shuffled(
    halo_file,
    galaxy_file,
    M_min,
    M_max,
    N_bins,
    output_file,
    halo_format="txt",
    halo_pid = 13,       
    halo_mass = 17,       
    galaxy_format="h5",
    gal_is_central="is_central", 
    gal_main_id="MainHaloID",
    gal_main_mass="MainMhalo",    
    verbose=True
):
    """
    Computes the satellite conformity parameters k1 and k2 as a function of halo mass.

    This function identifies central and satellite galaxies, separates satellites into
    those with/without a central galaxy in the same halo, bins all relevant quantities
    by halo mass, and computes conformity parameters per bin and globally. Results
    are saved in an output H5 file for further analysis.

    Parameters
    ----------
    halo_file : str
        Path to the input halo catalog (TXT or F5).
    galaxy_file : str
        Path to the input galaxy catalog (TXT or F5), previously filtered as needed.
    M_min : float
        Lower limit of halo mass (in log10 solar masses) for binning.
    M_max : float
        Upper limit of halo mass (in log10 solar masses) for binning.
    N_bins : int
        Number of mass bins to use for analysis.
    output_file : str
        Path for the H5 output file to store the results.
    halo_format : {'txt', 'h5'}, optional
        Format of the halo file. Default is 'txt'.
    halo_pid : int or str, optional
        Column index (if txt) or dataset name (if h5) for PID. Default is 13.
    halo_mass : int or str, optional
        Column index (if txt) or dataset name (if h5) for halo mass. Default is 17.
    galaxy_format : {'txt', 'h5'}, optional
        Format of the galaxy file. Default is 'h5'.
    gal_host_id : int or str, optional
        Column index (if txt) or dataset name (if h5) for HostHaloID.
    gal_main_id : int or str, optional
        Column index (if txt) or dataset name (if h5) for MainHaloID.
    gal_main_mass : int or str, optional
        Column index (if txt) or dataset name (if h5) for halo mass.
    verbose : bool, optional
        If True, prints progress and status messages.

    Returns
    -------
    None

    Outputs
    -------
    An H5 file is written to `output_file` containing:
        - Bin edges and counts for centrals, satellites, satellites with/without a central, and halos.
        - Per-bin k1 and k2 parameters.
        - Global k1 and k2 parameters.

    Raises
    ------
    ValueError
        If any file format or required field is not recognized.

    Notes
    -----
    Central galaxies are defined as those with HostHaloID == MainHaloID.
    Satellite galaxies are those with HostHaloID != MainHaloID.
    Satellites are further split into those with a central galaxy in their halo and those without.
    """

    # 1) Read & filter host halos (PID == -1 in col 13, logMvir from col 17)
    pid, mass = load_halos(halo_file, halo_format, halo_pid, halo_mass)
    mask_host = (pid == -1.0)
    logM_halos = np.log10(mass[mask_host])

    if verbose:
        print(f"Loaded {np.sum(mask_host)} host halos from {halo_file}")

    with h5py.File(galaxy_file, "r") as f:
            is_central = np.asarray(f[gal_is_central][:])
            main_id = np.asarray(f[gal_main_id][:])
            main_mass    = np.asarray(f[gal_main_mass][:])

    # 2) Read galaxies (no further flux cut!
    
    logM_gal = np.log10(main_mass)
    is_c = (is_central == 1)
    is_s = ~is_c

    if verbose:
        print(f"Loaded {len(logM_gal)} galaxies from {galaxy_file}")
    
    # ========= KEY PART: does each halo have a central? =========
    # Group by MainHaloID and compute "has_central" per halo efficiently.
    order = np.argsort(main_id)
    mid_sorted = main_id[order]
    cen_sorted = is_c[order].astype(np.int8)

    # start indices of groups (where halo id changes)
    grp_starts = np.r_[0, np.nonzero(mid_sorted[1:] != mid_sorted[:-1])[0] + 1]
    # for each group, max of is_c (i.e. any central?)
    has_central_by_group = np.maximum.reduceat(cen_sorted, grp_starts).astype(bool)
    unique_ids = mid_sorted[grp_starts]  # one id per group (sorted)

    # Map 'has_central' back to every galaxy (vectorized)
    pos = np.searchsorted(unique_ids, main_id)
    halo_has_central = has_central_by_group[pos]

    # Satellites WITH a central in their own halo / WITHOUT a central
    sat_with_c_mask  = is_s &  halo_has_central
    sat_without_mask = is_s & ~halo_has_central

    # Mass arrays (in log10) for each subset
    C_m    = logM_gal[is_c]
    S_m    = logM_gal[is_s]
    S_mwc  = logM_gal[sat_with_c_mask]    # satellites with central
    S_mwoc = logM_gal[sat_without_mask]   # satellites without central

    S_parent_ids = main_id[is_s]  # halo padre de cada satélite
    if S_parent_ids.size > 0:
        uniq_parents, sats_per_parent = np.unique(S_parent_ids.astype(np.int64), return_counts=True)
        k_max = int(sats_per_parent.max())
        H_full = np.bincount(sats_per_parent, minlength=k_max + 1).astype(np.int64)
        # Guardamos solo k >= 1 (tal y como pediste)
        k_vals = np.arange(1, k_max + 1, dtype=np.int64)
        H_k = H_full[1:]
        total_halos_with_satellites = int(uniq_parents.size)
    else:
        k_vals = np.array([], dtype=np.int64)
        H_k = np.array([], dtype=np.int64)
        k_max = 0
        total_halos_with_satellites = 0

    # --- sanity check: per bin, N_S should equal N_swc + N_swoc
    # (lo comprobaremos tras el binning)

    # ========= BINNING =========
    edges = np.linspace(M_min, M_max, N_bins + 1)
    M_min_bin = edges[:-1]
    M_max_bin = edges[1:]

    N_C,   _ = np.histogram(C_m,    bins=edges)
    N_S,   _ = np.histogram(S_m,    bins=edges)
    N_swc, _ = np.histogram(S_mwc,  bins=edges)
    N_swoc,_ = np.histogram(S_mwoc, bins=edges)
    N_H,   _ = np.histogram(logM_halos, bins=edges)

    # Optional: assert consistency
    if verbose:
        bad = (N_S != (N_swc + N_swoc))
        if np.any(bad):
            idx = np.where(bad)[0]
            print(f"[WARN] In {len(idx)} bins N_S != N_swc + N_swoc; indices: {idx}")

    # ========= k1, k2 per bin (keeping your formulas/logic) =========
    K1 = np.zeros_like(N_S, dtype=float)
    K2 = np.zeros_like(N_S, dtype=float)

    for i in range(N_bins):
        n_c, n_s, n_swc_i, n_swoc_i, n_h = N_C[i], N_S[i], N_swc[i], N_swoc[i], N_H[i]

        if n_swc_i != 0 and n_swoc_i != 0 and n_c > 0 and n_s > 0:
            k1 = (n_swc_i) * (1.0/n_c) * (1.0/n_s) * n_h
            k2 = (1 - k1 * (n_c/n_h)) / (1 - (n_c/n_h)) if (1 - (n_c/n_h)) != 0 else 0.0
        elif n_swc_i == 0 and n_swoc_i != 0 and (n_h - n_c) > 0 and n_s > 0:
            k1 = 0.0
            k2 = (n_swoc_i) * (1.0/(n_h - n_c)) * (1.0/n_s) * n_h
        elif n_swc_i != 0 and n_swoc_i == 0 and n_c > 0 and n_s > 0:
            k1 = (n_swc_i) * (1.0/n_c) * (1.0/n_s) * n_h
            k2 = 0.0
        else:
            k1 = 0.0
            k2 = 0.0

        K1[i] = float(k1)
        K2[i] = float(k2)

    # Convert to np.array if needed later
    N_C   = np.asarray(N_C)
    N_S   = np.asarray(N_S)
    N_swc = np.asarray(N_swc)
    N_swoc= np.asarray(N_swoc)
    N_H   = np.asarray(N_H)

    # ========= global k1, k2 (keeping your approach) =========
    sum_c   = N_C.sum()
    sum_s   = N_S.sum()
    sum_swc = N_swc.sum()
    sum_h   = N_H.sum()

    Ncmv = np.array([ N_C[i]/N_H[i] if N_H[i] > 0 else 0.0 for i in range(len(N_H)) ])

    if sum_s > 0 and sum_h > 0 and Ncmv.sum() > 0:
        k1_global = sum_swc / np.sum(N_S * Ncmv)
        k2_global = (N_swoc.sum()) / np.sum(N_S * (1 - Ncmv))
    else:
        k1_global = 0.0
        k2_global = 0.0

    if verbose:
        print(f"Global k1 = {k1_global:.4f}, k2 = {k2_global:.4f}")

    bybin_k_list = []
    bybin_H_list = []
    bybin_counts_hosts = []

    if S_parent_ids.size > 0:
        # Sort satellites by parent to aggregate per host
        order = np.argsort(S_parent_ids.astype(np.int64))
        pid_sorted  = S_parent_ids.astype(np.int64)[order]
        mass_sorted = S_m[order]  # host mass per satellite

        grp_starts = np.r_[0, np.nonzero(pid_sorted[1:] != pid_sorted[:-1])[0] + 1]
        sats_per_parent = np.diff(np.r_[grp_starts, pid_sorted.size]).astype(np.int64)  # K per host (>=1)
        host_mass_per_parent = mass_sorted[grp_starts]

        for i in range(N_bins):
            lo, hi = edges[i], edges[i+1]
            sel_bin = (host_mass_per_parent >= lo) & (host_mass_per_parent < hi)
            counts_in_bin = sats_per_parent[sel_bin]  # >=1 by construction
            bybin_counts_hosts.append(int(counts_in_bin.size))

            if counts_in_bin.size > 0:
                k_max_i = int(counts_in_bin.max())
                H_full_i = np.bincount(counts_in_bin, minlength=k_max_i + 1).astype(np.int64)
                k_i = np.arange(1, k_max_i + 1, dtype=np.int64)
                H_i = H_full_i[1:]  # drop 0
            else:
                k_i = np.array([], dtype=np.int64)
                H_i = np.array([], dtype=np.int64)

            bybin_k_list.append(k_i)
            bybin_H_list.append(H_i)
    else:
        bybin_counts_hosts = [0 for _ in range(N_bins)]
        bybin_k_list = [np.array([], dtype=np.int64) for _ in range(N_bins)]
        bybin_H_list = [np.array([], dtype=np.int64) for _ in range(N_bins)]

    # 6) Save to H5
    create_dir(os.path.dirname(output_file))
    with h5py.File(output_file, "w") as out:
        grp = out.create_group("data")
        bins = grp.create_group("bins")
        bins.create_dataset("M_min_bin", data=np.array(M_min_bin))
        bins.create_dataset("M_max_bin", data=np.array(M_max_bin))
        bins.create_dataset("N_C",        data=N_C)
        bins.create_dataset("N_S",        data=N_S)
        bins.create_dataset("N_swc",      data=N_swc)
        bins.create_dataset("N_swoc",     data=N_swoc)
        bins.create_dataset("N_Halos",    data=N_H)
        bins.create_dataset("k1",         data=np.array(K1))
        bins.create_dataset("k2",         data=np.array(K2))

        glob = grp.create_group("global")
        glob.create_dataset("k1_global", data=k1_global)
        glob.create_dataset("k2_global", data=k2_global)

        sat_grp = glob.create_group("satellites_per_halo")
        sat_grp.create_dataset("k", data=k_vals)  # k = nº de satélites (1..k_max)
        sat_grp.create_dataset("N_halos_with_k_satellites", data=H_k)
        sat_grp.attrs["base"] = "Parent halos with >=1 satellite"
        sat_grp.attrs["k_min"] = 1
        sat_grp.attrs["k_max"] = int(k_max)
        sat_grp.attrs["total_halos_counted"] = int(total_halos_with_satellites)

        bybin_root = grp.create_group("by_bin")
        sph_root = bybin_root.create_group("satellites_per_halo")
        for i in range(N_bins):
            g = sph_root.create_group(f"bin_{i:03d}")
            g.attrs["M_min"] = float(M_min_bin[i])
            g.attrs["M_max"] = float(M_max_bin[i])
            g.attrs["num_halos_with_ge1_sat"] = int(bybin_counts_hosts[i])  # hosts with ≥1 sat in this bin
            g.create_dataset("k", data=bybin_k_list[i])                      # 1..k_max_in_bin (may be empty)
            g.create_dataset("N_halos_with_k_satellites", data=bybin_H_list[i])
        
    if verbose:
        print("Results written to:", output_file)
