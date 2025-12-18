import os.path
import sys
import h5py
import numpy as np
import src.h2s_const as const

def stop_if_no_file(infile):
    '''
    Stop if the file does not exist
    '''
    if (not os.path.isfile(infile)):
        print('STOP: no input file {}'.format(infile)) 
        sys.exit()
    return


def check_file(infile,verbose=False):
    '''
    Return True if the file exists
    '''
    file_fine = True  
    if (not os.path.isfile(infile)):
        file_fine = False
        if verbose:
            print('WARNING (io.check_file): file not found {}'.format(infile))

    return file_fine


def create_dir(dirout):
    '''
    Return True if directory already exists or it has been created
    '''
    if not os.path.exists(dirout):
        try:
            os.makedirs(dirout)
        except:
            print('WARNING (h2s_io.create_dir): problem creating directory ',dirout)
            return False
    return True

def print_h5attr(infile,inhead='Header'):
    """
    Print out the group attributes of a hdf5 file

    Parameters
    ----------
    infile : string
      Name of input file (this should be a hdf5 file)
    inhead : string
      Name of the group to read the attributes from

    Example
    -------
    >>> import h2s_io as io
    >>> infile = '/hpcdata0/simulations/BAHAMAS/AGN_TUNED_nu0_L100N256_WMAP9/Data/Snapshots/snapshot_026/snap_026.27.hdf5'
    >>> io.print_h5attr(infile,inhead='Units')
    """

    filefine = check_file(infile) #print(filefine)
    if (not filefine):
        print('WARNING (h2s_io.printh5attr): Check that the file provided is correct')
        return ' '
    
    f = h5py.File(infile, 'r')
    header = f[inhead]
    for hitem in list(header.attrs.items()): 
        print(hitem)
    f.close()

    return ' '

def get_nheader(infile,firstchar=None):
    '''
    Given a text file with a structure: header+data, 
    counts the number of header lines

    Parameters
    -------
    infile : string
        Input file

    Returns
    -------
    ih : integer
        Number of lines with the header text
    '''


    ih = 0
    with open(infile,'r') as ff:
        for line in ff:
            if not line.strip():
                # Count any empty lines in the header
                ih += 1
            else:
                sline = line.strip()
                
                # Check that the first character is not a digit
                char1 = sline[0]
                word1 = sline.split()[0]
                if not firstchar:
                    if (not char1.isdigit()):
                        if (char1 != '-'):
                            ih += 1
                        else:
                            try:
                                float(word1)
                                return ih
                            except:
                                ih += 1
                    else:
                        return ih
                else:
                    if char1 == firstchar:
                        ih+=1
    return ih
        
def generate_header(filenom, infile,redshift,snap,
                    h0,omega0,lambda0,vol,
                    units_h0=False,outpath=None,verbose=True):
    """
    Generate the header of the file with the line data

    Parameters
    -----------
    infile : string
        Path to input
    zz: float
        Redshift of the simulation snapshot
    snap: integer
        Simulation snapshot number
    h0 : float
        Hubble constant divided by 100
    omega0 : float
        Matter density at z=0
    omegab : float
        Baryonic density at z=0
    lambda0 : float
        Cosmological constant z=0
    vol : float
        Simulation volume
    units_h0: boolean
        True if input units with h
    outpath : string
        Path to output
    verbose : bool
        True for messages
 
    Returns
    -----
    filenom : string
       Full path to the output file
    """

    # Change units if required
    if units_h0:
        vol = vol/(h0*h0*h0)
    
    # Generate the output file (the file is rewrtitten)
    hf = h5py.File(filenom, 'w')

    # Generate a header
    headnom = 'header'
    head = hf.create_dataset(headnom,(100,))
    head.attrs[u'redshift'] = redshift
    head.attrs[u'h0'] = h0
    head.attrs[u'omega0'] = omega0
    head.attrs[u'lambda0'] = lambda0
    head.attrs[u'vol_Mpc3'] = vol
    hf.close()
    
    return filenom


def add2header(filenom,names,values,verbose=True):
    """
    Add attributes to header

    Parameters
    -----------
    filenom : string
        Path to file 
    names : list of strings
        Atribute names
    values: list
        Values of attributes
    verbose : bool
        True for messages
    """
    
    # Open the file header
    hf = h5py.File(filenom, 'a')
    head = hf['header']
    
    # Append attributes
    count = 0
    for ii, nom in enumerate(names):
        if nom is not None:
            head.attrs[nom] = values[ii]
            count += 1
    hf.close()

    if verbose: print(f'* gne_io.add2header: Appended {count} attributes out of {len(names)}')
    
    return count


def get_selection(infile, inputformat='hdf5',
                  cutcols=None, mincuts=[None], maxcuts=[None],
                  testing=True,verbose=True):
    '''
    Get indexes of selected galaxies

    Parameters
    ----------
    infile : strings
     List with the name of the input files. 
     - In text files (*.dat, *txt, *.cat), columns separated by ' '.
     - In csv files (*.csv), columns separated by ','.
    inputformat : string
     Format of the input file.
    cutcols : list
     Parameters to look for cutting the data.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    mincuts : strings
     Minimum value of the parameter of cutcols in the same index. All the galaxies below won't be considered.
    maxcuts : strings
     Maximum value of the parameter of cutcols in the same index. All the galaxies above won't be considered.
    verbose : boolean
      If True print out messages
    testing : boolean
      If True only run over few entries for testing purposes

    Returns
    -------
    selection : array of integers
    '''

    selection = None
    
    check_file(infile, verbose=verbose)

    if testing:
        limit = const.testlimit
    else:
        limit = None    

    if inputformat not in const.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(const.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            ind = np.arange(len(hf[cutcols[0]][:]))
            for i in range(len(cutcols)):
                if cutcols[i]:
                    param = hf[cutcols[i]][:]
                    mincut = mincuts[i]
                    maxcut = maxcuts[i]

                    if mincut and maxcut:
                        ind = np.intersect1d(ind,np.where((mincut<param)&(param<maxcut))[0])
                    elif mincut:
                        ind = np.intersect1d(ind,np.where(mincut<param)[0])
                    elif maxcut:
                        ind = np.intersect1d(ind,np.where(param<maxcut)[0])
            selection = ind[:limit]
    elif inputformat=='txt':
        ih = get_nheader(infile)
        ind = np.arange(len(np.loadtxt(infile,usecols=cutcols[0],skiprows=ih)))
        for i in range(len(cutcols)):
            if cutcols[i]:
                param = np.loadtxt(infile,usecols=cutcols[i],skiprows=ih)
                mincut = mincuts[i]
                maxcut = maxcuts[i]

                if mincut and maxcut:
                    ind = np.intersect1d(ind,np.where((mincut<param)&(param<maxcut))[0])
                elif mincut:
                    ind = np.intersect1d(ind,np.where(mincut<param)[0])
                elif maxcut:
                    ind = np.intersect1d(ind,np.where(param<maxcut)[0])
        selection = ind[:limit]
    else:
        if verbose:
            print('STOP (gne_io.get_selection): ',
                  'Input file has not been found.')
        sys.exit()

    return selection

def filter_log_flux(
    infile,
    fmin,
    outfile_name,
    input_format="h5",
    output_format="h5",        
    param_name="logFHalpha_att",
    verbose=True,
    testing=False,
    delimiter=" ",
):
    """
    Filters galaxies based on flux threshold for both H5 and TXT files.

    Parameters:
    -----------
    infile : str
        Path to the input file.
    fmin : float
        Minimum flux (not in log).
    outfile_name : str
        Output file name (saved in 'data/' folder).
    param_name : str
        Name or column index of the log flux field to apply cut.
        - For HDF5: field name (str)
        - For TXT: column index (int)
    input_format : str
        Format of input file: 'h5' or 'txt'.
    output_format : str
        Format of output file: 'h5' or 'txt'.
    verbose : bool
        If True, prints messages.
    testing : bool
        If True, limits output to 100 galaxies for speed testing.
    delimiter : str
        Delimiter for TXT files (default is space).

    Returns:
    --------
    output_path : str
        Path to output filtered file.
    """
    output_path = os.path.join("data", outfile_name)

    log_fmin = np.log10(fmin)

    if input_format == "h5":
        with h5py.File(infile, "r") as f:
            if param_name not in f:
                raise ValueError(f"Field '{param_name}' not found in H5 file.")

            logF = f[param_name][:]
            mask = logF > log_fmin

            if testing:
                mask[np.where(mask)[0][const.testlimit_galaxies:]] = False

            n_selected = np.sum(mask)
            if verbose:
                print(f"Galaxies selected (H5): {n_selected}")

            if output_format == "h5":
                with h5py.File(output_path, "w") as fout:
                    for key in f.keys():
                        data = f[key][:]
                        if data.shape[0] != len(mask):
                            if verbose:
                                print(f"Skipping {key} (non-matching shape)")
                            continue
                        fout.create_dataset(key, data=data[mask])
            else:
                raise NotImplementedError("Currently only H5 output supported for H5 input.")

    elif input_format == "txt":
        if not isinstance(param_name, int):
            raise ValueError("For TXT files, param_name must be an integer indicating the column index.")

        ih = get_nheader(infile)
        data = np.loadtxt(infile, skiprows=ih, delimiter=delimiter)

        logF = data[:, param_name]
        mask = logF > log_fmin

        if testing:
            mask[np.where(mask)[0][const.testlimit_galaxies:]] = False

        n_selected = np.sum(mask)
        if verbose:
            print(f"Galaxies selected (TXT): {n_selected}")

        if output_format == "txt":
            np.savetxt(output_path, data[mask], delimiter=delimiter)
        else:
            with h5py.File(output_path, "w") as fout:
                for col in range(data.shape[1]):
                    fout.create_dataset(f"col{col}", data=data[mask, col])

    else:
        raise ValueError("input_format must be 'h5' or 'txt'.")

    if verbose:
        print(f"Filtered data saved to {output_path}")

    return output_path

import numpy as np
import os

def split_halo_catalog_by_mass(
    halo_file,
    mass_column=17,
    logmass=False,
    n_bins=70,
    min_logmass=10.5,
    max_logmass=14.5,
    output_h5="data/halo_mass_bins.h5",
    columns={"id": 1, "X": 3, "Y": 4, "Z": 5, "vx": 6, "vy": 7, "vz": 8, "pid": 13, "Mass": 17},
    delimiter=None,
    verbose=True
):
    """
    Splits a halo catalog into N logarithmic mass bins and saves them in a single HDF5 file with separate groups per bin.

    Parameters
    ----------
    halo_file : str
        Path to the full halo catalog file (plain text, no header).
    mass_column : int, optional
        Index (0-based) of the column containing the halo mass.
    logmass : bool, optional
        If True, mass_column is already log10(M). Otherwise, apply log10.
    n_bins : int, optional
        Number of mass bins.
    min_logmass : float, optional
        Lower edge of the first mass bin (log10(M)).
    max_logmass : float, optional
        Upper edge of the last mass bin (log10(M)).
    output_hdf5 : str, optional
        Path to the HDF5 output file.
    columns : dict, optional
        Mapping from dataset names to column indices.
    delimiter : str or None, optional
        Delimiter for np.loadtxt (None = whitespace).
    verbose : bool, optional
        Print progress information if True.

    Returns
    -------
    bin_edges : np.ndarray
        Edges of the bins used.

    Output file structure
    ---------------------
    halo_mass_bins.h5
    ├── bin_00
    │   ├── id, X, Y, Z, vx, vy, vz, pid, Mass
    ├── bin_01
    │   ├── id, X, Y, Z, vx, vy, vz, pid, Mass
    ...
    └── bin_69
        ├── id, X, Y, Z, vx, vy, vz, pid, Mass
    """

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    # Load halo catalog
    halos = np.loadtxt(halo_file, delimiter=delimiter)
    if testing:
        masses = halos[0:c.testlimit_halos, mass_column]
    else:
        masses = halos[:, mass_column]
    mask = masses > 0  # Ensure no negative or zero masses
    halos = halos[mask]
    masses = masses[mask]
    if not logmass:
        masses = np.log10(masses)

    # Bin edges
    bin_edges = np.linspace(min_logmass, max_logmass, n_bins + 1)

    with h5py.File(output_h5, "w") as h5f:
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (masses >= lo) & (masses < hi)
            halos_bin = halos[mask]

            grp = h5f.create_group(f"bin_{i:02d}")

            # Create datasets for each specified column
            for dset_name, col_idx in columns.items():
                data = halos_bin[:, col_idx] if halos_bin.size > 0 else np.array([])
                grp.create_dataset(dset_name, data=data, dtype='f8')

            if verbose:
                print(f"Bin {i:02d}: log(M) in [{lo:.3f}, {hi:.3f}): {halos_bin.shape[0]} halos saved.")

    if verbose:
        print(f"All bins successfully saved to {output_h5}")

    return bin_edges

def kaiser_factor(omega_m, b, gamma=0.55):
    """
    Computes the Kaiser factor relating real-space and redshift-space correlation functions,
    with the growth rate exponent gamma.

    Parameters
    ----------
    omega_m : float
        Matter density parameter at the redshift of interest (Omega_m').
    b : float
        Linear bias of the galaxy/halo sample.
    gamma : float, optional
        Growth rate exponent (default: 0.55 for LCDM).

    Returns
    -------
    f : float
        Kaiser factor to multiply the real-space correlation function.
    """
    growth_rate = omega_m**gamma
    ratio = growth_rate / b
    f = 1 + (2/3) * ratio + (1/5) * ratio**2
    return f
