import numpy as np

testlimit = 50               # Number of Galaxies for testing 
mp   = 1.67e-27              # Proton mass, kg
c    = 2.998e8               # Light velocity, m/s
c_cm = 2.998e10              # Light velocity, cm/s
planck = 4.135e-15*1.6e-12   # Planck constant, erg*s
G = 4.3e-9                   # Gravitational constant, km^2 Mpc Ms^-1 s^-2

Lbolsun = 3.826e33 # erg/s
Msun    = 1.989e30 # kg
pc      = 3.086e16 # m


#--------------------------------------------
#   Conversion factors:
#--------------------------------------------
kg_to_Msun= 1./Msun
Mpc_to_cm = pc*1e8
yr_to_s   = 365*24*60*60
#--------------------------------------------
boltzmann = 1.38e-23 * 1e4 * kg_to_Msun/(Mpc_to_cm**2)  # Boltzmann constant, Mpc^2 Ms s^-2 K^-1


#--------------------------------------------
#   Possible options and models:
#--------------------------------------------
inputformats = ['txt','hdf5']
