#!/usr/bin/python
# -*- coding: utf-8 -*-

## Tidal analysis - Comparison of simulations against observations on the North-Western European shelf #

# 29/09/2016: Dr. Karen Guihou, NOC.
#
# 3 observation files available: latobs.txt, lonobs.txt, amplobs.txt
# The original dataset is provided by the BODC (http://www.bodc.ac.uk/)
#
# 1) Extraction of model grid points at the location of the tide-gauges. latmod.txt, lonmod.txt
# The land-sea mask (lsm) is needed in order to extract only ocean points.
#
# 2) calculation of the amplitude at lonmod/latmod for each constituent
# Harmonical analysis tool is provided by NEMO (key_diaharm).
#
# 3) Calculation of RMSE and bias for statistical analysis.
#

import os
import sys
import glob
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy
from scipy import spatial
from scipy.io import netcdf
import h5py
import config

paths = config.paths
mskvar = config.mskvar
use_bathy = config.use_bathy
bathyvar = config.bathyvar
min_depth = config.min_depth
constituents = config.constituents


## Functions

def readMODELnc(filename, var):
    """
    Read a variable from a NEMO output (netcdf 3 or 4)
    """

    f = netcdf.netcdf_file(filename, 'r')
    data = f.variables[var].data
    f.close()

    return data


def readMODELhdf5(filepath, var):
    """
    Read a variable from a NEMO output (hdf5)
    """

    f = h5py.File(filepath, 'r')
    variable = (f[var])[:]
    f.close()
    return variable


def do_kdtree(combined_x_y_arrays, points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    (dist, indexes) = mytree.query(points)
    return indexes


## Read the coordinates and masks
# Obs

lonobs = np.genfromtxt(paths['data'] + 'lonobs.txt', dtype='float',
                       delimiter='\n')
latobs = np.genfromtxt(paths['data'] + 'latobs.txt', dtype='float',
                       delimiter='\n')
coordobs = np.transpose(np.concatenate((lonobs, latobs)).reshape(2,
                        len(lonobs)))

# Model

lonmod = readMODELhdf5(paths['msk'], 'nav_lon').flatten()
latmod = readMODELhdf5(paths['msk'], 'nav_lat').flatten()
coordmod = np.transpose(np.concatenate((lonmod, latmod)).reshape(2,
                        len(lonmod)))
mask = readMODELhdf5(paths['msk'], mskvar)[0, 0, :, :].flatten()
if use_bathy == 1:
    bathy = readMODELhdf5(paths['bathy'], bathyvar).flatten()

## Get indexes of model grid points where there are observations. Remove data located inland.

ind = do_kdtree(coordmod, coordobs)
counter = 0
indmod = np.zeros(len(ind), dtype=int)
indobs = np.zeros(len(ind), dtype=int)
for idx, val in enumerate(inc):
    if use_bathy == 0:
        if mask[val] == 1:

        # if abs((indmod-val)).min() != 0 : # remove double points? With that method, keeps the first tide gauge, not a mean

            indmod[counter] = val
            indobs[counter] = idx
            counter += 1
    elif bathy[val] > min_depth:
        if mask[val] == 1:
            indmod[counter] = val
            indobs[counter] = idx
            counter += 1

indmod = indmod[0:counter]
indobs = indobs[0:counter]

print ('There are ', counter, ' valid observations (out of ',
       len(lonobs), ')')

## Save the filtrered coordinates.

latobs_filt = latobs[indobs]
lonobs_filt = lonobs[indobs]
latmod_filt = latmod[indmod]
lonmod_filt = lonmod[indmod]

## Loop over constituents, extract amplitude and phase for each

for const in constituents:
    print const + ' ...'

    # Obs

    amplobs = np.genfromtxt(paths['data'] + 'amplitude_obs_'
                            + const + '.txt',
                            dtype='float', delimiter='\n')
    phaobs = np.genfromtxt(paths['data'] + 'phase_obs_'
                           + const + '.txt', dtype='float'
                           , delimiter='\n')
    amplobs_filt = amplobs[indobs]
    phaobs_filt = phaobs[indobs]

    # Testing data, just in case...
    if not len(amplobs_filt) == len(phaobs_filt) == len(lonobs_filt) == len(latobs_filt) == len(lonmod_filt) == len(latmod_filt) : #== len(indmod) ?
        sys.exit("There is something wrong with data for const %s" % const)
        continue

    data_length = len(amplobs_filt)

    # second filtering, for amplobs = 9999

    amplobs_filt2 = np.zeros(len(data_length))
    phaobs_filt2 = np.zeros(len(data_length))
    lonobs_filt2 = np.zeros(len(data_length))
    latobs_filt2 = np.zeros(len(data_length))
    lonmod_filt2 = np.zeros(len(data_length))
    latmod_filt2 = np.zeros(len(data_length))
    indmod_filt2 = np.zeros(len(data_length), dtype=int)
    valid_idx = 0
    for idx in range(latobs_filt):
        if amplobs_filt[idx] != 9999:
            amplobs_filt2[valid_ix] = amplobs_filt[idx]
            phaobs_filt2[valid_idx] = phaobs_filt[idx]
            lonobs_filt2[valid_idx] = lonobs_filt[idx]
            latobs_filt2[valid_idx] = latobs_filt[idx]
            lonmod_filt2[valid_idx] = lonmod_filt[idx]
            latmod_filt2[valid_idx] = latmod_filt[idx]
            indmod_filt2[valid_idx] = indmod[idx]
            valid_idx += 1

    # Model

    xval = readMODELhdf5(paths['model'], str(const) + 'x'
                         ).flatten()
    yval = readMODELhdf5(paths['model'], str(const) + 'y'
                         ).flatten()
    xval_filt = xval[indmod_filt2]
    yval_filt = yval[indmod_filt2]
    amplmod_filt2 = np.sqrt(np.square(xval_filt) + np.square(yval_filt))

    # Stats

    N = counter
    rmse = np.sqrt(np.sum(np.square(amplmod_filt2 * 100 - amplobs_filt2
                   * 100)) * 1 / N)
    mean = np.sum(amplmod_filt2 * 100 - amplobs_filt2 * 100) * 1 / N
    print ('N = ', N)
    print ('RMSE = ', rmse)
    print ('MEAN = ', mean)

    np.savetxt(const + '_latobs.txt', latobs_filt2)
    np.savetxt(const + '_lonobs.txt', lonobs_filt2)
    np.savetxt(const + '_latmod.txt', latmod_filt2)
    np.savetxt(const + '_lonmod.txt', lonmod_filt2)
    np.savetxt(const + '_amplobs.txt', amplobs_filt2)
    np.savetxt(const + '_amplmod.txt', amplmod_filt2)
