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
for ii in range(0, len(ind)):
    if use_bathy == 0:
        if mask[ind[ii]] == 1:

        # if abs((indmod-ind[ii])).min() != 0 : # remove double points? With that method, keeps the first tide gauge, not a mean

            indmod[counter] = ind[ii]
            indobs[counter] = ii
            counter += 1
    elif bathy[ind[ii]] > min_depth:
        if mask[ind[ii]] == 1:
            indmod[counter] = ind[ii]
            indobs[counter] = ii
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

for const in range(0, len(constituents)):
    print constituents[const] + ' ...'

    # Obs

    amplobs = np.genfromtxt(paths['data'] + 'amplitude_obs_'
                            + constituents[const] + '.txt',
                            dtype='float', delimiter='\n')
    phaobs = np.genfromtxt(paths['data'] + 'phase_obs_'
                           + constituents[const] + '.txt', dtype='float'
                           , delimiter='\n')
    amplobs_filt = amplobs[indobs]
    phaobs_filt = phaobs[indobs]

    # second filtering, for amplobs = 9999

    counter = 0
    amplobs_filt2 = np.zeros(len(amplobs_filt))
    phaobs_filt2 = np.zeros(len(phaobs_filt))
    lonobs_filt2 = np.zeros(len(lonobs_filt))
    latobs_filt2 = np.zeros(len(latobs_filt))
    lonmod_filt2 = np.zeros(len(lonmod_filt))
    latmod_filt2 = np.zeros(len(latmod_filt))
    indmod_filt2 = np.zeros(len(indmod), dtype=int)
    for ii in range(0, len(latobs_filt)):
        if amplobs_filt[ii] != 9999:
            amplobs_filt2[counter] = amplobs_filt[ii]
            phaobs_filt2[counter] = phaobs_filt[ii]
            lonobs_filt2[counter] = lonobs_filt[ii]
            latobs_filt2[counter] = latobs_filt[ii]
            lonmod_filt2[counter] = lonmod_filt[ii]
            latmod_filt2[counter] = latmod_filt[ii]
            indmod_filt2[counter] = indmod[ii]
            counter += 1

    amplobs_filt2 = amplobs_filt2[0:counter]
    phaobs_filt2 = phaobs_filt2[0:counter]
    lonobs_filt2 = lonobs_filt2[0:counter]
    latobs_filt2 = latobs_filt2[0:counter]
    lonmod_filt2 = lonmod_filt2[0:counter]
    latmod_filt2 = latmod_filt2[0:counter]
    indmod_filt2 = indmod_filt2[0:counter]

    # Model

    xval = readMODELhdf5(paths['model'], str(constituents[const]) + 'x'
                         ).flatten()
    yval = readMODELhdf5(paths['model'], str(constituents[const]) + 'y'
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

    np.savetxt(constituents[const] + '_latobs.txt', latobs_filt2)
    np.savetxt(constituents[const] + '_lonobs.txt', lonobs_filt2)
    np.savetxt(constituents[const] + '_latmod.txt', latmod_filt2)
    np.savetxt(constituents[const] + '_lonmod.txt', lonmod_filt2)
    np.savetxt(constituents[const] + '_amplobs.txt', amplobs_filt2)
    np.savetxt(constituents[const] + '_amplmod.txt', amplmod_filt2)
