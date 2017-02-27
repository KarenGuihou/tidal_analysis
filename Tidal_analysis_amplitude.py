#!/usr/bin/python
# -*- coding: utf-8 -*-

## Tidal analysis - Comparison of simulations against observations on the North-Western European shelf #

# 29/09/2016: Dr. Karen Guihou, NOC (karen.guihou@gmail.com)
#
# The tidegauge data used here come from the database of sea level constituents used by Marine Data Products
# and are restricted. Please contact the BODC (http://www.bodc.ac.uk/) for more informations.
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
import argparse

parser = argparse.ArgumentParser(description='Do some tidal analysis.')
parser.add_argument('--verbose', '-v', action='count', help='Add a verbose flag')
args = parser.parse_args()
verbose = args.verbose

paths = config.paths
mskvar = config.mskvar
umskvar = config.umskvar
vmskvar = config.vmskvar
lonvar = config.lonvar
latvar = config.latvar
use_bathy = config.use_bathy
use_inland_points = config.use_inland_points
bathyvar = config.bathyvar
min_depth = config.min_depth
constituents = config.constituents_ampl
filetype = config.filetype
filetypebat = config.filetypebat
outdir = config.outdir

if config.root_path == 'REPLACE_ME':
    sys.exit("You should edit the config file first")

## Functions

def readMODEL(filename, var,filetype):
    """
    Read a variable from a NEMO output (netcdf 3 or 4)
    """
    if filetype == 'nc':
        f = netcdf.netcdf_file(filename, 'r')
        data = f.variables[var].data
        f.close()
        return data
    elif filetype == 'hdf5':
        f = h5py.File(filename, 'r')
        variable = (f[var])[:]
        f.close()
        return variable
    else:
        print('unknown filetype. Is it something else that hdf5 or nc?')


def do_kdtree(combined_x_y_arrays, points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    (dist, indexes) = mytree.query(points)
    return indexes

def filter_invalid_cols(d, idx, ntype=float):
    data = np.zeros(len(idx))
    data = np.array([ d[x] for x in idx ],dtype=ntype)

    return data

def get_siblings(mask,lon,lat,val_center,ydim):
    siblings = []
    for ii in range(val_center-ydim-1,val_center-ydim+2) + [val_center-1, val_center+1] + range(val_center+ydim-1,val_center+ydim+2):
        if ii < 0 or ii > len(mask) - 1: # out of bound (W-E bdy)
            continue
        if ii%(ydim) == 0 or ii%(ydim-1) == 0 : # out of bound (N-S bdy)
            continue
        point=[mask[ii],lon[ii],lat[ii],ii]
        siblings.append(point)

    return np.array(siblings)

def get_closest_ocean_point(coordobs,siblings,idx):
    ind_sea = np.array(np.where(siblings[:,0] == 1)[0])
    siblings_sea = siblings[ind_sea,:]
    lonlat = np.concatenate((siblings_sea[:,1], siblings_sea[:,2]))
    lonlat2d = lonlat.reshape(2,len(siblings_sea[:,1]))
    coordmod_siblings_sea = np.transpose(lonlat2d)
    ind_closest = do_kdtree(coordmod_siblings_sea, coordobs[idx])
    
    return ind_closest,siblings_sea
    


lonobs = np.genfromtxt(paths['data'] + '/lonobs.txt', dtype='float',
                       delimiter='\n')
latobs = np.genfromtxt(paths['data'] + '/latobs.txt', dtype='float',
                       delimiter='\n')
coordobs = np.transpose(np.concatenate((lonobs, latobs)).reshape(2,
                        len(lonobs)))

# Model
if verbose :
    print ('Analysis using : %s ' % paths['idmod'])

lon = readMODEL(paths['msk'], lonvar,filetype)
xdim = lon.shape[0]
ydim = lon.shape[1]

lonmod = lon.flatten()
latmod = readMODEL(paths['msk'], latvar,filetype).flatten()
coordmod = np.transpose(np.concatenate((lonmod, latmod)).reshape(2,
                        len(lonmod)))
mask = readMODEL(paths['msk'], mskvar,filetype)[0, 0, :, :].flatten('C')
if use_bathy == 1:
    bathy = readMODEL(paths['bathy'], bathyvar,filetypebat).flatten('C')
## Get indexes of model grid points where there are observations. Remove data located inland.

ind = do_kdtree(coordmod, coordobs)
counter = 0
indmod = np.zeros(len(ind), dtype=int)
indobs = np.zeros(len(ind), dtype=int)
for idx, val in enumerate(ind):
    if use_bathy == 0:
        if mask[val] == 1:
            indmod[counter] = val
            indobs[counter] = idx
            counter += 1
        elif use_inland_points == 1:
            siblings = get_siblings(mask,lonmod,latmod,val,ydim)
            if np.sum(siblings[:,0]) > 0:
                [ind_closest,siblings_sea] = get_closest_ocean_point(coordobs,siblings,idx)
                indmod[counter] = siblings_sea[ind_closest,-1]
                indobs[counter] = idx
                counter += 1
    elif bathy[val] > min_depth:
        if mask[val] == 1:
            indmod[counter] = val
            indobs[counter] = idx
            counter += 1

indmod = indmod[0:counter]
indobs = indobs[0:counter]


if verbose:
    print ('There are %d valid observations (out of %d)' % (counter, len(lonobs)))

## Save the filtrered coordinates.

latobs = latobs[indobs]
lonobs = lonobs[indobs]
latmod = latmod[indmod]
lonmod = lonmod[indmod]


## Loop over constituents, extract amplitude and phase for each

for const in constituents:
    if verbose == 2:
        print const, ' ...'

    # Obs

    amplobs = np.genfromtxt(paths['data'] + '/amplitude_obs_'
                            + const + '.txt',
                            dtype='float', delimiter='\n')
    phaobs = np.genfromtxt(paths['data'] + '/phase_obs_'
                           + const + '.txt', dtype='float'
                           , delimiter='\n')
    amplobs_filt = amplobs[indobs]
    phaobs_filt = phaobs[indobs]


    # second filtering, for amplobs = 9999
    valid_idx = []
    for idx, val_obs_val in enumerate(amplobs_filt):
        if int(val_obs_val) != 9999 :
            valid_idx.append(idx)
    amplobs_filt = filter_invalid_cols( amplobs_filt, valid_idx )
    phaobs_filt = filter_invalid_cols( phaobs_filt, valid_idx )
    lonobs_filt = filter_invalid_cols( lonobs, valid_idx )
    latobs_filt = filter_invalid_cols( latobs, valid_idx )
    lonmod_filt = filter_invalid_cols( lonmod, valid_idx )
    latmod_filt = filter_invalid_cols( latmod, valid_idx )
    indmod_filt = filter_invalid_cols( indmod, valid_idx , int)
    
    # Testing data, just in case...
    if not len(amplobs_filt) == len(phaobs_filt) == len(lonobs_filt) == len(latobs_filt) == len(lonmod_filt) == len(latmod_filt) : #== len(indmod) ?
        sys.exit("There is something wrong with data for const %s" % const)

    # Model

    xval = readMODEL(paths['model'], str(const) + 'x'
                         ,filetype).flatten()
    yval = readMODEL(paths['model'], str(const) + 'y'
                         ,filetype).flatten()
    xval_filt = xval[indmod_filt]
    yval_filt = yval[indmod_filt]

    # third filtering, in case there are bad values in amplmod
    valid_idx = []
    for idx, val_obs_val in enumerate(xval_filt):
        if int(val_obs_val) <= 9999 :
            valid_idx.append(idx)
    amplobs_filt = filter_invalid_cols( amplobs_filt, valid_idx )
    phaobs_filt = filter_invalid_cols( phaobs_filt, valid_idx )
    lonobs_filt = filter_invalid_cols( lonobs_filt, valid_idx )
    latobs_filt = filter_invalid_cols( latobs_filt, valid_idx )
    lonmod_filt = filter_invalid_cols( lonmod_filt, valid_idx )
    latmod_filt = filter_invalid_cols( latmod_filt, valid_idx )
    indmod_filt = filter_invalid_cols( indmod_filt, valid_idx , int)
    xval_filt = filter_invalid_cols( xval_filt, valid_idx)
    yval_filt = filter_invalid_cols( yval_filt, valid_idx)
    amplmod_filt = np.sqrt(np.square(xval_filt) + np.square(yval_filt))
    
    # 4th filtering, removing points where val > 9*std
    valid_idx = []
    for idx, val_obs_val in enumerate(xval_filt):
        if amplobs_filt[idx] <= np.std(amplobs_filt)*5 or amplmod_filt[idx] <= np.std(amplmod_filt)*5:
            valid_idx.append(idx)
        else:
            print('std obs')
            print(np.std(amplobs_filt))
            print(amplobs_filt[idx])
            print('std mod')
            print(np.std(amplmod_filt))
            print(amplmod_filt[idx])
    amplobs_filt = filter_invalid_cols( amplobs_filt, valid_idx )
    phaobs_filt = filter_invalid_cols( phaobs_filt, valid_idx )
    lonobs_filt = filter_invalid_cols( lonobs_filt, valid_idx )
    latobs_filt = filter_invalid_cols( latobs_filt, valid_idx )
    lonmod_filt = filter_invalid_cols( lonmod_filt, valid_idx )
    latmod_filt = filter_invalid_cols( latmod_filt, valid_idx )
    indmod_filt = filter_invalid_cols( indmod_filt, valid_idx , int)
    amplmod_filt = filter_invalid_cols( amplmod_filt, valid_idx )
    #xval_filt = filter_invalid_cols( xval_filt, valid_idx)
    #yval_filt = filter_invalid_cols( yval_filt, valid_idx)

    # Stats

    N = len(valid_idx)
    rmse = np.sqrt(np.sum(np.square(amplmod_filt * 100 - amplobs_filt
                   * 100)) /N )
    mean = np.sum(amplmod_filt * 100 - amplobs_filt * 100) / N
    if verbose == 2:
        print ('N = ', N)
        print ('RMSE = ', rmse)
        print ('MEAN = ', mean)

    np.savetxt(outdir + const + '_latobs.txt', latobs_filt)
    np.savetxt(outdir + const + '_lonobs.txt', lonobs_filt)
    np.savetxt(outdir + const + '_latmod.txt', latmod_filt)
    np.savetxt(outdir + const + '_lonmod.txt', lonmod_filt)
    np.savetxt(outdir + const + '_amplobs.txt', amplobs_filt)
    np.savetxt(outdir + const + '_amplmod.txt', amplmod_filt)
