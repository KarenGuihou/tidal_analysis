#!/usr/bin/python
# -*- coding: utf-8 -*-

## Tidal analysis - Comparison of simulations against observations on the North-Western European shelf ##
#
# The original current meter data are published under the terms of the NERC Open Data Licence:
# http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
#
# 29/09/2016: Dr. Karen Guihou, NOC (karen.guihou@gmail.com)
#
# 1) Extraction of model grid points at the location of the current meters
# The land-sea mask (lsm) is needed in order to extract only ocean points.
#
# 2) calculation of the current at lonmod/latmod for each constituent
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
import tidal_ellipse

parser = argparse.ArgumentParser(description='Do some tidal analysis.')
parser.add_argument('--verbose', '-v', action='count', help='Add a verbose flag')
args = parser.parse_args()
verbose = args.verbose

paths = config.paths
mskvar = config.mskvar
umskvar = config.umskvar
vmskvar = config.vmskvar
lonvar = config.lonvar
latvar=config.latvar
use_bathy = config.use_bathy
bathyvar = config.bathyvar
min_depth = config.min_depth
constituents = config.constituents_currents
filetype = config.filetype
outdir = config.outdir

## Functions

def readMODEL(filename, var):
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

## Read the coordinates and masks
# Model
if verbose :
    print ('Analysis using : %s ' % paths['idmod'])

# calculation of location of the model points are done at T-points. But mask is applied at both U and V points.
lonmod_all = readMODEL(paths['msk'], lonvar).flatten()
latmod_all = readMODEL(paths['msk'], latvar).flatten()
coordmod_all = np.transpose(np.concatenate((lonmod_all, latmod_all)).reshape(2,
                        len(lonmod_all)))
umask = readMODEL(paths['msk'], umskvar)[0, 0, :, :].flatten()
vmask = readMODEL(paths['msk'], vmskvar)[0, 0, :, :].flatten()
if use_bathy == 1:
    bathy = readMODEL(paths['bathy'], bathyvar).flatten()


## Loop over constituents, extract amplitude and phase for each
for const in constituents:
    if verbose == 2:
        print const, ' ...'

    # Obs
    obsdata3d = np.genfromtxt(paths['data'] + 'list_ukobs.' + const + 'a', 
            dtype='float', delimiter=" ")
    lonobs3d = obsdata3d[:,1]
    latobs3d = obsdata3d[:,0]

    # Mean over depth at each tide gauge location
    counter = 0
    N = 1
    obsdata1d = np.zeros((obsdata3d.shape))
    obsdata1d[0] = obsdata3d[0]
    for ii in range(1,len(lonobs3d)):
        if latobs3d[ii] == latobs3d[ii-1] and lonobs3d[ii] == lonobs3d[ii-1]:
            obsdata1d[counter] =  obsdata3d[ii] 
        else:
            counter += 1
            obsdata1d[counter] = obsdata3d[ii]

    obsdata = obsdata1d[0:counter,:]
    lonobs = obsdata[:,1]
    latobs = obsdata[:,0]
    coordobs = np.transpose(np.concatenate((lonobs,latobs))
            .reshape(2,(len(lonobs))))
    UhObs = obsdata[:,4]
    UgObs = obsdata[:,5]
    VhObs = obsdata[:,6]
    VgObs = obsdata[:,7]

    ## Get indexes of model grid points where there are observations. 
    ## Remove data located inland.

    ind = do_kdtree(coordmod_all, coordobs)
    counter = 0
    indmod = np.zeros(len(ind), dtype=int)
    indobs = np.zeros(len(ind), dtype=int)
    for idx, val in enumerate(ind):
        if use_bathy == 0:
            if umask[val] == 1 and vmask[val] == 1:
                indmod[counter] = val
                indobs[counter] = idx
                counter += 1
        elif bathy[val] > min_depth:
            if umask[val] == 1 and vmask[val] == 1:
                indmod[counter] = val
                indobs[counter] = idx
                counter += 1

    indmod = indmod[0:counter]
    indobs = indobs[0:counter]


    ## Save the filtrered coordinates and obs.

    latobs = latobs[indobs]
    lonobs = lonobs[indobs]
    latmod = latmod_all[indmod]
    lonmod = lonmod_all[indmod]
    UhObs = UhObs[indobs]
    UgObs = UgObs[indobs]
    VhObs = VhObs[indobs]
    VgObs = VgObs[indobs]


    
    # second filtering, for UhObs,UgObs,VhObs,VgObs = -999

    valid_idx = []
    for idx, val_obs_val in enumerate(UhObs):
        if int(val_obs_val) != -999 and int(UgObs[idx]) != -999 and int(VhObs[idx]) != -999 and int(VgObs[idx]) != -999 :
            valid_idx.append(idx)
    UhObs_filt = filter_invalid_cols( UhObs, valid_idx )
    UgObs_filt = filter_invalid_cols( UgObs, valid_idx )
    VhObs_filt = filter_invalid_cols( VhObs, valid_idx )
    VgObs_filt = filter_invalid_cols( VgObs, valid_idx )
    lonobs_filt = filter_invalid_cols( lonobs, valid_idx )
    latobs_filt = filter_invalid_cols( latobs, valid_idx )
    lonmod_filt = filter_invalid_cols( lonmod, valid_idx )
    latmod_filt = filter_invalid_cols( latmod, valid_idx )
    indmod_filt = filter_invalid_cols( indmod, valid_idx , int)

    # Testing data, just in case...
    if not len(UhObs_filt) == len(UgObs_filt) == len(VhObs_filt) == len(VgObs_filt) == len(lonobs_filt) == len(latobs_filt) == len(lonmod_filt) == len(latmod_filt) : #== len(indmod) ?
        sys.exit("There is something wrong with data for const %s" % const)
    
    if verbose:
        print ('There are %d valid observations (out of %d)' % (counter, len(lonobs)))

    # Model
    # Get model data
    XuMod = readMODEL(paths['model'],str(const)+'x_u').flatten()
    XvMod = readMODEL(paths['model'],str(const)+'x_v').flatten()
    YuMod = readMODEL(paths['model'],str(const)+'y_u').flatten()
    YvMod = readMODEL(paths['model'],str(const)+'y_v').flatten()
    XuMod_filt = XuMod[indmod_filt]*100
    XvMod_filt = XvMod[indmod_filt]*100
    YuMod_filt = YuMod[indmod_filt]*100
    YvMod_filt = YvMod[indmod_filt]*100


    UhMod = np.sqrt(np.square(XuMod_filt)+np.square(YuMod_filt))
    UgMod = np.arctan(np.divide(YuMod_filt,XuMod_filt))
    VhMod = np.sqrt(np.square(XvMod_filt)+np.square(YvMod_filt)) #0
    VgMod = np.arctan(np.divide(YvMod_filt,XvMod_filt))   #nan
    
    
    ## Calculation of SEMA and stats
    
    SEMAobs, ECCobs, INCobs, PHAobs, wobs = tidal_ellipse.ap2ep(UhObs_filt, UgObs_filt, VhObs_filt, VgObs_filt , plot_demo=False)
    SEMAmod, ECCmod, INCmod, PHAmod, wmod = tidal_ellipse.ap2ep(UhMod, UgMod, VhMod, VgMod , plot_demo=False)
    
    ## Stats
    N = len(indmod_filt)
    rmse = np.sqrt(np.sum(np.square(SEMAmod - SEMAobs)) / N )
    mean = np.sum(SEMAmod - SEMAobs) / N
    print('N = ',N)
    print('RMSE = ',rmse)
    print('MEAN = ',mean)

    ## Save data
    np.savetxt(outdir + const + '_latobs_currents.txt', latobs_filt)
    np.savetxt(outdir + const + '_lonobs_currents.txt', lonobs_filt)
    np.savetxt(outdir + const + '_latmod_currents.txt', latmod_filt)
    np.savetxt(outdir + const + '_lonmod_currents.txt', lonmod_filt)
    np.savetxt(outdir + const + '_SEMAobs_currents.txt', SEMAobs)
    np.savetxt(outdir + const + '_SEMAmod_currents.txt', SEMAmod)
