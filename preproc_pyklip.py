from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import urllib.request

from astropy.table import Table
from scipy.ndimage import fourier_shift, median_filter
from scipy.optimize import leastsq


# =============================================================================
# PARAMETERS
# =============================================================================

# idir = '../Test/jwst_s1s2_data copy/'
# odir = '../Test/jwst_s1s2_data_prep/'
# idir = '../HR8799/jwst_s1s2_data copy/'
# odir = '../HR8799/jwst_s1s2_data_prep/'
idir = '../HIP65426/jwst_s1s2_data/'
odir = '../HIP65426/jwst_s1s2_data_prep/'
ftyp = 'calints'
psfmaskdir = '../psfmasks/'

fix_bad = True
median_size = 3 # pix

align = False

mask_data = True
mask_size = 150 # pix

# pxsc = 62.92632954741567 # mas, measured
pxsc = 63. # mas, simulated


# =============================================================================
# FUNCTIONS
# =============================================================================

def align_fourierLSQ(data,
                     data_mean,
                     mask=None):
    
    p0 = [0., 0., 1.]
    pp, _ = leastsq(shift_subtract,
                    p0,
                    args=(data, data_mean, mask))
    
    return pp

def shift_subtract(pp,
                   data,
                   data_mean,
                   mask=None):
    
    data_fshift = np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(data.copy()), pp[:2][::-1])))
    
    if (mask is None):
        return (data_mean-pp[2]*data_fshift).ravel()
    else:
        return ((data_mean-pp[2]*data_fshift)*mask).ravel()


# =============================================================================
# READ
# =============================================================================

fitsfiles = np.array([f for f in os.listdir(idir) if ftyp in f and f.endswith('.fits')])
Nfitsfiles = len(fitsfiles)

TARGPROP = []
TARG_RA = [] # deg
TARG_DEC = [] # deg
INSTRUME = []
DETECTOR = []
FILTER = []
PUPIL = []
CORONMSK = []
READPATT = []
NINTS = []
NGROUPS = []
NFRAMES = []
EFFINTTM = []
SUBARRAY = []
SUBPXPTS = []
HASH = []
for i in range(Nfitsfiles):
    hdul = pyfits.open(idir+fitsfiles[i])
    head = hdul[0].header
    TARGPROP += [str(head['TARGPROP'])]
    TARG_RA += [float(head['TARG_RA'])] # deg
    TARG_DEC += [float(head['TARG_DEC'])] # deg
    INSTRUME += [str(head['INSTRUME'])]
    DETECTOR += [str(head['DETECTOR'])]
    FILTER += [str(head['FILTER'])]
    PUPIL += [str(head['PUPIL'])]
    CORONMSK += [str(head['CORONMSK'])]
    READPATT += [str(head['READPATT'])]
    NINTS += [int(head['NINTS'])]
    NGROUPS += [int(head['NGROUPS'])]
    NFRAMES += [int(head['NFRAMES'])]
    EFFINTTM += [float(head['EFFINTTM'])]
    SUBARRAY += [str(head['SUBARRAY'])]
    try:
        SUBPXPTS += [int(head['SUBPXPTS'])]
    except:
        SUBPXPTS += [1]
    HASH += [INSTRUME[-1]+'_'+DETECTOR[-1]+'_'+FILTER[-1]+'_'+PUPIL[-1]+'_'+CORONMSK[-1]+'_'+SUBARRAY[-1]]
    hdul.close()
TARGPROP = np.array(TARGPROP)
TARG_RA = np.array(TARG_RA) # deg
TARG_DEC = np.array(TARG_DEC) # deg
INSTRUME = np.array(INSTRUME)
DETECTOR = np.array(DETECTOR)
FILTER = np.array(FILTER)
PUPIL = np.array(PUPIL)
CORONMSK = np.array(CORONMSK)
READPATT = np.array(READPATT)
NINTS = np.array(NINTS)
NGROUPS = np.array(NGROUPS)
NFRAMES = np.array(NFRAMES)
EFFINTTM = np.array(EFFINTTM)
SUBARRAY = np.array(SUBARRAY)
SUBPXPTS = np.array(SUBPXPTS)
HASH = np.array(HASH)

HASH_unique = np.unique(HASH)
NHASH_unique = len(HASH_unique)
obs = {}
for i in range(NHASH_unique):
    ww = HASH == HASH_unique[i]
    dpts = SUBPXPTS[ww]
    dpts_unique = np.unique(dpts)
    if ((len(dpts_unique) == 2) and (dpts_unique[0] == 1)):
        ww_sci = np.where(dpts == dpts_unique[0])[0]
        ww_cal = np.where(dpts == dpts_unique[1])[0]
    else:
        raise UserWarning()
    tab = Table(names=('TYP', 'TARGPROP', 'TARG_RA', 'TARG_DEC', 'READPATT', 'NINTS', 'NGROUPS', 'NFRAMES', 'EFFINTTM', 'FITSFILE'), dtype=('S', 'S', 'f', 'f', 'S', 'i', 'i', 'i', 'f', 'S'))
    for j in range(len(ww_sci)):
        tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], idir+fitsfiles[ww][ww_sci][j]))
    for j in range(len(ww_cal)):
        tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], idir+fitsfiles[ww][ww_cal][j]))
    obs[HASH_unique[i]] = tab.copy()

print('--> Identified %.0f observation sequences' % len(obs))
maxnumbasis = []
for i, key in enumerate(obs.keys()):
    print('--> Sequence %.0f: ' % (i+1)+key)
    print(obs[key])
    ww = obs[key]['TYP'] == 'CAL'
    maxnumbasis += [np.sum(obs[key]['NINTS'][ww])]


# =============================================================================
# MAIN
# =============================================================================

if (not os.path.exists(odir)):
    os.makedirs(odir)

for i, key in enumerate(obs.keys()):
    print('Preparing '+key)
    ww_sci = np.where(obs[key]['TYP'] == 'SCI')[0]
    scifiles = sorted(np.array(obs[key]['FITSFILE'][ww_sci], dtype=str).tolist())
    ww_cal = np.where(obs[key]['TYP'] == 'CAL')[0]
    calfiles = sorted(np.array(obs[key]['FITSFILE'][ww_cal], dtype=str).tolist())
    fitsfiles = scifiles+calfiles
    
    for j in range(len(fitsfiles)):
        hdul = pyfits.open(fitsfiles[j])
        data = hdul['SCI'].data
        pxdq = hdul['DQ'].data
        
        ww = pxdq != 0
        if (fix_bad == True):
            for k in range(data.shape[0]):
                data[k][ww[k]] = median_filter(data[k], size=median_size)[ww[k]]
        ww = np.sum(ww, axis=(1, 2)) > 0.1*data.shape[1]*data.shape[2]
        data = np.delete(data, ww, axis=0)
        pxdq = np.delete(pxdq, ww, axis=0)
        print('   Removed %.0f bad frames' % np.sum(ww))
        
        if (align == True):
            if (fix_bad == False):
                raise UserWarning()
            if ('NIRCAM_NRCALONG_F356W_MASKRND_MASKA430R_SUB320A430R' in key):
                psfmaskname = 'jwst_nircam_psfmask_0065.fits'
            elif ('NIRCAM_NRCALONG_F444W_MASKRND_MASKA430R_SUB320A430R' in key):
                psfmaskname = 'jwst_nircam_psfmask_0004.fits'
            elif ('NIRCAM_NRCALONG_F250M_MASKBAR_MASKALWB_SUB320ALWB' in key):
                psfmaskname = 'jwst_nircam_psfmask_0042.fits'
            elif ('NIRCAM_NRCALONG_F300M_MASKBAR_MASKALWB_SUB320ALWB' in key):
                psfmaskname = 'jwst_nircam_psfmask_0045.fits'
            elif ('NIRCAM_NRCALONG_F335M_MASKBAR_MASKALWB_SUB320ALWB' in key):
                psfmaskname = 'jwst_nircam_psfmask_0003.fits'
            elif ('NIRCAM_NRCALONG_F410M_MASKBAR_MASKALWB_SUB320ALWB' in key):
                psfmaskname = 'jwst_nircam_psfmask_0048.fits'
            elif ('NIRCAM_NRCALONG_F430M_MASKBAR_MASKALWB_SUB320ALWB' in key):
                psfmaskname = 'jwst_nircam_psfmask_0055.fits'
            elif ('NIRCAM_NRCALONG_F460M_MASKBAR_MASKALWB_SUB320ALWB' in key):
                psfmaskname = 'jwst_nircam_psfmask_0014.fits'
            else:
                raise UserWarning()
            try:
                psfmask = pyfits.getdata(psfmaskdir+psfmaskname, 0)
            except:
                if (not os.path.exists(psfmaskdir)):
                    os.makedirs(psfmaskdir)
                urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'+psfmaskname, psfmaskdir+psfmaskname)
                psfmask = pyfits.getdata(psfmaskdir+psfmaskname, 0)
            if (j == 0):
                data_mean = np.mean(data, axis=0)
            
            xyzs = []
            for k in range(data.shape[0]):
                xyzs += [align_fourierLSQ(data[k], data_mean, mask=psfmask)]
            xyzs = np.array(xyzs)
            
            temp = np.mean(xyzs, axis=0)
            xoff_expt = hdul[0].header['XOFFSET']*1000./pxsc
            yoff_expt = hdul[0].header['YOFFSET']*1000./pxsc
            if (xoff_expt == 0.):
                print('   Offset x: %.5f (meas), %.5f (expt)' % (-temp[0], xoff_expt))
            else:
                print('   Offset x: %.5f (meas), %.5f (expt), %.0f%%' % (-temp[0], xoff_expt, (-temp[0]-xoff_expt)/xoff_expt*100.))
            if (yoff_expt == 0.):
                print('   Offset y: %.5f (meas), %.5f (expt)' % (-temp[1], yoff_expt))
            else:
                print('   Offset y: %.5f (meas), %.5f (expt), %.0f%%' % (-temp[1], yoff_expt, (-temp[1]-yoff_expt)/yoff_expt*100.))
            
            for k in range(data.shape[0]):
                data[k] = np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(data[k].copy()), xyzs[k][:2][::-1])))
        
        if (mask_data == True):
            ycen = hdul['SCI'].header['CRPIX1']-1
            yrmp = np.arange(data.shape[1])-ycen
            xcen = hdul['SCI'].header['CRPIX2']-1
            xrmp = np.arange(data.shape[1])-xcen
            xx, yy = np.meshgrid(yrmp, xrmp)
            mask = np.sqrt(xx**2+yy**2) > mask_size
            data[:, mask] = np.nan
        
        hdul['SCI'].data = data
        hdul['DQ'].data = pxdq
        hdul[0].header['NINTS'] = data.shape[0]
        if (align == True):
            hdul[0].header['XOFFSET'] = 0.
            hdul[0].header['YOFFSET'] = 0.
        hdul.writeto(odir+fitsfiles[j][fitsfiles[j].rfind('/')+1:], output_verify='fix', overwrite=True)
        hdul.close()

print('DONE')
