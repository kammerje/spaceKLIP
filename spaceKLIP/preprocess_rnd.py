from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from scipy.ndimage import fourier_shift, median_filter
from scipy.ndimage.filters import convolve
from scipy.optimize import leastsq, minimize

import webbpsf, webbpsf_ext
webbpsf_ext.setup_logging(level='ERROR', verbose=False)



# =============================================================================
# INPUTS
# =============================================================================

idir = '/Users/jkammerer/Documents/JWST Commissioning/coro/coro_realdata/NIRCam/CAR-31_sky/obs12obs349_paper/jwst_s1s2_data/'
odir = '/Users/jkammerer/Documents/JWST Commissioning/coro/coro_realdata/NIRCam/CAR-31_sky/obs12obs349_paper/jwst_s1s2_data_f335m/'

scifiles = ['jw01441001001_03106_00001_nrcalong_calints.fits',
            'jw01441002001_03106_00001_nrcalong_calints.fits']
reffiles = ['jw01441003001_03106_00001_nrcalong_calints.fits',
            'jw01441003001_03106_00002_nrcalong_calints.fits',
            'jw01441003001_03106_00003_nrcalong_calints.fits',
            'jw01441003001_03106_00004_nrcalong_calints.fits',
            'jw01441003001_03106_00005_nrcalong_calints.fits',
            'jw01441003001_03106_00006_nrcalong_calints.fits',
            'jw01441003001_03106_00007_nrcalong_calints.fits',
            'jw01441003001_03106_00008_nrcalong_calints.fits',
            'jw01441003001_03106_00009_nrcalong_calints.fits',
            'jw01441004001_03106_00001_nrcalong_calints.fits',
            'jw01441004001_03106_00002_nrcalong_calints.fits',
            'jw01441004001_03106_00003_nrcalong_calints.fits',
            'jw01441004001_03106_00004_nrcalong_calints.fits',
            'jw01441004001_03106_00005_nrcalong_calints.fits',
            'jw01441004001_03106_00006_nrcalong_calints.fits',
            'jw01441004001_03106_00007_nrcalong_calints.fits',
            'jw01441004001_03106_00008_nrcalong_calints.fits',
            'jw01441004001_03106_00009_nrcalong_calints.fits',
            'jw01441009001_03106_00001_nrcalong_calints.fits',
            'jw01441009001_03106_00002_nrcalong_calints.fits',
            'jw01441009001_03106_00003_nrcalong_calints.fits',
            'jw01441009001_03106_00004_nrcalong_calints.fits',
            'jw01441009001_03106_00005_nrcalong_calints.fits',
            'jw01441009001_03106_00006_nrcalong_calints.fits',
            'jw01441009001_03106_00007_nrcalong_calints.fits',
            'jw01441009001_03106_00008_nrcalong_calints.fits',
            'jw01441009001_03106_00009_nrcalong_calints.fits']
tamfiles = ['jw01441001001_02102_00001_nrcalong_cal.fits']

pxdq_flags = [1]

fix_bad_pixels = True
reject_bad_frames = True
fix_cross_around_isolated = False
identify_additional = False

align_frames = True


# =============================================================================
# INITIAL CHECKS
# =============================================================================

nrc = webbpsf.NIRCam()
mir = webbpsf.MIRI()

TARGPROP = []
INSTRUME = []
DETECTOR = []
FILTER = []
PUPIL = []
CORONMSK = []
SUBARRAY = []
APERNAME = []
PIXSCALE = [] # mas
HASH = []
XOFFSET = [] # mas
YOFFSET = [] # mas
ROLL_REF = [] # deg

for i in range(len(scifiles)):
    if (not scifiles[i].endswith('_calints.fits')):
        raise ValueError('Requires calints files!')
    hdul = pyfits.open(idir+scifiles[i], memmap=False)
    head = hdul[0].header
    TARGPROP += [head['TARGPROP']]
    INSTRUME += [head['INSTRUME']]
    DETECTOR += [head['DETECTOR']]
    FILTER += [head['FILTER']]
    PUPIL += [head['PUPIL']]
    CORONMSK += [head['CORONMSK']]
    SUBARRAY += [head['SUBARRAY']]
    APERNAME += [head['APERNAME']]
    if (INSTRUME[i] == 'NIRCAM'):
        if ('LONG' in DETECTOR[i]):
            PIXSCALE += [nrc._pixelscale_long*1e3] # mas
        else:
            PIXSCALE += [nrc._pixelscale_short*1e3] # mas
    elif (INSTRUME[i] == 'MIRI'):
        PIXSCALE += [mir.pixelscale*1e3] # mas
    else:
        raise UserWarning('Unknown instrument!')
    HASH += [INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]+'_'+APERNAME[i]]
    XOFFSET += [head['XOFFSET']*1e3] # mas
    YOFFSET += [head['YOFFSET']*1e3] # mas
    head = hdul['SCI'].header
    ROLL_REF += [head['ROLL_REF']] # deg
    hdul.close()

if (len(np.unique(np.array(TARGPROP))) != 1):
    raise UserWarning('There seem to be at least two different science targets!')

for i in range(len(reffiles)):
    if (not reffiles[i].endswith('_calints.fits')):
        raise ValueError('Requires calints files!')
    hdul = pyfits.open(idir+reffiles[i], memmap=False)
    head = hdul[0].header
    TARGPROP += [head['TARGPROP']]
    INSTRUME += [head['INSTRUME']]
    DETECTOR += [head['DETECTOR']]
    FILTER += [head['FILTER']]
    PUPIL += [head['PUPIL']]
    CORONMSK += [head['CORONMSK']]
    SUBARRAY += [head['SUBARRAY']]
    APERNAME += [head['APERNAME']]
    if (INSTRUME[i] == 'NIRCAM'):
        if ('LONG' in DETECTOR[i]):
            PIXSCALE += [nrc._pixelscale_long*1e3] # mas
        else:
            PIXSCALE += [nrc._pixelscale_short*1e3] # mas
    elif (INSTRUME[i] == 'MIRI'):
        PIXSCALE += [mir.pixelscale*1e3] # mas
    else:
        raise UserWarning('Unknown instrument!')
    HASH += [INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]+'_'+APERNAME[i]]
    XOFFSET += [head['XOFFSET']*1e3] # mas
    YOFFSET += [head['YOFFSET']*1e3] # mas
    head = hdul['SCI'].header
    ROLL_REF += [head['ROLL_REF']] # deg
    hdul.close()

if (len(np.unique(np.array(HASH))) != 1):
    raise UserWarning('There seem to be at least two incompatible files!')

for i in range(len(tamfiles)):
    if (not tamfiles[i].endswith('_cal.fits')):
        raise ValueError('Requires cal files!')
    hdul = pyfits.open(idir+tamfiles[i], memmap=False)
    head = hdul[0].header
    TARGPROP += [head['TARGPROP']]
    INSTRUME += [head['INSTRUME']]
    DETECTOR += [head['DETECTOR']]
    FILTER += [head['FILTER']]
    PUPIL += [head['PUPIL']]
    CORONMSK += ['TACQMASK']
    SUBARRAY += [head['SUBARRAY']]
    APERNAME += [head['APERNAME']]
    if (INSTRUME[i] == 'NIRCAM'):
        if ('LONG' in DETECTOR[i]):
            PIXSCALE += [nrc._pixelscale_long*1e3] # mas
        else:
            PIXSCALE += [nrc._pixelscale_short*1e3] # mas
    elif (INSTRUME[i] == 'MIRI'):
        PIXSCALE += [mir.pixelscale*1e3] # mas
    else:
        raise UserWarning('Unknown instrument!')
    HASH += [INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]+'_'+APERNAME[i]]
    XOFFSET += [head['XOFFSET']*1e3] # mas
    YOFFSET += [head['YOFFSET']*1e3] # mas
    head = hdul['SCI'].header
    ROLL_REF += [head['ROLL_REF']] # deg
    hdul.close()

if (not os.path.exists(odir)):
    os.makedirs(odir)


# =============================================================================
# FIX BAD PIXELS
# =============================================================================

if (fix_bad_pixels == True):
    
    allfiles = scifiles+reffiles+tamfiles
    
    for i in range(len(allfiles)):
        hdul = pyfits.open(idir+allfiles[i], memmap=False)
        data = hdul['SCI'].data
        errs = hdul['ERR'].data
        pxdq = hdul['DQ'].data
        flag = False
        if (data.ndim == 2):
            data = data[np.newaxis, :]
            errs = errs[np.newaxis, :]
            pxdq = pxdq[np.newaxis, :]
            flag = True
        threshold = 1000.
        
        if (reject_bad_frames == True):
            
            mask = pxdq < 0
            for j in range(len(pxdq_flags)):
                mask = mask | (pxdq & pxdq_flags[j] == pxdq_flags[j])
            bpfrac = np.sum(mask, axis=(1, 2))/np.prod(mask.shape[1:])
            fpfrac = np.sum(pxdq != 0, axis=(1, 2))/np.prod(pxdq.shape[1:])
            good = fpfrac/bpfrac < 2.
            data = data[good]
            errs = errs[good]
            pxdq = pxdq[good]
            print('--> Rejected %.0f of %.0f frames because of too many flagged pixels.' % (len(good)-np.sum(good), len(good)))
        
        if (fix_cross_around_isolated == True):
            
            for j in range(data.shape[0]):
                mask = pxdq[j] != 0
                kernel = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]])
                isis = convolve(mask, kernel, mode='constant')
                mask = (mask > 0.5) & (isis < 0.5)
                kernel = np.array([[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 0]])
                mask = convolve(mask, kernel, mode='constant') > 0.5
                pxdq[j][mask] = 4
        
        for j in range(data.shape[0]):
            ww = pxdq[j] != 0
            data[j][ww] = median_filter(data[j], size=7)[ww]
        
        if (identify_additional == True):
            
            for j in range(data.shape[0]):
                mfildata = median_filter(data[j], size=3)
                diffdata = np.abs(data[j]-mfildata)
                ww = (diffdata > threshold*np.nanmedian(diffdata)) | np.isnan(data[j])
                data[j][ww] = median_filter(data[j], size=7)[ww]
                # plt.imshow(diffdata/np.nanmedian(diffdata), origin='lower')
                # plt.colorbar()
                # plt.show()
                # import pdb; pdb.set_trace()
        
        else:
            
            for j in range(data.shape[0]):
                ww = np.isnan(data[j])
                data[j][ww] = median_filter(data[j], size=7)[ww]
        
        if (flag == False):
            hdul['SCI'].data = data
            hdul[0].header['NINTS'] = data.shape[0]
            hdul['ERR'].data = errs
            hdul['DQ'].data = pxdq
        else:
            hdul['SCI'].data = data[0]
            hdul[0].header['NINTS'] = data.shape[0]
            hdul['ERR'].data = errs[0]
            hdul['DQ'].data = pxdq[0]
        hdul.writeto(odir+allfiles[i], output_verify='fix', overwrite=True)
        hdul.close()


# =============================================================================
# ALIGN FRAMES
# =============================================================================

def fourier_imshift(data, shft):
    return np.fft.ifftn(fourier_shift(np.fft.fftn(data), shft[::-1])).real

def alignlsq(shft, data, base):
    return (base-shft[2]*fourier_imshift(data, shft[:2])).ravel()

def recenterlsq(shft, data):
    return 1./np.max(fourier_imshift(data, shft))

if (align_frames == True):
    
    allfiles = scifiles+reffiles+tamfiles
    
    allshfts = []
    for i in range(len(allfiles)):
        if (fix_bad_pixels == True):
            hdul = pyfits.open(odir+allfiles[i], memmap=False)
        else:
            hdul = pyfits.open(idir+allfiles[i], memmap=False)
        data = hdul['SCI'].data
        errs = hdul['ERR'].data
        pxdq = hdul['DQ'].data
        flag = False
        if (data.ndim == 2):
            data = data[np.newaxis, :]
            errs = errs[np.newaxis, :]
            pxdq = pxdq[np.newaxis, :]
            flag = True
        sh = 25
        
        head = hdul['SCI'].header
        xref = head['CRPIX1']-1.
        yref = head['CRPIX2']-1.
        
        if (i == 0):
            base = data[0][int(round(yref))-sh:int(round(yref))+sh+1, int(round(xref))-sh:int(round(xref))+sh+1].copy()
            s0 = 5
            temp = data[0][int(round(yref))-s0:int(round(yref))+s0+1, int(round(xref))-s0:int(round(xref))+s0+1].copy()
            p0 = np.array([0., 0.])
            pp = minimize(recenterlsq,
                          p0,
                          args=(temp))['x']
            test = fourier_imshift(temp, pp)
            ymax, xmax = np.unravel_index(np.argmax(test), test.shape)
            shft0 = [pp[0]+xref-int(round(xref))+s0-xmax, pp[1]+yref-int(round(yref))+s0-ymax]
        
        shfts = []
        for j in range(data.shape[0]):
            if (i < len(scifiles)+len(reffiles)):
                temp = data[j][int(round(yref))-sh:int(round(yref))+sh+1, int(round(xref))-sh:int(round(xref))+sh+1].copy()
                p0 = np.array([0., 0., 1.])
                pp = leastsq(alignlsq,
                             p0,
                             args=(temp, base))[0]
                
                shfts += [[pp[0], pp[1], pp[2]]]
                data[j] = fourier_imshift(data[j], [shft0[0]+shfts[j][0], shft0[1]+shfts[j][1]])
            else:
                temp = data[j][int(round(yref))-s0:int(round(yref))+s0+1, int(round(xref))-s0:int(round(xref))+s0+1].copy()
                p0 = np.array([0., 0.])
                pp = minimize(recenterlsq,
                              p0,
                              args=(temp))['x']
                data[j] = fourier_imshift(data[j], pp)
                ymax, xmax = np.unravel_index(np.argmax(data[j]), data[j].shape)
                pe = np.array([data[j].shape[1]//2-xmax, data[j].shape[0]//2-ymax])
                data[j] = fourier_imshift(data[j], pe)
                shfts += [[pp[0]+pe[0], pp[1]+pe[1], 1.]]
        
        shfts = np.array(shfts)
        allshfts += [shfts]
        
        if (flag == False):
            hdul['SCI'].data = data
            hdul['ERR'].data = errs
            hdul['DQ'].data = pxdq
        else:
            hdul['SCI'].data = data[0]
            hdul['SCI'].header['CRPIX1'] = data.shape[2]//2+1
            hdul['SCI'].header['CRPIX2'] = data.shape[1]//2+1
            hdul['ERR'].data = errs[0]
            hdul['DQ'].data = pxdq[0]
        hdul.writeto(odir+allfiles[i], output_verify='fix', overwrite=True)
        hdul.close()
    
    f = plt.figure()
    ax = plt.gca()
    for i in range(0, len(scifiles)):
        ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i], s=5, marker='o', label='PA = %.0f deg' % ROLL_REF[i])
    ax.axhline(0., color='gray')
    ax.axvline(0., color='gray')
    ax.set_aspect('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrng = xlim[1]-xlim[0]
    yrng = ylim[1]-ylim[0]
    if (xrng > yrng):
        ax.set_ylim(np.mean(ylim)-xrng/2., np.mean(ylim)+xrng/2.)
    else:
        ax.set_xlim(np.mean(xlim)-yrng/2., np.mean(xlim)+yrng/2.)
    ax.set_xlabel('x-shift [mas]')
    ax.set_ylabel('y-shift [mas]')
    ax.legend(loc='upper right')
    ax.set_title('Science frame alignment')
    plt.savefig('align_sci.pdf')
    plt.close()
    
    f = plt.figure()
    ax = plt.gca()
    seen = []
    reps = []
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    syms = ['o', 'v', '^', '<', '>']
    for i in range(len(scifiles), len(scifiles)+len(reffiles)):
        this = '%.0f_%.0f' % (XOFFSET[i], YOFFSET[i])
        if (this not in seen):
            ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i], s=5, color=cols[len(seen)], marker=syms[0], label='dpos %.0f' % (len(seen)+1))
            ax.vlines(-XOFFSET[i], -YOFFSET[i]-4., -YOFFSET[i]+4., color=cols[len(seen)])
            ax.hlines(-YOFFSET[i], -XOFFSET[i]-4., -XOFFSET[i]+4., color=cols[len(seen)])
            seen += [this]
            reps += [1]
        else:
            ww = np.where(np.array(seen) == this)[0][0]
            ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i], s=5, color=cols[ww], marker=syms[reps[ww]])
            reps[ww] += 1
    ax.set_aspect('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrng = xlim[1]-xlim[0]
    yrng = ylim[1]-ylim[0]
    if (xrng > yrng):
        ax.set_ylim(np.mean(ylim)-xrng/2., np.mean(ylim)+xrng/2.)
    else:
        ax.set_xlim(np.mean(xlim)-yrng/2., np.mean(xlim)+yrng/2.)
    ax.set_xlabel('x-shift [mas]')
    ax.set_ylabel('y-shift [mas]')
    ax.legend(loc='upper right')
    ax.set_title('Reference frame alignment')
    plt.savefig('align_ref.pdf')
    plt.close()
    
    f = plt.figure()
    ax = plt.gca()
    for i in range(len(scifiles)+len(reffiles), len(scifiles)+len(reffiles)+len(tamfiles)):
        ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i], s=5, marker='o')
    ax.set_aspect('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrng = xlim[1]-xlim[0]
    yrng = ylim[1]-ylim[0]
    if (xrng > yrng):
        ax.set_ylim(np.mean(ylim)-xrng/2., np.mean(ylim)+xrng/2.)
    else:
        ax.set_xlim(np.mean(xlim)-yrng/2., np.mean(xlim)+yrng/2.)
    ax.set_xlabel('x-shift [mas]')
    ax.set_ylabel('y-shift [mas]')
    ax.set_title('TA frame alignment')
    plt.savefig('align_tam.pdf')
    plt.close()

# allshfts = []
# for i in range(len(allfiles)):
#     hdul = pyfits.open(odir+allfiles[i], memmap=False)
#     data = hdul['SCI'].data
#     errs = hdul['ERR'].data
#     pxdq = hdul['DQ'].data
#     flag = False
#     if (data.ndim == 2):
#         data = data[np.newaxis, :]
#         errs = errs[np.newaxis, :]
#         pxdq = pxdq[np.newaxis, :]
#         flag = True
#     sh = 25
    
#     head = hdul['SCI'].header
#     xref = head['CRPIX1']-1.
#     yref = head['CRPIX2']-1.
    
#     if (i == 0):
#         base = data[0][int(round(yref))-sh:int(round(yref))+sh+1, int(round(xref))-sh:int(round(xref))+sh+1]
    
#     shfts = []
#     for j in range(data.shape[0]):
#         temp = data[j][int(round(yref))-sh:int(round(yref))+sh+1, int(round(xref))-sh:int(round(xref))+sh+1]
#         p0 = np.array([0., 0., 1.])
#         pp = leastsq(alignlsq,
#                      p0,
#                      args=(temp, base))[0]
        
#         shfts += [[pp[0], pp[1], pp[2]]]
#     shfts = np.array(shfts)
#     allshfts += [shfts]

# f = plt.figure()
# ax = plt.gca()
# for i in range(0, len(scifiles)):
#     ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i])
# ax.set_aspect('equal')
# ax.set_xlabel('x-shift [mas]')
# ax.set_ylabel('y-shift [mas]')
# ax.set_title('Science frame alignment')
# plt.show()

# f = plt.figure()
# ax = plt.gca()
# for i in range(len(scifiles), len(scifiles)+len(reffiles)):
#     ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i])
# ax.set_aspect('equal')
# ax.set_xlabel('x-shift [mas]')
# ax.set_ylabel('y-shift [mas]')
# ax.set_title('Reference frame alignment')
# plt.show()

# f = plt.figure()
# ax = plt.gca()
# for i in range(len(scifiles)+len(reffiles), len(scifiles)+len(reffiles)+len(tamfiles)):
#     ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i])
# ax.set_aspect('equal')
# ax.set_xlabel('x-shift [mas]')
# ax.set_ylabel('y-shift [mas]')
# ax.set_title('TA frame alignment')
# plt.show()

# import pdb; pdb.set_trace()

# allshfts = []
# for i in range(len(allfiles)):
#     hdul = pyfits.open(odir+allfiles[i], memmap=False)
#     data = hdul['SCI'].data
#     errs = hdul['ERR'].data
#     pxdq = hdul['DQ'].data
#     flag = False
#     if (data.ndim == 2):
#         data = data[np.newaxis, :]
#         errs = errs[np.newaxis, :]
#         pxdq = pxdq[np.newaxis, :]
#         flag = True
#     sh = 5
    
#     head = hdul['SCI'].header
#     xref = head['CRPIX1']-1.
#     yref = head['CRPIX2']-1.
    
#     shfts = []
#     for j in range(data.shape[0]):
#         temp = data[j][int(round(yref))-sh:int(round(yref))+sh+1, int(round(xref))-sh:int(round(xref))+sh+1]
#         p0 = np.array([0., 0.])
#         pp = minimize(recenterlsq,
#                       p0,
#                       args=(temp))['x']
#         test = fourier_imshift(temp, pp)
#         ymax, xmax = np.unravel_index(np.argmax(test), test.shape)
        
#         shfts += [[pp[0]+xref-int(round(xref))+sh-xmax, pp[1]+yref-int(round(yref))+sh-ymax]]
#     shfts = np.array(shfts)
#     allshfts += [shfts]

# f = plt.figure()
# ax = plt.gca()
# for i in range(0, len(scifiles)):
#     ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i])
# ax.set_aspect('equal')
# ax.set_xlabel('x-shift [mas]')
# ax.set_ylabel('y-shift [mas]')
# ax.set_title('Science frame alignment')
# plt.show()

# f = plt.figure()
# ax = plt.gca()
# for i in range(len(scifiles), len(scifiles)+len(reffiles)):
#     ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i])
# ax.set_aspect('equal')
# ax.set_xlabel('x-shift [mas]')
# ax.set_ylabel('y-shift [mas]')
# ax.set_title('Reference frame alignment')
# plt.show()

# f = plt.figure()
# ax = plt.gca()
# for i in range(len(scifiles)+len(reffiles), len(scifiles)+len(reffiles)+len(tamfiles)):
#     ax.scatter(allshfts[i][:, 0]*PIXSCALE[i], allshfts[i][:, 1]*PIXSCALE[i])
# ax.set_aspect('equal')
# ax.set_xlabel('x-shift [mas]')
# ax.set_ylabel('y-shift [mas]')
# ax.set_title('TA frame alignment')
# plt.show()

# import pdb; pdb.set_trace()
