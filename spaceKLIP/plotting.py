import glob, os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import numpy as np
from astropy.io import fits

from itertools import chain

def plot_contrast_images(meta, data, data_masked, pxsc=None, savefile='./maskimage.pdf'):
    """
    Plot subtracted images to be used for contrast estimation, one with
    companions marked, one with the masking adopted.

    """

    # Set some quick information depending on whether a pixel scale was passed
    if pxsc == None:
        extent=(-0.5, data.shape[1]-0.5, data.shape[1]-0.5, -0.5)
        pxsc = 1
        xlabel, ylabel = 'Pixels', 'Pixels'
    else:
        extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
        extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
        extent = (extl, -extr, -extl, extr)
        xlabel, ylabel = '$\Delta$RA [arcsec]', '$\Delta$Dec [arcsec]'

    # Initialize plots
    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))

    # Plot subtracted image, circle input companion locations
    ax[0].imshow(np.log10(np.abs(data[-1])), origin='lower', cmap='inferno', extent=extent)
    for j in range(len(meta.ra_off)):
        cc = plt.Circle((meta.ra_off[j]/1000., meta.de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
        ax[0].add_artist(cc)
    if hasattr(meta, 'ra_off_mask') and hasattr(meta, 'de_off_mask'):
        for j in range(len(meta.ra_off_mask)):
            cc = plt.Circle((meta.ra_off_mask[j]/1000., meta.de_off_mask[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
            ax[0].add_artist(cc)
        
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title('KLIP-subtracted')

    # Plot subtracted image, with adopted masking
    ax[1].imshow(np.log10(np.abs(data_masked[-1])), origin='lower', cmap='inferno', extent=extent)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    if 'SWB' in savefile or 'LWB' in savefile:
        ax[1].set_title('Companions & bar masked')
    else:
        ax[1].set_title('Companions masked')
    plt.tight_layout()

    # Save and close plot
    plt.savefig(savefile)
    plt.close()

    return

def plot_contrast_raw(meta, seps, cons, labels='default', savefile='./rawcontrast.pdf'):
    """
    Plot raw contrast curves for different KL modes.
    """

    # Initialize figure
    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()

    # Figure out if we're plotting one contrast curve, or multiple
    if len(cons) == 1:
        if labels == 'default':
            labels == 'contrast'
        ax.plot(seps, cons, label=labels)
    elif len(cons) > 1:
        if labels == 'default':
            labels = ['contrast_{}'.format(i+1) for i in range(len(seps))]

        # Loop over contrast curves to plot
        for i in range(len(cons)):
            ax.plot(seps, cons[i], label=labels[i])

    # Plot settings
    ax.set_yscale('log')
    ax.grid(axis='y')
    ax.set_xlabel('Separation [arcsec]')
    ax.set_ylabel('Contrast [5$\sigma$]')
    ax.set_title('Raw contrast curve')
    ax.legend(loc='upper right', prop=dict(size=8))
    plt.tight_layout()

    # Save and close plot
    plt.savefig(savefile)
    ax.set_xlim([0., 6.]) # arcsec
    plt.savefig(savefile.replace('raw', 'raw_trim'))
    plt.close()

    return

def plot_injected_locs(meta, data, transmission, seps, pas, pxsc=None, savefile='./injected.pdf'):
    '''
    Plot subtracted image and 2D transmission alongside locations of injected planets. 
    '''
    #Set some quick information depending on whether a pixel scale was passed
    if pxsc == None:
        extent=(-0.5, data.shape[1]-0.5, data.shape[1]-0.5, -0.5)
        extent_tr=(-0.5, transmission.shape[1]-0.5, transmission.shape[1]-0.5, -0.5)
        pxsc = 1
        xlabel, ylabel = 'Pixels', 'Pixels'
    else:
        extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
        extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
        extent = (extl, -extr, -extl, extr)

        extl = (transmission.shape[1]/2.)*pxsc/1000 # arcsec
        extr = (transmission.shape[1]/2.)*pxsc/1000 # arcsec
        extent_tr = (extl, -extr, -extl, extr)

        xlabel, ylabel = '$\Delta$RA [arcsec]', '$\Delta$DEC [arcsec]'

    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
    ax[0].imshow(np.log10(np.abs(data)), origin='lower', cmap='inferno', extent=extent)
    for i in range(len(meta.ra_off)):
        cc = plt.Circle((meta.ra_off[i]/1000., meta.de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
        ax[0].add_artist(cc)
    for i in range(len(seps)):
        ra = seps[i]*pxsc*np.sin(np.deg2rad(pas[i])) # mas
        de = seps[i]*pxsc*np.cos(np.deg2rad(pas[i])) # mas
        cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
        ax[0].add_artist(cc)
    # ax[0].set_xlim([5., -5.])
    # ax[0].set_ylim([-5., 5.])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title('KLIP-subtracted')

    p1 = ax[1].imshow(transmission, origin='lower', cmap='viridis', vmin=0., vmax=1., extent=extent_tr)
    c1 = plt.colorbar(p1, ax=ax[1])
    c1.set_label('Transmission', rotation=270, labelpad=20)
    for i in range(len(meta.ra_off)):
        cc = plt.Circle((meta.ra_off[i]/1000., meta.de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
        ax[1].add_artist(cc)
    for i in range(len(seps)):
        ra = seps[i]*pxsc*np.sin(np.deg2rad(pas[i])) # mas
        de = seps[i]*pxsc*np.cos(np.deg2rad(pas[i])) # mas
        cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
        ax[1].add_artist(cc)
    # ax[1].set_xlim([5., -5.])
    # ax[1].set_ylim([-5., 5.])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_title('Transmission')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

    return
            
def plot_contrast_calibrated(thrput, med_thrput, fit_thrput, con_seps, cons, corr_cons, savefile='./calcontrast.pdf'):
    '''
    Plot calibrated throughput alongside calibrated contrast curves. 
    '''
    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
    ax[0].plot(med_thrput['seps'], med_thrput['tps'], color='mediumaquamarine', label='Median throughput')
    ax[0].scatter(thrput['seps'], thrput['tps'], s=75, color='mediumaquamarine', alpha=0.5)
    ax[0].plot(fit_thrput['seps'], fit_thrput['tps'], color='teal', label='Best fit model')
    ax[0].set_xlim([fit_thrput['seps'][0], fit_thrput['seps'][-1]])
    ax[0].set_ylim([0.0, 1.2])
    ax[0].grid(axis='y')
    ax[0].set_xlabel('Separation [pix]')
    ax[0].set_ylabel('Throughput')
    ax[0].set_title('Algo & coronmsk throughput')
    ax[0].legend(loc='lower right')
    ax[1].plot(con_seps, cons, color='mediumaquamarine', label='Raw contrast')
    ax[1].plot(con_seps, corr_cons, color='teal', label='Calibrated contrast')
    ax[1].set_yscale('log')
    ax[1].set_xlim([0., np.max(con_seps)]) # arcsec
    ax[1].set_ylim(top=3e-3)
    ax[1].grid(axis='y')
    ax[1].set_xlabel('Separation [arcsec]')
    ax[1].set_ylabel('Contrast [5$\sigma$]')
    ax[1].set_title('Calibrated contrast curve')
    ax[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

    return

def plot_fm_psf(meta, fm_frame, data_frame, guess_flux, pxsc=None, j=0, savefile='./fmpsf.pdf'):
    '''
    Plot forward model psf
    '''
    
    #Set some quick information depending on whether a pixel scale was passed
    if pxsc == None:
        extent=(-0.5, fm_frame.shape[1]-0.5, fm_frame.shape[1]-0.5, -0.5)
        pxsc = 1
        xlabel, ylabel = 'Pixels', 'Pixels'
    else:
        extl = (fm_frame.shape[1]+1.)/2.*pxsc/1000. # arcsec
        extr = (fm_frame.shape[1]-1.)/2.*pxsc/1000. # arcsec
        extent = (extl, -extr, -extl, extr)
        xlabel, ylabel = '$\Delta$RA [arcsec]', '$\Delta$DEC [arcsec]'

    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
    p0 = ax[0].imshow(fm_frame*guess_flux, origin='lower', cmap='inferno', extent=extent)
    c0 = plt.colorbar(p0, ax=ax[0])
    c0.set_label('DN', rotation=270, labelpad=20)
    cc = plt.Circle((meta.ra_off[j]/1000., meta.de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
    ax[0].add_artist(cc)
    ax[0].set_xlim([5., -5])
    ax[0].set_ylim([-5., 5.])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(r'FM PSF ($\alpha$ = %.0e)' % guess_flux)
    p1 = ax[1].imshow(data_frame, origin='lower', cmap='inferno', extent=extent)
    c1 = plt.colorbar(p1, ax=ax[1])
    c1.set_label('DN', rotation=270, labelpad=20)
    cc = plt.Circle((meta.ra_off[j]/1000., meta.de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
    ax[1].add_artist(cc)
    ax[1].set_xlim([5., -5.])
    ax[1].set_ylim([-5., 5.])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_title('KLIP-subtracted')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

    return

def plot_chains(chain, savefile):
    '''
    Plot MCMC chains from companion fitting
    '''
    f, ax = plt.subplots(4, 1, figsize=(1*6.4, 2*4.8))
    ax[0].plot(chain[:, :, 0].T, color='black', alpha=1./3.)
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel(r'$\Delta$RA [mas]')
    ax[1].plot(chain[:, :, 1].T, color='black', alpha=1./3.)
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel(r'$\Delta$Dec [mas]')
    ax[2].plot(chain[:, :, 2].T, color='black', alpha=1./3.)
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel(r'$\alpha$ [sec/pri]')
    ax[3].plot(chain[:, :, 3].T, color='black', alpha=1./3.)
    ax[3].set_xlabel('Steps')
    ax[3].set_ylabel(r'$l$ [pix]')
    plt.suptitle('MCMC chains')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

def plot_subimages(imgdirs, subdirs, filts, submodes):
    '''
    Create a "publication ready" plot of the coronagraphic images, alongside
    the PSF subtracted images. 

    Parameters
    ----------

    imgdirs : list of strings
        Parent directories of the unsubtracted images, filters won't be repeated 
    subdirs : list of strings
        Parent directories of the subtracted images
    filts : list of strings
        List of filter strings to include in the plot
    '''

    # Get the files we care about
    imgfiles = list(chain.from_iterable([glob.glob(imgdir+'*') for imgdir in imgdirs]))
    subfiles = list(chain.from_iterable([glob.glob(subdir+'*') for subdir in subdirs]))
    filts = [filt.upper() for filt in filts]

    # Filter the imgfiles
    used_filts = []
    true_imgfiles = []
    for imgfile in imgfiles:
        hdr = fits.getheader(imgfile)
        if hdr['SUBPXPTS'] != 1:
            # This is a dithered reference observation, which we don't want.
            continue
        elif 'TACQ' in hdr['EXP_TYPE']:
            # This is a target acquisition image
            continue
        elif hdr['FILTER'] not in filts:
            # We don't want this filter
            continue
        elif hdr['FILTER'] in used_filts:
            # We've already got a file for this filter
            continue
        else:
            # We want this file
            true_imgfiles.append(imgfile)
            used_filts.append(hdr['FILTER'])

    # Filter the subfiles
    # Note that we allow repeat filters for different reductions.
    true_subfiles = []
    for subfile in subfiles:
        if any(filt in subfile for filt in filts):
            true_subfiles.append(subfile)

    ydim = len(true_imgfiles)
    xdim = int(len(true_subfiles) / ydim) + 1

    # Start making the figure
    fig = plt.figure(figsize=[xdim*5, ydim*5])
    grid = gs.GridSpec(ydim, xdim, figure=fig)

    for i, imgfile in enumerate(true_imgfiles):
        hdr = fits.getheader(imgfile)
        row = filts.index(hdr['FILTER'])

        ax = plt.subplot(grid[row, 0])
        with fits.open(imgfile) as hdul:
            img = hdul['SCI'].data[-1]
        ax.imshow(img)

    # for i, subfile in enumerate(true_subfiles):

    # plt.show()

    return