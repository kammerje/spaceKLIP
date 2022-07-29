import glob, os

import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import font_manager

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
    #ax.set_xlim([0., 5.]) # arcsec
    ax.grid(axis='y')
    ax.set_xlabel('Separation [arcsec]')
    ax.set_ylabel('Contrast [5$\sigma$]')
    ax.set_title('Raw contrast curve')
    ax.legend(loc='upper right', prop=dict(size=8))
    plt.tight_layout()

    # Save and close plot
    plt.savefig(savefile)
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
    ax[0].set_xlim([5., -5.])
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

def plot_subimages(imgdirs, subdirs, filts, submodes, numKL, 
                   window_size=2.5, cmaps_list=['viridis'],
                   imgVmin=[-40], imgVmax=[40], subVmin=[-10], subVmax=[10]):
    '''
    Create a "publication ready" plot of the coronagraphic images, alongside
    the PSF subtracted images. A grid of images will be made. Rows will correspond to 
    filters, first column the unsubtracted PSF, following columns different
    submodes and numKLs

    Parameters
    ----------

    imgdirs : list of strings
        Parent directories of the unsubtracted images, filters won't be repeated 
    subdirs : list of strings
        Parent directories of the subtracted images
    filts : list of strings
        List of filter strings to include in the plot
    submodes : list of strings
        'ADI', 'RDI', 'RDI+ADI' (or 'ADI+RDI')
    numKL : list of ints
        output images are 3-D with the third axis corresponding to different KL
        modes used in subtraction. Indicate the number of KL modes you want to
        display, if more than one, display each for each submode on the same row .
    window_size : float
        the number of arcseconds on a side
    cmaps_list : list
        a list of strings naming the colormaps for each filter. If len(cmaps_list)==1
        the same cmap will be used for each filter.
    imgVmin : list
        a list of the min values for the the cmaps for each filter. if len(imgVmin)==1
        the same min value will be used for each filter.
    imgVmax : list
        a list of the max values for the the cmaps for each filter. if len(imgVmax)==1
        the same max value will be used for each filter.
    subVmin : list
        a list of the min values for the the subtracted image cmaps for each filter. 
        if len(imgVmin)==1 the same min value will be used for each filter.
    subVmax : list
        a list of the max values for the the subtracted image cmaps for each filter. 
        if len(imgVmin)==1 the same min value will be used for each filter.
    '''

    # Get the files we care about
    imgfiles = sorted(list(chain.from_iterable([glob.glob(imgdir+'*.fits') for imgdir in imgdirs])))
    subfiles = list(chain.from_iterable([glob.glob(subdir+'*.fits') for subdir in subdirs]))
    filts = [filt.upper() for filt in filts]

    # Filter the imgfiles
    used_filts = []
    true_imgfiles = {}
    for imgfile in imgfiles:
        hdr = fits.getheader(imgfile)
        if hdr.get('SUBPXPTS') is not None:
            # This keyword only exists when larger than 1. If so,
            # this is a dithered reference observation, which we don't want.
            continue
        elif (hdr['EXP_TYPE'] != 'NRC_CORON') and (hdr['EXP_TYPE'] != 'MIR_4QPM'):
            # Don't want TACONFIRM or TACQ
            continue
        elif hdr['FILTER'] not in filts:
            # We don't want this filter
            continue
        elif hdr['FILTER'] in used_filts:
            # We've already got a file for this filter
            continue
        else:
            # We want this file
            print(imgfile)
            true_imgfiles[hdr['FILTER']] = imgfile
            used_filts.append(hdr['FILTER'])

    # Filter the subfiles
    # Note that we allow repeat filters for different reductions.
    true_subfiles = []
    for subfile in subfiles:
        print(subfile)
        if any(filt in subfile for filt in filts):
            true_subfiles.append(subfile)


    #read the subtracted images, store relevant info
    sub_dict = {}
    for flt in filts:
        for fn in true_subfiles:
            print('subfile: ', fn)
            if flt in fn:
                with fits.open(fn) as hdul:
                    imgcube = hdul[0].data
                    psfparam = hdul[0].header['PSFPARAM']
                    center_pix = (int(np.rint(hdul[0].header['PSFCENTY'])), 
                                  int(np.rint(hdul[0].header['PSFCENTX'])))
                mode = psfparam.split('mode=')[-1].split(',')[0]#ADI/RDI/ADI+RDI
                if '+' in mode:
                    reverse_mode = mode.split('+')[-1] + '+' + mode.split('+')[0]
                else:
                    reverse_mode = None
                numbasis = [int(KLnum) for KLnum in 
                      psfparam.split('numbasis=')[-1].split(',')[0].split(']')[0].split('[')[-1].split()]
                print(imgcube.shape, len(numbasis))
                if flt in sub_dict.keys():
                    sub_dict[flt][mode] = {'image':imgcube, 'numbasis':numbasis, 
                                           'filename':fn, 'center_pix':center_pix}
                    if reverse_mode is not None:
                        sub_dict[flt][reverse_mode] ={'image':imgcube, 'numbasis':numbasis, 
                                              'filename':fn, 'center_pix':center_pix}


                else:
                    sub_dict[flt] = {mode:{'image':imgcube, 'numbasis':numbasis, 
                                           'filename':fn, 'center_pix':center_pix}}
                    if reverse_mode is not None:
                        sub_dict[flt][reverse_mode] = {'image':imgcube, 'numbasis':numbasis, 
                                                       'filename':fn, 'center_pix':center_pix}

    #miri centers from pyklip v2.6 (y, x)
    miri_img_centers = {'F1065C': (int(np.rint(111.89-5)), int(np.rint(120.81-14))),
                        'F1140C': (int(np.rint(112.2-7)) , int(np.rint(119.99-13))) ,
                        'F1550C': (int(np.rint(113.33-8)), int(np.rint(119.84-13))),  }
    pltscale = {'NIRCAM': 0.063, 'MIRI': 0.11}
    plot_extent = (-1*window_size/2, window_size/2, -1*window_size/2, window_size/2)

    ydim = len(filts)
    xdim = len(submodes)*len(numKL) + 1

    # Start making the figure
    fig = plt.figure(figsize=[xdim*3, ydim*3])
    grid = gs.GridSpec(ydim, xdim, figure=fig)

    if len(cmaps_list) == 1:
        cmaps_list *= len(filts)
    if len(imgVmin) == 1:
        imgVmin *= len(filts)
    if len(imgVmax) == 1:
        imgVmax *= len(filts)
    if len(subVmin) == 1:
        subVmin *= len(filts)
    if len(subVmax) == 1:
        subVmax *= len(filts)

    for row, flt in enumerate(filts):
        prihdr = fits.getheader(true_imgfiles[flt])
        instrument = prihdr['INSTRUME']
        ax = plt.subplot(grid[row, 0])
        with fits.open(true_imgfiles[flt]) as hdul:
            img = hdul['SCI'].data[-1]
            hdr = hdul['SCI'].header
        if instrument == 'NIRCAM':
            center_pix = (int(np.rint(hdr['CRPIX2'])), 
                          int(np.rint(hdr['CRPIX1'])))
            window_pix = int(np.rint(window_size / pltscale['NIRCAM'] / 2))
        else:
            center_pix = miri_img_centers[flt][::-1]
            window_pix = int(np.rint(window_size / pltscale['MIRI'] / 2))
        print(center_pix)
        focus_slices_img = (slice(center_pix[0] - window_pix, center_pix[0] + window_pix),
                            slice(center_pix[1] - window_pix, center_pix[1] + window_pix))

        ax.imshow(img[focus_slices_img], extent=plot_extent, cmap=cmaps_list[row], vmin=imgVmin[row], vmax=imgVmax[row])
        ax.set_ylabel(prihdr['FILTER']+'\nDec offset [arcsec]', fontproperties=compModern(14))
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3, prune='both'))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        if row != (len(filts)-1):
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('RA offset [arcsec]', fontproperties=compModern(14))
            tickFont(ax, 'x', fontproperties=compModern(10))
        tickFont(ax, 'y', fontproperties=compModern(10))
        ax.tick_params(which='major', length=4)
        ax.tick_params(which='minor', length=2)

        for mm, mde in enumerate(submodes):
            focus_slices_sub = (slice(sub_dict[flt][mde]['center_pix'][0] - window_pix, 
                                      sub_dict[flt][mde]['center_pix'][0] + window_pix),
                                slice(sub_dict[flt][mde]['center_pix'][1] - window_pix, 
                                      sub_dict[flt][mde]['center_pix'][1] + window_pix))
            for kk, nkl in enumerate(numKL):
                column = 1 + (len(numKL)*mm + kk)
                ax = plt.subplot(grid[row, column])
                ax.imshow(sub_dict[flt][mde]['image'][sub_dict[flt][mde]['numbasis'].index(nkl),:,:][focus_slices_sub], 
                          extent=plot_extent, cmap=cmaps_list[row], vmin=subVmin[row], vmax=subVmax[row])
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(1))
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(which='major', length=4)
                ax.tick_params(which='minor', length=2)
                if row != (len(filts)-1):
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    #ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3, prune='both'))
                    tickFont(ax, 'x', fontproperties=compModern(10))
                    ax.set_xlabel('RA offset [arcsec]', fontproperties=compModern(14))
                if row == 0:
                    ax.set_title('{:s} {:d} modes'.format(mde, nkl), fontproperties=compModern(16))
    fig.tight_layout()
    fig.tight_layout()
    fig.tight_layout()
    plt.show()
    return fig


def tickFont(ax, xy, fontproperties):
    if xy == 'x':
        plt.setp(ax.get_xticklabels(), fontproperties=fontproperties)
    if xy == 'y':
        plt.setp(ax.get_yticklabels(), fontproperties=fontproperties)


def compModern(size=20):
    computerModern = font_manager.FontProperties(size=size,
                                            fname=os.path.join(os.path.dirname(__file__),
                                            'cmunrm.ttf'))
    return computerModern

