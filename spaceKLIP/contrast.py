from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from pyklip import klip
from spaceKLIP import utils as ut

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class Contrast():
    """
    The spaceKLIP contrast estimation class.
    """
    
    def __init__(self,
                 Database):
        
        # Make an internal alias of the spaceKLIP database class.
        self.Database = Database
        
        pass
    
    def raw_contrast(self,
                     subdir='rawcon'):
        
        # Set output directory.
        output_dir = os.path.join(self.Database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Loop through concatenations.
        for i, key in enumerate(self.Database.red.keys()):
            log.info('--> Concatenation ' + key)
            
            # Loop through FITS files.
            Nfitsfiles = len(self.Database.red[key])
            for j in range(Nfitsfiles):
                
                # Read FITS file.
                fitsfile = self.Database.red[key]['FITSFILE'][j]
                data, head_pri, head_sci, is2d = ut.read_red(fitsfile)
                
                # Compute raw contrast.
                seps = []
                cons = []
                iwa = 1
                owa = data.shape[1] // 2
                pxsc_rad = self.Database.red[key]['PIXSCALE'][j] / 1000. / 3600. / 180. * np.pi
                resolution = 1e-6 * self.Database.red[key]['CWAVEL'][j] / 6. / pxsc_rad
                center = (head_pri['PSFCENTX'], head_pri['PSFCENTY'])
                for k in range(data.shape[0]):
                    sep, con = klip.meas_contrast(dat=data[k], iwa=iwa, owa=owa, resolution=resolution, center=center, low_pass_filter=False)
                    seps += [sep * self.Database.red[key]['PIXSCALE'][j] / 1000.] # arcsec
                    cons += [con]
                seps = np.array(seps)
                cons = np.array(cons)
                
                # Plot raw contrast.
                klmodes = self.Database.red[key]['KLMODES'][j].split(',')
                fitsfile = os.path.join(output_dir, os.path.split(fitsfile)[1])
                f = plt.figure(figsize=(6.4, 4.8))
                ax = plt.gca()
                for k in range(data.shape[0]):
                    ax.plot(seps[k], cons[k], label=klmodes[k] + ' KL')
                ax.set_yscale('log')
                ax.set_xlabel('Separation [arcsec]')
                ax.set_ylabel(r'5-$\sigma$ contrast')
                ax.legend(loc='upper right')
                ax.set_title('Raw contrast')
                plt.tight_layout()
                plt.savefig(fitsfile[:-5] + '_rawcon.pdf')
                # plt.show()
                plt.close()
                np.save(fitsfile[:-5] + '_seps.npy', seps)
                np.save(fitsfile[:-5] + '_cons.npy', cons)
        
        pass
