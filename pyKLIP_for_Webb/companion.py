    def extract_companions(self,
                           mstar, # vegamag
                           ra_off=ra_off, # mas
                           de_off=de_off,
                           KL=-1, # mas
                           overwrite=False):
        """
        Extract astrometry and photometry from any detected companions using
        the pyKLIP forward modeling class.
        
        TODO: tries to convert MJy/sr to contrast using the host star
              magnitude, but the results are wrong by a factor of ~5.
        
        TODO: until RDI is implemented for forward modeling, simply use the
              offset PSF with the correct coronagraphic mask transmission
              applied.
        
        TODO: use a position dependent offset PSF from pyNRC instead of the
              completely unocculted offset PSF from WebbPSF.
        
        Parameters
        ----------
        mstar: dict of float
            Host star magnitude in each filter. Must contain one entry for
            each filter used in the data in the input directory.
        ra_off: list of float
            RA offset of the known companions in the same order as in the
            NIRCCoS config file.
        de_off: list of float
            DEC offset of the known companions in the same order as in the
            NIRCCoS config file.
        KL: int
            Index of the KL component for which the calibrated contrast curve
            and the companion properties shall be computed.
        overwrite: bool
            If true overwrite existing data.
        """
        
        if (self.verbose == True):
            print('--> Extracting companion properties...')
        
        # Loop through all modes, numbers of annuli, and numbers of
        # subsections.
        Nscenarios = len(self.mode)*len(self.annuli)*len(self.subsections)
        counter = 1
        for mode in self.mode:
            for annuli in self.annuli:
                for subsections in self.subsections:
                    
                    # Define the input and output directories for each set of
                    # pyKLIP parameters.
                    idir = self.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                    odir = self.odir+mode+'_annu%.0f_subs%.0f/FLUX/' % (annuli, subsections)
                    if (not os.path.exists(odir)):
                        os.makedirs(odir)
                    
                    # Create an output directory for the forward modeled datasets.
                    odir_temp = odir+'FITS/'
                    if (not os.path.exists(odir_temp)):
                        os.makedirs(odir_temp)
                    
                    # Loop through all sets of observing parameters.
                    res = {}
                    for i, key in enumerate(self.obs.keys()):
                        ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
                        filepaths = np.array(self.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                        ww_cal = np.where(self.obs[key]['TYP'] == 'CAL')[0]
                        psflib_filepaths = np.array(self.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                        hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
                        data = hdul[0].data
                        pxsc = self.obs[key]['PIXSCALE'][0] # mas
                        cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
                        temp = [s.start() for s in re.finditer('_', key)]
                        filt = key[temp[1]+1:temp[2]]
                        mask = key[temp[3]+1:temp[4]]
                        subarr = key[temp[4]+1:]
                        wave = self.wave[filt] # m
                        fwhm = wave/self.diam*rad2mas/pxsc # pix
                        hdul.close()
                        
                        # Create a new pyKLIP dataset for forward modeling the
                        # companion PSFs.
                        dataset = JWST.JWSTData(filepaths=filepaths,
                                                psflib_filepaths=psflib_filepaths)
                        
                        # 2D map of the total throughput, i.e., an integration
                        # time weighted average of the coronmsk transmission
                        # over the rolls.
                        self.get_transmission(pxsc, filt, mask, subarr, odir, key)
                        
                        # Offset PSF from WebbPSF, i.e., an integration time
                        # weighted average of the unocculted offset PSF over
                        # the rolls (does account for pupil mask throughput).
                        offsetpsf = self.get_offsetpsf(filt, mask, key)
                        offsetpsf *= self.F0[filt]/10.**(mstar[filt]/2.5)/1e6/pxsc**2*(180./np.pi*3600.*1000.)**2 # MJy/sr
                        
                        # Loop through all companions.
                        res[key] = {}
                        for j in range(len(ra_off)):
                            
                            # Guesses for the fit parameters.
                            guess_dx = ra_off[j]/pxsc # pix
                            guess_dy = de_off[j]/pxsc # pix
                            guess_sep = np.sqrt(guess_dx**2+guess_dy**2) # pix
                            guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy)) # deg
                            guess_flux = 1e-4
                            guess_spec = np.array([1.])
                            
                            # If overwrite is false, check whether the forward
                            # modeled datasets have been computed already.
                            fmdataset = odir_temp+'FM_C%.0f-' % (j+1)+key+'-fmpsf-KLmodes-all.fits'
                            klipdataset = odir_temp+'FM_C%.0f-' % (j+1)+key+'-klipped-KLmodes-all.fits'
                            if ((overwrite == True) or ((not os.path.exists(fmdataset)) or (not os.path.exists(klipdataset)))):
                                
                                # Initialize the forward modeling pyKLIP class.
                                input_wvs = np.unique(dataset.wvs)
                                if (len(input_wvs) != 1):
                                    raise UserWarning('Only works with broadband photometry')
                                fm_class = fmpsf.FMPlanetPSF(inputs_shape=dataset.input.shape,
                                                             numbasis=np.array(self.truenumbasis[key]),
                                                             sep=guess_sep,
                                                             pa=guess_pa,
                                                             dflux=guess_flux,
                                                             input_psfs=np.array([offsetpsf]),
                                                             input_wvs=input_wvs,
                                                             spectrallib=[guess_spec],
                                                             spectrallib_units='contrast',
                                                             field_dependent_correction=self.correct_transmission)
                                
                                # Compute the forward modeled datasets.
                                annulus = [[guess_sep-20., guess_sep+20.]] # pix
                                subsection = 1
                                fm.klip_dataset(dataset=dataset,
                                                fm_class=fm_class,
                                                mode=mode,
                                                outputdir=odir_temp,
                                                fileprefix='FM_C%.0f-' % (j+1)+key,
                                                annuli=annulus,
                                                subsections=subsection,
                                                movement=1,
                                                numbasis=self.truenumbasis[key],
                                                maxnumbasis=self.maxnumbasis[key],
                                                calibrate_flux=False,
                                                psf_library=dataset.psflib,
                                                highpass=False,
                                                mute_progression=True)
                            
                            # Open the forward modeled datasets.
                            with pyfits.open(fmdataset) as hdul:
                                fm_frame = hdul[0].data[KL]
                                fm_centx = hdul[0].header['PSFCENTX']
                                fm_centy = hdul[0].header['PSFCENTY']
                            with pyfits.open(klipdataset) as hdul:
                                data_frame = hdul[0].data[KL]
                                data_centx = hdul[0].header["PSFCENTX"]
                                data_centy = hdul[0].header["PSFCENTY"]
                            
                            # TODO: until RDI is implemented for forward
                            # modeling, simply use the offset PSF with the
                            # correct coronagraphic mask transmission applied.
                            if (mode == 'RDI'):
                                fm_frame = np.zeros_like(fm_frame)
                                if ((fm_centx % 1. != 0.) or (fm_centy % 1. != 0.)):
                                    raise UserWarning('Requires forward modeled PSF with integer center')
                                else:
                                    fm_centx = int(fm_centx)
                                    fm_centy = int(fm_centy)
                                sx, sy = offsetpsf.shape
                                if ((sx % 2 != 1) or (sy % 2 != 1)):
                                    raise UserWarning('Requires offset PSF with odd shape')
                                temp = shift(offsetpsf.copy(), (guess_dy-int(guess_dy), -(guess_dx-int(guess_dx))))
                                rx = np.arange(sx)-sx//2+guess_dx
                                ry = np.arange(sy)-sy//2+guess_dy
                                xx, yy = np.meshgrid(rx, ry)
                                temp = self.correct_transmission(temp, xx, yy)
                                fm_frame[fm_centy+int(guess_dy)-sy//2:fm_centy+int(guess_dy)+sy//2+1, fm_centx-int(guess_dx)-sx//2:fm_centx-int(guess_dx)+sx//2+1] = temp # scale forward modeled PSF similar as in pyKLIP
                                
                                # # FIXME!
                                # # Uncomment the following to use the PSF
                                # # that was injected by pyNRC.
                                # test = []
                                # totet = 0. # s
                                # for ii in range(len(ww_sci)):
                                #     inttm = self.obs[key]['NINTS'][ww_sci[ii]]*self.obs[key]['EFFINTTM'][ww_sci[ii]] # s
                                #     test += [inttm*rotate(np.load('../HIP65426/pynrc_figs/seq_000_filt_'+filt+'_psfs_%+04.0fdeg.npy' % self.obs[key]['PA_V3'][ww_sci[ii]])[j], -self.obs[key]['PA_V3'][ww_sci[ii]], reshape=False, mode='constant', cval=0.)]
                                #     totet += inttm # s
                                # test = np.sum(np.array(test), axis=0)/totet
                                # test *= self.F0[filt]/10.**(mstar[filt]/2.5)/1e6/pxsc**2*(180./np.pi*3600.*1000.)**2 # MJy/sr
                                # fm_frame[fm_centy+int(guess_dy)-sy//2:fm_centy+int(guess_dy)+sy//2+1, fm_centx-int(guess_dx)-sx//2:fm_centx-int(guess_dx)+sx//2+1] = test # scale forward modeled PSF similar as in pyKLIP
                                
                            elif ('RDI' in mode):
                                raise UserWarning('Not implemented yet')
                            
                            # Plot.
                            extl = (fm_frame.shape[1]+1.)/2.*pxsc/1000. # arcsec
                            extr = (fm_frame.shape[1]-1.)/2.*pxsc/1000. # arcsec
                            f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                            p0 = ax[0].imshow(fm_frame*guess_flux, origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                            c0 = plt.colorbar(p0, ax=ax[0])
                            c0.set_label('DN', rotation=270, labelpad=20)
                            cc = plt.Circle((ra_off[j]/1000., de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[0].add_artist(cc)
                            ax[0].set_xlim([5., -5.])
                            ax[0].set_ylim([-5., 5.])
                            ax[0].set_xlabel('$\Delta$RA [arcsec]')
                            ax[0].set_ylabel('$\Delta$DEC [arcsec]')
                            ax[0].set_title(r'FM PSF ($\alpha$ = %.0e)' % guess_flux)
                            p1 = ax[1].imshow(data_frame, origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                            c1 = plt.colorbar(p1, ax=ax[1])
                            c1.set_label('DN', rotation=270, labelpad=20)
                            cc = plt.Circle((ra_off[j]/1000., de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[1].add_artist(cc)
                            ax[1].set_xlim([5., -5.])
                            ax[1].set_ylim([-5., 5.])
                            ax[1].set_xlabel('$\Delta$RA [arcsec]')
                            ax[1].set_ylabel('$\Delta$DEC [arcsec]')
                            ax[1].set_title('KLIP-subtracted')
                            plt.tight_layout()
                            plt.savefig(odir+key+'-fmpsf_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Fit the forward modeled PSF to the
                            # KLIP-subtracted data.
                            fitboxsize = 17 # pix
                            fma = fitpsf.FMAstrometry(guess_sep=guess_sep,
                                                      guess_pa=guess_pa,
                                                      fitboxsize=fitboxsize)
                            fma.generate_fm_stamp(fm_image=fm_frame,
                                                  fm_center=[fm_centx, fm_centy],
                                                  padding=5)
                            fma.generate_data_stamp(data=data_frame,
                                                    data_center=[data_centx, data_centy],
                                                    dr=4,
                                                    exclusion_radius=12.*fwhm)
                            corr_len_guess = 3. # pix
                            corr_len_label = r'$l$'
                            fma.set_kernel('matern32', [corr_len_guess], [corr_len_label])                        
                            x_range = 1. # pix
                            y_range = 1. # pix
                            flux_range = 1. # mag
                            corr_len_range = 1. # mag
                            fma.set_bounds(x_range, y_range, flux_range, [corr_len_range])
                            
                            # Make sure that noise map is invertible.
                            noise_map_max = np.nanmax(fma.noise_map)
                            fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
                            fma.noise_map[fma.noise_map == 0.] = noise_map_max
                            
                            fma.fit_astrometry(nwalkers=100, nburn=200, nsteps=800, numthreads=2)
                            
                            # Plot.
                            chain = fma.sampler.chain
                            f, ax = plt.subplots(4, 1, figsize=(1*6.4, 2*4.8))
                            ax[0].plot(chain[:, :, 0].T, color='black', alpha=1./3.)
                            ax[0].set_xlabel('Steps')
                            ax[0].set_ylabel(r'$\Delta$RA [pix]')
                            ax[1].plot(chain[:, :, 1].T, color='black', alpha=1./3.)
                            ax[1].set_xlabel('Steps')
                            ax[1].set_ylabel(r'$\Delta$DEC [pix]')
                            ax[2].plot(chain[:, :, 2].T, color='black', alpha=1./3.)
                            ax[2].set_xlabel('Steps')
                            ax[2].set_ylabel(r'$\alpha$')
                            ax[3].plot(chain[:, :, 3].T, color='black', alpha=1./3.)
                            ax[3].set_xlabel('Steps')
                            ax[3].set_ylabel(r'$l$ [pix]')
                            plt.suptitle('MCMC chains')
                            plt.tight_layout()
                            plt.savefig(odir+key+'-chains_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Plot.
                            fma.make_corner_plot()
                            plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Plot.
                            fma.best_fit_and_residuals()
                            plt.savefig(odir+key+'-model_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Write best fit values into results dictionary.
                            temp = 'c%.0f' % (j+1)
                            res[key][temp] = {}
                            res[key][temp]['ra'] = fma.raw_RA_offset.bestfit*pxsc # mas
                            res[key][temp]['dra'] = fma.raw_RA_offset.error*pxsc # mas
                            res[key][temp]['de'] = fma.raw_Dec_offset.bestfit*pxsc # mas
                            res[key][temp]['dde'] = fma.raw_Dec_offset.error*pxsc # mas
                            res[key][temp]['f'] = fma.raw_flux.bestfit
                            res[key][temp]['df'] = fma.raw_flux.error
                            
                            if (self.verbose == True):
                                print('Companion %.0f' % (j+1))
                                print('   RA  = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['ra'], res[key][temp]['dra'], ra_off[j]))
                                print('   DEC = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['de'], res[key][temp]['dde'], de_off[j]))
                                try:
                                    condir = self.idir[:self.idir.find('/')]+'/pynrc_figs/'
                                    confiles = [f for f in os.listdir(condir) if filt in f and f.endswith('_cons.npy')]
                                    if (len(confiles) != 1):
                                        raise UserWarning()
                                    else:
                                        con = np.load(condir+confiles[0])[j]
                                    print('   CON = %.2e+/-%.2e (%.2e inj.)' % (res[key][temp]['f'], res[key][temp]['df'], con))
                                except:
                                    print('   CON = %.2e+/-%.2e' % (res[key][temp]['f'], res[key][temp]['df']))
        
        return res