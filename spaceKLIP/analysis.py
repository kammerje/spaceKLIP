from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import os, re, sys
import json

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from . import io
from . import utils
from . import plotting



rad2mas = 180./np.pi*3600.*1000.

# =============================================================================
# MAIN
# =============================================================================


def analyze_companions(meta):
    """
    Analyze astrometry and photometry from extracted companions using
    orbitize and species.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    """

    if (meta.verbose == True):
        print('--> Analyzing companion properties...')

    try:
        import species
    except:
        print('species is not installed! please try $pip install species to use the analysis module')

    # todo, add meta to specify where to init species
    species.SpeciesInit()
    database = species.Database()

    # add host photometry to fit stellar model
    magnitudes = {}
    for phot in meta.host_photometry:
        filt = phot[0]
        mag = (phot[1][0], phot[1][1])
        magnitudes[filt] = mag

    filters = list(magnitudes.keys())

    plx = (meta.system_plx[0],meta.system_plx[1])

    # add host object
    database.delete_data('objects/'+meta.host_name)
    database.add_object(object_name=meta.host_name,
                        parallax=plx,
                        app_mag=magnitudes,
                       )

    # add model stellar atmospheres
    teff_range = (meta.host_teff[0],meta.host_teff[1])
    logg_range = (meta.host_logg[0],meta.host_logg[1])
    feh_range = (meta.host_feh[0],meta.host_feh[1])
    radius_range = (meta.host_radius[0],meta.host_radius[1])
    database.add_model(meta.host_model, teff_range=teff_range)

    # create fitmodel object for host
    fit = species.FitModel(object_name=meta.host_name,
                           model=meta.host_model,
                           bounds={'teff': teff_range,
                                   'logg': logg_range,
                                   'feh': feh_range,
                                   'radius': radius_range,
                                   },
                           inc_phot=True,
                           inc_spec=False)

    # fit model to host with ultranest
    fit.run_ultranest(tag=meta.host_tag,
                      min_num_live_points=300,
                      output='ultranest/',
                      prior=None)

    read_model = species.ReadModel(model='bt-nextgen', wavel_range=(0.2, 30.))
    host_model_box = read_model.get_model(model_param=median, spec_res=500., smooth=True)

    # get result dictionary
    res = meta.extracted_comps

    # loop through companions
    for j in range(len(meta.ra_off)):
        for i, key in enumerate(meta.obs.keys()):
            temp = 'c%.0f' % (j+1)
            ra = res[key][temp]['ra']
            ra_e = res[key][temp]['dra']
            dec = res[key][temp]['de']
            dec_e = res[key][temp]['dde']
            con = res[key][temp]['f']
            con_e = res[key][temp]['df']
            contrast = (con, con_e)

            filt = meta.filter[key]
            print('filter is: ',filt)

            comp_magnitudes = {}
            for phot in meta.comp_photometry:
                filt = phot[0]
                mag = (phot[1][0], phot[1][1])
                comp_magnitudes[filt] = mag

            comp_magnitudes[filt] = contrast_to_app(contrast, host_model_box)

            filters = list(comp_magnitudes.keys())

            database.delete_data('objects/'+meta.comp_names[j])
            database.add_object(meta.comp_names[j],
                                parallax=meta.system_plx,
                                app_mag=comp_magnitudes,
                                spectrum=None,
                                deredden=None)

            teff_range = (meta.comp_teff[0],meta.comp_teff[1])
            database.add_model(model=meta.comp_model, teff_range=teff_range)

            fit = species.FitModel(object_name=meta.comp_names[j],
                                   model=meta.comp_model,
                                   bounds={'teff': teff_range,
                                           # 'radius': (0.5, 2.),
                                           # 'SPHERE_IFU': ((0.5, 1.5), None)
                                          },
                                   inc_phot=filters,
                                   inc_spec=False,
                                   fit_corr=None,
                                   weights=None)

            objectbox = database.get_object(object_name=meta.comp_names[j],
                                            inc_phot=True,
                                            inc_spec=True)

            species.plot_spectrum(boxes=[objectbox],
                                  filters=objectbox.filters,
                                  residuals=None,
                                  plot_kwargs=[{
                                                'JWST/NIRCAM.F335M':{'marker': 's', 'markersize': 6., 'color': 'tomato', 'mfc':'white', 'ls': 'none'},
                                                }
                                                ],
                                  xlim=(0.8, 20.),
                                  # ylim=(-1.15e-17, 1e-12),
                                  # ylim_res=(-4.9, 4.9),
                                  scale=('log', 'log'),
                                  offset=(-0.06, -0.06),
                                  legend=[{'loc': 'lower left', 'frameon': False, 'fontsize': 11.},
                                          {'loc': 'upper right', 'frameon': False, 'fontsize': 11.}],
                                  figsize=(12., 6.),
                                  quantity='flux density',
                                  output=odir+key+'-flux_cal_SED_c%.0f' % (j+1)+'.pdf')


    return None

def contrast_to_app(contrast, host_model, filt='JWST/NIRCAM.F335M'):

    mag = -2.5*np.log10(contrast[0])
    magerr = 2.5*1/np.log(10)*contrast[1]/contrast[0]

    synphot = species.SyntheticPhotometry(filt)
    stellar_mag, _ = synphot.spectrum_to_magnitude(host_model.wavelength,host_model.flux)
    mag = (mag + stellar_mag[0], magerr)

    return mag

# def assemble_photometry(meta):
