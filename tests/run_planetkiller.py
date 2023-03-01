import os, sys
import numpy as np
sys.path.append('..')
from spaceKLIP.engine import JWST
import spaceKLIP as sklip

config_file = os.path.dirname(__file__)+'/nircam_config_hr8799.yaml'
#config_file = os.path.dirname(__file__)+'/miri_config.yaml'

ra = -253.
dec = 307.
pxsc = 63.
guess_dx = ra/pxsc # pix
guess_dy = dec/pxsc # pix
guess_sep = np.sqrt(guess_dx**2+guess_dy**2) # pix
guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy)) # deg

small_planet = {'sep':guess_sep,
				'pa':guess_pa,
				'flux':1e-5}

if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=True,
				 skip_imgproc=True,
				 skip_sub=True,
				 skip_rawcon=True,
				 skip_calcon=True,
				 skip_comps=True)
	sklip.companion.planet_killer(pipe.meta, small_planet=small_planet)
