import os, sys
sys.path.append('..')
from spaceKLIP.engine import JWST
import spaceKLIP as sklip

# config_file = os.path.dirname(__file__)+'/nircam_config.yaml'
# config_file = os.path.dirname(__file__)+'/miri_config.yaml'
config_file = os.path.dirname(__file__)+'/nircam_config_spring_symposium_ers.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=True,
				 skip_imgproc=True,
				 skip_sub=False,
				 skip_rawcon=True,
				 skip_calcon=True,
				 skip_comps=False)
