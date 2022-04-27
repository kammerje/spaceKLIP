import os, sys
sys.path.append('..')

from spaceKLIP.engine import JWST

config_file = os.path.dirname(__file__)+'/example_config.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=False, 
				 skip_imgproc=False, 
				 skip_sub=False, 
				 skip_rawcon=False, 
				 skip_calcon=False, 
				 skip_comps=False)