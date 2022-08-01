import os, sys
sys.path.append('..')
from spaceKLIP.engine import JWST
import spaceKLIP as sklip

config_file = os.path.dirname(__file__)+'/example_config.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=True, 
				 skip_imgproc=True, 
				 skip_sub=True, 
				 skip_rawcon=False, 
				 skip_calcon=True, 
				 skip_comps=True)


# imgdir = '/Users/acarter/Documents/DIRECT_IMAGING/CORONAGRAPHY_PIPELINE/20220628_NIRCam/IMGPROCESS/'
# subdir = '/Users/acarter/Documents/DIRECT_IMAGING/CORONAGRAPHY_PIPELINE/20220628_NIRCam/2022_06_28_RDI_annu1_subs1_run1/SUBTRACTED/'

#sklip.plotting.plot_subimages([imgdir], [subdir], ['F250M', 'F444W'], ['RDI'])
