import os, sys
sys.path.append('..')
from spaceKLIP.engine import JWST
import spaceKLIP as sklip

config_file = os.path.dirname(__file__)+'/nircam_config_hip65426b.yaml'
#config_file = os.path.dirname(__file__)+'/miri_config.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=True,
				 skip_imgproc=True,
				 skip_sub=False, 
				 skip_rawcon=True,
				 skip_calcon=True,
				 skip_comps=False)


# imgdir = '/Users/wbalmer/JWST-HCI/HIP65426/MIRI/F1140C/IMGPROCESS/'
# subdir = '/Users/wbalmer/JWST-HCI/HIP65426/MIRI/F1140C/2022_06_28_RDI_annu1_subs1_run1/SUBTRACTED/'

#sklip.plotting.plot_subimages([imgdir], [subdir], ['F250M', 'F444W'], ['RDI+ADI'])
