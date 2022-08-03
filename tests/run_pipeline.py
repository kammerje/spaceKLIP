import os, sys
sys.path.append('..')
from spaceKLIP.engine import JWST
import spaceKLIP as sklip

# config_file = os.path.dirname(__file__)+'/HIP65426-F1550.yaml'
config_file = os.path.dirname(__file__)+'/HIP65426-F1140.yaml'
# config_file = os.path.dirname(__file__)+'/HD114174-round.yaml'

if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=True,
				 skip_imgproc=False,
				 skip_sub=False,
				 skip_rawcon=True,
				 skip_calcon=True,
				 skip_comps=True)


# imgdir = '/Users/wbalmer/JWST-HCI/HIP65426/MIRI/F1140C/IMGPROCESS/'
# subdir = '/Users/wbalmer/JWST-HCI/HIP65426/MIRI/F1140C/2022_06_28_RDI_annu1_subs1_run1/SUBTRACTED/'

#sklip.plotting.plot_subimages([imgdir], [subdir], ['F250M', 'F444W'], ['RDI+ADI'])
