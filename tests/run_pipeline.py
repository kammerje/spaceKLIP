import os, sys
sys.path.append('../')

import spaceKLIP
from spaceKLIP.engine import JWST

config_file = os.path.dirname(__file__)+'/example_config.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run()
		
