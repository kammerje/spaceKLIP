import os, sys
sys.path.append('..')

from spaceKLIP.engine import JWST

config_file = os.path.dirname(__file__)+'/hr8799_config.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run()
