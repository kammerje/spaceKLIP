import sys
sys.path.append('..')

import pyKLIP_for_Webb
from pyKLIP_for_Webb.engine import JWSTPipeline

pipe = JWSTPipeline('config.yaml')
if __name__ == '__main__':
	pipe.run()
	
