import pyKLIP_for_Webb
from pyKLIP_for_Webb.engine import JWSTReduction

pipe = JWSTReduction('config.yaml')
if __name__ == '__main__':
	pipe.run()
	
