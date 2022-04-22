import os, glob
from jwst.pipeline.calwebb_image2 import Image2Pipeline

from . import utils
from . import io

def stsci_image_processing(meta):
	"""
	Use the JWST pipeline to process *rateints.fits files to *calints.fits files
	"""

	files = io.get_working_files(meta, meta.do_rampfit, subdir='RAMPFIT', search=meta.imgproc_ext)

	# Run the pipeline on every file
	for file in files:
		# Set up pipeline
		pipeline = Image2Pipeline()

		# Set up directory to save to
		pipeline.output_dir = meta.odir + 'IMGPROCESS/'
		pipeline.logcfg = pipeline.output_dir + 'imageprocess-log.cfg'
		pipeline.save_results = True
		
		if os.path.exists(pipeline.output_dir) == False:
			    os.mkdir(pipeline.output_dir)

		pipeline.run(file)

	return
