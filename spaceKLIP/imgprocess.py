import os, glob
from jwst.pipeline.calwebb_image2 import Image2Pipeline

from . import utils
from . import io

def run_image_processing(meta, subdir_str, itype):

	files = io.get_working_files(meta, meta.done_rampfit, subdir=subdir_str, search=meta.imgproc_ext, itype=itype)

	# Run the pipeline on every file
	for file in files:
		# Set up pipeline
		pipeline = Image2Pipeline()

		# Set up directory to save to
		pipeline.output_dir = meta.odir + subdir_str.replace('RAMPFIT', 'IMGPROCESS') + '/'
		pipeline.logcfg = pipeline.output_dir + 'imageprocess-log.cfg'
		pipeline.save_results = True
		
		if os.path.exists(pipeline.output_dir) == False:
			    os.makedirs(pipeline.output_dir)

		pipeline.run(file)

def stsci_image_processing(meta):
	"""
	Use the JWST pipeline to process *rateints.fits files to *calints.fits files
	"""

	if meta.imgproc_idir:
		run_image_processing(meta, 'RAMPFIT', itype='default')
	if meta.imgproc_bgdirs:
		if meta.bg_sci_dir != 'None':
			run_image_processing(meta, 'RAMPFIT_BGSCI', itype='bgsci')
		if meta.bg_ref_dir != 'None':
			run_image_processing(meta, 'RAMPFIT_BGREF', itype='bgref')

	return
