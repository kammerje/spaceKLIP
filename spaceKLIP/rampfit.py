import os, glob
from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

def run_ramp_fitting(meta, idir, osubdir):
	search = '*' + meta.ramp_ext
	# Get all of the files in the input directory
	files = glob.glob(idir+search)
	if len(files) == 0:
		raise ValueError('Unable to locate any {} files in directory {}'.format(search, idir))
	# Run the pipeline on every file in a directory
	for file in files:
		# Set up pipeline
		pipeline = Detector1Pipeline()
		pipeline.jump.skip = meta.skip_jump
		pipeline.jump.rejection_threshold = meta.jump_threshold
		pipeline.ramp_fit.maximum_cores = meta.ramp_fit_max_cores
		pipeline.save_results = True

		# Set up directory to save into
		pipeline.output_dir = meta.odir + osubdir

		if os.path.exists(pipeline.output_dir) == False:
			os.makedirs(pipeline.output_dir)
		pipeline.run(file)

	return

def stsci_ramp_fitting(meta):
	"""
	Use the JWST pipeline to process *uncal.fits files to *rateints.fits files
	"""
	if meta.rampfit_idir:
		run_ramp_fitting(meta, meta.idir, 'RAMPFIT/')
	if meta.rampfit_bgdirs:
		if meta.bg_sci_dir != 'None':
			run_ramp_fitting(meta, meta.bg_sci_dir, 'RAMPFIT_BGSCI/')
		if meta.bg_ref_dir != 'None':
			run_ramp_fitting(meta, meta.bg_ref_dir, 'RAMPFIT_BGREF/')

	return