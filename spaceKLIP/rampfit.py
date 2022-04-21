import os, glob
from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

def stsci_ramp_fitting(meta):
	"""
	Use the JWST pipeline to process *uncal.fits files to *rateints.fits files
	"""

	search = '*' + meta.ramp_ext
	# Get all of the files in the input directory
	files = glob.glob(meta.idir+search)
	if len(files) == 0:
		raise ValueError('Unable to locate any {} files in directory {}'.format(search, meta.idir))

	# Run the pipeline on every file
	for file in files:
		# Set up pipeline
		pipeline = Detector1Pipeline()
		pipeline.jump.skip = meta.skip_jump
		pipeline.jump.rejection_threshold = meta.jump_threshold
		pipeline.save_results = True

		# Set up directory to save into
		pipeline.output_dir = meta.odir + 'RAMPFIT'
		if os.path.exists(pipeline.output_dir) == False:
			    os.mkdir(pipeline.output_dir)
		pipeline.run(file)

	return