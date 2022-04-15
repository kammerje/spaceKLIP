import os, glob
from jwst.pipeline.calwebb_image2 import Image2Pipeline

def stsci_image_processing(meta):
	"""
	Use the JWST pipeline to process *rateints.fits files to *calints.fits files
	"""

	# Check if the ramp fitting was performed and assign idir accordingly.
	if meta.do_rampfit:
		idir = meta.odir + 'RAMPFIT/'
	else:
		idir = meta.idir

	# Get all the files in the input directory
	files = glob.glob(idir+'*rateints.fits')
	if len(files) == 0:
		raise ValueError('Unable to locate any *rateints.fits files in directory {}'.format(meta.idir))

	# Run the pipeline on every file
	for file in files:
		# Set up pipeline
		pipeline = Image2Pipeline()
		pipeline.save_results = True

		# Set up directory to save into
		pipeline.output_dir = meta.odir + 'IMGPROCESS/'
		if os.path.exists(pipeline.output_dir) == False:
			    os.mkdir(pipeline.output_dir)

		pipeline.run(file)

	return
