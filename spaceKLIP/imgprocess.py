import os, glob
from jwst.pipeline.calwebb_image2 import Image2Pipeline

def stsci_image_processing(meta):
	"""
	Use the JWST pipeline to process *rateints.fits files to *calints.fits files
	"""

	# Figure out where to look for files
	if meta.do_rampfit:
		#Use the output directory that was just created
		rdir = meta.odir + 'RAMPFIT/'
	else:
		#Use the specified input directory
		rdir = meta.idir

	# Grab the files
	files = glob.glob(rdir + '*rateints.fits')
	if len(files) == 0:
		# Let's look for a 'RAMPFIT' subdir
		if os.path.exists(rdir + 'RAMPFIT'):
			print('Located RAMPFIT folder within input directory.')
			rdir += 'RAMPFIT/*rateints.fits'
			files = glob.glob(rdir)
		
		# If there are still no files, look in output directory
		if (len(files) == 0) and ('/RAMPFIT/' not in rdir):
			print('WARNING: No *rateints.fits files found in input directory, searching output directory.')
			rdir = meta.odir + 'RAMPFIT/*rateints.fits'
			files = glob.glob(rdir)

		if len(files) == 0:
			raise ValueError('WARNING: Unable to find any *rateints.fits files in specified input or output directories.')

	if meta.verbose:
		print('Found {} files in directory: {}'.format(len(files), rdir))

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
