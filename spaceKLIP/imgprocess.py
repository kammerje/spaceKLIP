import os, glob
from jwst.pipeline.calwebb_image2 import Image2Pipeline

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from . import utils
from . import io

def run_image_processing(meta, subdir_str, itype, dqcorr='None'):

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

		if meta.dq_only != True:
			pipeline.run(file)

	# Perform additional cleaning
	if meta.dqcorr != 'None':
		files = glob.glob(pipeline.output_dir+'*')

		if os.path.exists(subdir_str.replace('RAMPFIT', 'IMGPROCESS') + '/')
		for file in files:
			with fits.open(file) as hdu:
				dq = hdu['DQ'].data
				data_corr = hdu['SCI'].data
				for i in range(dq.shape[0]):
					toclean = np.where(dq != 0)

					if dqcorr == 'median':
						data_corr[i][toclean] = median_filter(data[i], size=5)[toclean]

				hdu['SCI'].data = data_corr
				
				hdu.writeto(file, overwrite=True)
				# from matplotlib.colors import LogNorm

				# fig, (ax1, ax2) = plt.subplots(1, 2)
				# ax2.imshow(data_corr[0], norm=LogNorm())
				# ax1.imshow(hdu['SCI'].data[0], norm=LogNorm())
				# plt.show()

def stsci_image_processing(meta):
	"""
	Use the JWST pipeline to process *rateints.fits files to *calints.fits files
	"""

	if meta.imgproc_idir:
		run_image_processing(meta, 'RAMPFIT/SCI+REF', itype='default')
	if meta.imgproc_bgdirs:
		if meta.bg_sci_dir != 'None':
			run_image_processing(meta, 'RAMPFIT/BGSCI', itype='bgsci')
		if meta.bg_ref_dir != 'None':
			run_image_processing(meta, 'RAMPFIT/BGREF', itype='bgref')

	return
