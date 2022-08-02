import os, glob
from jwst.pipeline.calwebb_detector1 import Detector1Pipeline
from jwst.datamodels import dqflags
from jwst.refpix import RefPixStep
from astropy.io import fits


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

		steps = pipeline.steps

		# If we're forcing reference pixels need to skip a lot of steps
		# that get performed elsewhere
		if meta.force_ref_pixels != False:
			file = preprocess_file(meta, file, osubdir)
			steps['group_scale']['skip'] = True
			steps['dq_init']['skip'] = True
			steps['saturation']['skip'] = True
			steps['superbias']['skip'] = True
			steps['refpix']['skip'] = True

		steps['saturation']['n_pix_grow_sat']=meta.saturation_boundary
		steps['jump']['skip'] = meta.skip_jump
		steps['jump']['rejection_threshold'] = meta.jump_threshold
		steps['jump']['maximum_cores'] = meta.ramp_fit_max_cores
		steps['dark_current']['skip'] = meta.skip_dark_current
		steps['ipc']['skip'] = meta.skip_ipc

		# Set up directory to save into
		pipeline.output_dir = meta.odir + osubdir
		if os.path.exists(pipeline.output_dir) == False:
			os.makedirs(pipeline.output_dir)
		
		# Call pipeline
		pipeline.call(file, output_dir=pipeline.output_dir, steps=pipeline.steps, \
						save_results=True)

	return

def stsci_ramp_fitting(meta):
	"""
	Use the JWST pipeline to process *uncal.fits files to *rateints.fits files
	"""
	if meta.rampfit_idir:
		run_ramp_fitting(meta, meta.idir, 'RAMPFIT/SCI+REF/')
	if meta.rampfit_bgdirs:
		if meta.bg_sci_dir != 'None':
			run_ramp_fitting(meta, meta.bg_sci_dir, 'RAMPFIT/BGSCI/')
		if meta.bg_ref_dir != 'None':
			run_ramp_fitting(meta, meta.bg_ref_dir, 'RAMPFIT/BGREF/')

	return


def preprocess_file(meta, file, osubdir):
	# Need to run things in steps, first skip everything after superbias
	pipe = Detector1Pipeline()

	steps = pipe.steps
	steps['refpix']['skip'] = True
	steps['saturation']['skip'] = True
	steps['linearity']['skip'] = True
	steps['persistence']['skip'] = True
	steps['dark_current']['skip'] = True
	steps['jump']['skip'] = True
	steps['ramp_fit']['skip'] = True
	steps['gain_scale']['skip'] = True
	steps['ipc']['skip'] = meta.skip_ipc
	steps['saturation']['n_pix_grow_sat'] = meta.saturation_boundary

	#Run pipeline
	result = pipe.call(file, steps=pipe.steps)

	# Redefine some of the reference pixels
	val = int(meta.force_ref_pixels)
	result.pixeldq[0:val] = result.pixeldq[0:val,:] | ~dqflags.pixel['REFERENCE_PIXEL']
	result.pixeldq[:-val] = result.pixeldq[:-val,:] | ~dqflags.pixel['REFERENCE_PIXEL']

	# Run reference pixel correction
	refpix_step_amp_corr = RefPixStep(use_side_ref_pixels=False)
	refpix_amp_corr = refpix_step_amp_corr.run(result)

	# Restore pixels
	refpix_amp_corr.pixeldq[0:val,:] = refpix_amp_corr.pixeldq[0:val,:] & ~dqflags.pixel['REFERENCE_PIXEL']
	refpix_amp_corr.pixeldq[-val:,:] = refpix_amp_corr.pixeldq[-val:,:] & ~dqflags.pixel['REFERENCE_PIXEL']

	#Return half finished pipeline object
	return refpix_amp_corr
