###############
spaceKLIP 2 🚀🪐
###############

SpaceKLIP is a data reduction pipeline for JWST high-contrast imaging. All code is currently under heavy development
and typically expected features may not be available. 

Compatible test data: `here <https://stsci.box.com/s/0oteh8smujl3pup07hyut6hr4ag1i2el>`_ 

Code status
***********

The current release of spaceKLIP is not stable enough for a straightforward install with conda / pip. At this stage
it is recommended that you clone the git repository directory for installation:

::

	database.py

- ``read_jwst_s012_data``: working
- ``read_jwst_s3_data``: working
- ``print_obs``: working
- ``print_red``: working
- ``update_obs``: working
- support for companion fitting: missing

::

	utils.py

- ``read_obs``: working
- ``write_obs``: working
- ``read_red``: working

::

	coron1pipeline.py

- ``run_obs``: working
- custom saturation step: working
- custom reference pixel step: needs updating (side reference pixels not working for subarrays)

::

	coron2pipeline.py

- ``run_obs``: working
- additional outlier detection step: working

::

	coron3pipeline.py

- ``make_asn_file``: working
- ``run_obs``: working

::

	pyklippipeline.py

- ``run_obs``: working
- no absolute PSF alignment, this is now done using the ``ImageTools`` library
- no relative frame alignment, this is now done using the ``ImageTools`` library
- the old ``JWST.py`` is now implemented here

::

	classpsfsubpipeline.py

- ``run_obs``: not working, under development

::

	imagetools.py

- ``remove_frames``: working
- ``crop_frames``: working
- ``pad_frames``: working
- ``coadd_frames``: working
- ``subtract_median``: working
- ``subtract_background``: working
- ``fix_bad_pixels``: working
- ``replace_nans``: working
- ``align_frames``: working (relative frame alignment)
- ``recenter_frames``: working only for TA images (absolute PSF alignment)
- blurring: missing

::

	contrast.py

- ``raw_contrast``: working with ``pyklippipeline``, but does not normalize flux to host star, not working with ``coron3pipeline`` since PSF center not being tracked
- need to add throughput map to database and take it into account for raw contrast
- need to add host star model
- need to add calibrated contrast

::

	companion.py

- missing