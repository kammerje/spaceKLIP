.. _imagetools:

Image Tools
============

TBD: Detailed documentation of the ``spaceklip.imagetools`` module will go here.



Steps for background removal and pixel cleaning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


subtract_background
--------------------
See :py:meth:`~spaceKLIP.imagetools.ImageTools.subtract_background` docstring for details. 

If you have dedicated background exposures (exp type=SCI_BG or REF_BG), subtract those from your SCI or REF data. If there are multiple ints in the BG exposure, they are median-combined before the subtraction. 

subtract_median
--------------------
See :meth:`~spaceKLIP.imagetools.ImageTools.subtract_median` docstring for details. 

Subtract the median background level from each frame. The median is performed after masking out bright sources, using sigma clipping or similar techniques (see function docstring for the choices of how to do this.)

Recommended for NIRCam only. 


fix_bad_pixels
--------------
See :meth:`~spaceKLIP.imagetools.ImageTools.fix_bad_pixels` docstring for details.

Flexible, complex set of multiple different ways to detect and replace bad pixels. Very complex set of options; see tutorial notebooks for examples of usage.

replace_nans
--------------
See :meth:`~spaceKLIP.imagetools.ImageTools.replace_nans` docstring for details.

Simple step to replace any NaNs with zero.


Steps for Image Registration and Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


update_nircam_centers
---------------------

See :meth:`~spaceKLIP.imagetools.ImageTools.update_nircam_centers` docstring for details.

Update header metadata for locations of the coronagraphs. This uses a table of better center locations measured from NIRCam commissioning data by J. Leisenring and J. Girard.  *This is a temporary step which should eventually be unnecessary, after a planned update of the SIAF calibratoin data.*  Replaces the header values for the CRPIX locations for the mask locations.  Only applies to NIRCam data, naturally. 


**Details:** For each file, read current CRPIXn values. Use APERNAME to look up crpix values from `crpix_jarron`. Use FILTER to look up filter shift from `filter_shifts_jarron`. Compute deltas based on crpix_jarron and filter_shifts relative to Siaf x/ysciref values. Apply those deltas to change the CRPIX values. 

**TBD: Document where these come from. Make some simple plots of those values to include here**.

 * This step modifies the WCS headers, but not the pixel data. 

recenter_frames
---------------
See :meth:`~spaceKLIP.imagetools.ImageTools.recenter_frames` docstring for details.

To better measure the location of the star with respect to the coronagraph, we create a simulation of the star behind the coronagraph (using webbpsf), and cross-correlate this with the observed PSF. The cross correlation peak is used to infer the offset of the star relative to the mask center. The measured offset is used to shift the first frame to be centered. Then subsequent frames are aligned to that first frame.

The accuracy of this algorithm is around 7 milliarcsec according to testing.

This step also shifts to account for the coronagraph not being precisely centered in the subarray. After this step, the star center will be at the center of the pixel array.

This step only works for NIRCam data; for MIRI data it will apply zero shifts, i.e. do nothing.

 * This step modifies the pixel data to apply shifts


align_frames
------------
See :meth:`~spaceKLIP.imagetools.ImageTools.align_frames` docstring for details.

This step applies shifts to the image pixel data to align frames. All subsequent frames are aligned to the first frame of the first science integration. (I.e. the second roll and all references are aligned to the first roll).

This measures and applies relative shifts between subsequent frames and the first frame.

pad_frames
----------
See :meth:`~spaceKLIP.imagetools.ImageTools.pad_frames` docstring for details.

Pad empty space around frames to give space to rotate and align during pyklip. THis puts a region of NAN pixels around the outside.


