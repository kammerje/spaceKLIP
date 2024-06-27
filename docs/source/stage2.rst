.. _stage2:

Stage 2
=======

The following steps are part of the ``Coron2Pipeline`` custom version of the JWST pipeline's ``Detector2Pipeline``.

Note, the ``coron2pipeline.run_obs()`` function provides a succinct interface to create and invoke Coron2Pipeline on all
observations in a spaceKLIP Database; that's the recommended way to invoke Coron2Pipeline.


All regular pipeline stage2 steps, mostly
------------------------------------------


The stage 2 pipeline steps from regular ``Image2Pipeline`` are mostly invoked.  

Background subtraction is skipped, to be performed later using functions in spaceKLIP image analyses. 


OutlierDetection
----------------

An outlier detection step is added. This is a step from the regular pipeline, so what's different here? Not sure, may be that this is run as part
of stage 2 instead of stage 3?  **TBC, write better docs of this**

This step is skipped for images with exptype = NRC_TA or NRC_TACONFIRM.

