import os, glob
import numpy as np
import astropy
import spaceKLIP

import pytest

testpath = os.path.dirname(os.path.abspath(__file__))

def test_utils_read_obs():
    """test the read_obs function

    """
    fitsfiles = glob.glob(os.path.join(testpath, "test_data", '*nrca*calints.fits'))
    fitsfiles.sort()
    fn = fitsfiles[0]

    data, erro, pxdq, head_pri, head_sci, is2d = spaceKLIP.utils.read_obs(fn)

    assert isinstance(data, np.ndarray), "read_obs didn't return a numpy data array"
    assert isinstance(head_pri, astropy.io.fits.Header), "read_obs didn't return a FITS header"
