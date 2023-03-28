import os, glob
import numpy as np
import astropy
import spaceKLIP

import pytest

testpath = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.parametrize('instname', ['nircam', 'miri'])   # Repeat this test for both NIRCam and MIRI
def test_database_init_and_read(instname):
    """ Basic test: can we initialize a database, and read in some level 0 data?

    """

    if instname=='nircam':
        pattern = '*nrca*uncal.fits'
    else:
        pattern = '*mirimage*uncal.fits'
    fitsfiles = glob.glob(os.path.join(testpath, "test_data", pattern))
    output_dir = os.path.join(testpath, 'test_outputs')

    db = spaceKLIP.database.Database(output_dir=output_dir)
    assert isinstance(db, spaceKLIP.database.Database), "Database object was not created as expected"
    assert os.path.abspath(db.output_dir) == os.path.abspath(output_dir), 'Output dir was not set as expected'
    assert len(db.obs)==0, "Database should start with zero contents"

    db.read_jwst_s012_data(datapaths=fitsfiles)
    assert len(db.obs)>0, "Database should have nonzero contents after reading in data"

    # We could do additional things here to validate the contents of the database

