import os, glob
import numpy as np
import astropy
import spaceKLIP

import pytest

testdatapath = os.getenv('SPACEKLIP_TEST_DATA_PATH')
testpath = os.path.dirname(os.path.abspath(__file__))

os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
os.environ['CRDS_PATH'] = '/grp/crds/cache/'

def test_has_test_data():

    assert os.getenv('SPACEKLIP_TEST_DATA_PATH'), "The $SPACEKLIP_TEST_DATA_PATH environment variable must be set"
    assert os.path.isdir(testdatapath), "The $SPACEKLIP_TEST_DATA_PATH variable must point to a directory that exists"
    assert len(glob.glob(os.path.join(testdatapath, "*")))>0, "There must be some data files in the $SPACEKLIP_TEST_DATA_PATH folder"


@pytest.mark.parametrize('instname', ['nircam', 'miri'])   # Repeat this test for both NIRCam and MIRI
def test_database_init_and_read(instname):
    """ Basic test: can we initialize a database, and read in some level 0 data?

    """

    if instname=='nircam':
        pattern = '*nrca*uncal.fits'
    else:
        pattern = '*mirimage*uncal.fits'
    fitsfiles = glob.glob(os.path.join(testdatapath, pattern))
    output_dir = os.path.join(testpath, 'test_outputs')

    db = spaceKLIP.database.Database(output_dir=output_dir)
    assert isinstance(db, spaceKLIP.database.Database), "Database object was not created as expected"
    assert os.path.abspath(db.output_dir) == os.path.abspath(output_dir), 'Output dir was not set as expected'
    assert len(db.obs)==0, "Database should start with zero contents"

    db.read_jwst_s012_data(datapaths=fitsfiles)
    assert len(db.obs)>0, "Database should have nonzero contents after reading in data"

    # We could do additional things here to validate the contents of the database

