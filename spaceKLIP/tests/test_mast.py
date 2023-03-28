import spaceKLIP
import astropy

import pytest


def test_query_coron_datasets():

    result = spaceKLIP.mast.query_coron_datasets('NIRCam','F356W')

    assert isinstance(result, astropy.table.Table)
