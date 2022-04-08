from __future__ import division

import yaml
import json

import numpy as np
from astropy.table import Table

def read_config(file):
    """
    Read a .yaml configuration file that defines the code execution. 

    Parameters
    ----------
    file : str  
        File path of .yaml configuration file

    Returns
    -------
    config : dict
        Dictionary of all configuration parameters

    """

    with open(file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            raise yaml.YAMLError

    temp = config['pa_ranges_bar']
    Nval = len(temp)
    pa_ranges_bar = []
    if (Nval % 2 != 0):
        raise UserWarning('pa_ranges_bar needs to be list of 2-tuples')
    for i in range(Nval//2):
        pa_ranges_bar += [(float(temp[2*i][1:]), float(temp[2*i+1][:-1]))]
    config['pa_ranges_bar'] = pa_ranges_bar

    return config

def meta_to_json(meta, savefile='./MetaSave.txt'):
    '''
    Convert meta object to a dictionary and save into a json file
    '''

    # Need to convert astropy table to strings
    for i in meta.obs:
        meta.obs[i] = Table.pformat_all(meta.obs[i])

    # Need to find numpy types and convert to default python types
    for i in vars(meta).keys():
        # Is it a numpy data type?
        if isinstance(vars(meta)[i], np.generic):
            to_py = getattr(meta, i).item()
            setattr(meta, i, to_py)

        # Check another level down
        if isinstance(vars(meta)[i], dict):
            d = getattr(meta, i)

            for key in d.keys():
                if isinstance(d[key], np.generic):
                    d[key] = d[key].item()
            setattr(meta, i, d)
        elif isinstance(vars(meta)[i], list):
            l = getattr(meta, i)

            for j in l:
                if isinstance(j, np.generic):
                    l[j] = l[j].item()
            setattr(meta, i, l)
                

    with open(savefile, 'w') as msavefile:
        json.dump(vars(meta), msavefile)

    return


def read_metajson(filepath):
    '''
    Load a Meta save file as a json dictionary, don't convert
    back into a class as it doesn't seem necessary yet. 
    '''
    with open(filepath) as f:
        metasave = json.load(f)

    return metasave