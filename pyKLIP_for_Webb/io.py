from __future__ import division

import yaml


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

    mstar = {'F250M': 6, # vegamag
             'F300M': 6, # vegamag
             'F335M': 6, # vegamag
             'F356W': 6, # vegamag
             'F410M': 6, # vegamag
             'F430M': 6, # vegamag
             'F444W': 6, # vegamag
             'F460M': 6, # vegamag
             }
    config['mstar'] = mstar

    return config

