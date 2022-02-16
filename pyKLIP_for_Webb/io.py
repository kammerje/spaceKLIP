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

    return config

