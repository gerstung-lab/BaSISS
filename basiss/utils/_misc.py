import numpy as np


def inverse_dict(d):
    """Invert the dict

    Parameters
    ----------
    d : dict
        Input dict {'key1': ['val1', 'val2', ...], ...}
    Returns
    -------
    dict
        Inverted dict {'val1': 'key1',  'val2': 'key1', ...}
    """
    d_inv = {}
    for k, vs in d.items():
        for v in vs:
            d_inv[v] = k
    return d_inv


def categ_df2color(mat, colours=np.array([[30, 144, 255],
                                          [255, 99, 71],
                                          [77, 175, 74],
                                          [152, 78, 163],
                                          [255, 127, 0]])):
    """Convert 0-1 values to colour

    Parameters
    ----------
    mat : np.array
        array of 0-1 values, dim1 correspond to colours
    colours : np.array
        rgb values of the colours

    Returns
    -------
    np.array
        Array with corresponding colours
    """
    ns, nr = mat.shape
    rgba = np.array([[list(colours[c_i] / 255) + [1.0]] * nr for c_i in range(colours.shape[0])])
    rgba_pre = rgba[mat, -1]
    rgba_pre[mat == -1] = [0.8, 0.8, 0.8, 1]
    return rgba_pre