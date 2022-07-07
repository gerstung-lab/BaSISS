import numpy as np

def inverse_dict(d):
    d_inv = {}
    for k, vs in d.items():
        for v in vs:
            d_inv[v] = k
    return d_inv

def categ_df2color(mat, colours = np.array([[30,144,255],
                                            [255,99,71],
                                            [77,175,74],
                                            [152,78,163],
                                            [255,127,0]])):
    
    ns, nr = mat.shape
    rgba = np.array([[list(colours[c_i]/255) + [1.0]]*nr for c_i in range(colours.shape[0])])
    rgba_pre = rgba[mat,-1]
    rgba_pre[mat == -1] = [0.8,0.8,0.8,1]
    return rgba_pre