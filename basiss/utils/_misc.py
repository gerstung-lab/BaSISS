def inverse_dict(d):
    d_inv = {}
    for k, vs in d.items():
        for v in vs:
            d_inv[v] = k
    return d_inv