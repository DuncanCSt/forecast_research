import rpy2.robjects as robjects
import numpy as np

readRDS = robjects.r['readRDS']

def load_rds_array(path):
    """Load an R array into numpy, preserving shape and dimnames."""
    r_obj    = readRDS(path)
    dims     = tuple(int(d) for d in robjects.r['dim'](r_obj))
    arr      = np.array(r_obj).reshape(dims, order='F')
    r_dimnames = robjects.r['dimnames'](r_obj)
    dimnames = []
    for dn in r_dimnames:
        if dn == robjects.rinterface.NULL:
            dimnames.append(None)
        else:
            dimnames.append(list(dn))
    return arr, dimnames
