# from Levenshtein import jaro

import scipy.cluster.hierarchy as sch
import numpy as np
import sys

distanceMatrixDimension = len(p_names)
upper_triangle = np.triu_indices(distanceMatrixDimension, 1)
distances = np.apply_along_axis(dis, 0, upper_triangle)
Z = sch.linkage(distances,'average')
fclster = sch.fcluster(Z, 0.2 * max_dist, 'distance')

print("done");