import numpy as np
import scipy.stats as stats
import random

import chromatogram_generator as cg

GEN = cg.ChromatogramGenerator()
DATA = GEN.generate_chromatogram()
for d in DATA:
  print(d)

path_w = './out.txt'

with open(path_w, mode='w') as f:
  for d in DATA:
    f.write('{:.5f}'.format(d) + '\n')
