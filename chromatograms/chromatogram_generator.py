"""ChromatogramGenerator"""

import random
import numpy as np
import scipy.stats as stats

class ChromatogramGenerator(object):
  """ ChromatogramGenerator
  """
  def __init__(self):
    """ __init__
    """
    pass

  def generate_chromatogram(self, length=100, max_height=1):
    """ get_chrom 
    """    
    # ベースのノイズを作成
    noise = 0.1
    data = [random.random() * noise for i in np.arange(100)]    
    peak_wing = 2
    e = 0.001/peak_wing
    peak = stats.norm.pdf(np.arange(-2, 2+e, 4.0/(peak_wing*2)))/stats.norm.pdf(0)
    peakStartPos = random.randint(0,length -(peak_wing*2+1))
    peakTopPos = peakStartPos + peak_wing
    data[peakStartPos:peakStartPos+peak_wing*2+1] += peak
    return data
