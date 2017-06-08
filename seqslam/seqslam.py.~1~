from utils import AttributeDict
import os
import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlab.cm as cm

class SeqSLAM():
    params = None
    
    def __init__(self, params):
        self.params = params

    def run(self):
        # step1: extract the map features 
        results = self.doFeatrueExtraction()
        
        # step2: caculate the feature difference
        results = self.doDifferenceMatrix(results)
        
        # step3: contrast enhancement
        results = self.doEnhancement(results)        

        # step4: find the matches
        results = self.doFindMatch(results)

        return results

    def doFeature
