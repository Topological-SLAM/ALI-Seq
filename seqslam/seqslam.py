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

    def doFeatureExtraction(self):

    def doDifferenceMatrix(self, results):
        filename = 'saved.mat'
        
        if self.params.differenceMatrix.load and os.path.isfile(filename):
            print('Error: Cannot calculate difference matrix with less than 2 datasets.')

            d = loadmat(filename)
            results.D = d.D
        else:
            if len(results.dataset)<2:
                return None

            print('Calculating image difference matrix')
            
            results.D = self.getDifferenceMatrix(results.dataset[0], results.dataset[1])
            
            # save
            if self.params.differenceMatrix.save:
                savemat(filename, {'D': results.D})
                
        return results
