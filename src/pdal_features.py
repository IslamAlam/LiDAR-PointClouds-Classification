from pathlib import Path
import time
import pandas as pd

import sys
#Second append the folder path

sys.path.insert(0, './../inc')
sys.path.insert(0, './../src')

from classfier_tools import plot_confusion_matrix, classic_classifier
from pdal_pipline import las_add_features, las_2_dataframe, las_eigen_features



from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--inputfile", dest="inputfilename",
                    help="write report to FILE", metavar="FILE")
parser.add_argument("-o", "--outputfile", dest="outputfilename",
                    help="write report to FILE", metavar="FILE")

parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")

args = parser.parse_args()
print(args)
print(args.inputfilename)

inputfile = str(args.inputfilename)

#outputfile = inputfile[:-4] + "_features" + inputfile[-4:]
outputfile = str(args.outputfilename)


las_eigen_features(inputfile, outputfile)
    