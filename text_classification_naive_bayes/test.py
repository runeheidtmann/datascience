# import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
import csv
import random
from string import punctuation

wdf = pd.read_csv('fuldformer.txt', delimiter=';', names=["root","inflection","gender"])

root = wdf.loc[wdf['inflection']=='vinduespudseref']
print(root)

if not len(root):
    print(root.iloc[0]['root'])
