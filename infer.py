import numpy as np
import json
#load model
W_path=""
b_path=""
W=np.load(W_path)
b=np.load(b_path)
mean=json.load('mean.json')
std=json.load('std.json')

