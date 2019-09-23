import sys
import os
import math
import numpy
# usage: parse_effects dir target_std

def weighted_std(values, weights_in):
    """
    Return weighted standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights_in)
    # Fast and numerically precise:
    values2 = (values-average)**2
    variance = numpy.average(values2, weights=weights_in)
    return math.sqrt(variance)


path = sys.argv[1] + "/"
tar_std = float(sys.argv[2])

with open(path + "core_features.list.txt", "w") as fout:
	for file in os.listdir(path): 
		if file.endswith(".effect.txt"):

			with open(path + file, "r") as fin:
				rawData = fin.readlines()

			title = rawData[0].split("\t")[1][:-7]

			values = []
			weights = []
			for i in range(2, len(rawData)):
				cur_line = rawData[i].split("\t")
				values.append(float(cur_line[2].rstrip()))
				weights.append(float(cur_line[0]))
				
			effect_scale = weighted_std(values, weights) / tar_std
			
			fout.write(title + "\t" + str(effect_scale) + "\n")
		
		
