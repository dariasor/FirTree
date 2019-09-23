import sys
import os
import math
import numpy
# usage: parse_iplots dir target_std

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

with open(path + "core_interactions.list.txt", "w") as fout:
	for file in os.listdir(path): 
		if file.endswith(".iplot.txt"):
		
			with open(path + file, "r") as fin:
				rawData = fin.readlines()
		
			titleRow = rawData[1].split("\t")[1].split("\n")[0]
			titleCol = rawData[2].split("\t")[1].split("\n")[0]

			values1 = []
			weights1 = []
			for i in range(7, len(rawData)):
				cur_line = rawData[i].split("\t")
				values1.append(float(cur_line[2]) - float(cur_line[-1]))
				weights1.append(float(cur_line[0]))		

			values2 = []
			weights2 = []
			w_line = rawData[5].split("\t")
			b_line = rawData[7].split("\t")
			e_line = rawData[-1].split("\t")
			for i in range(2, len(w_line)):
				weights2.append(float(w_line[i]))
				values2.append(float(e_line[i]) - float(b_line[i]))

			interaction_scale1 = weighted_std(values1, weights1) / tar_std
			interaction_scale2 = weighted_std(values2, weights2) / tar_std
			
			fout.write(titleRow + "\t" + titleCol + "\t" + str(interaction_scale1) + "\t" + str(interaction_scale2) + "\n")				
