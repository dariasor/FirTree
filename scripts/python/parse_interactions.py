import sys
import os
import math
import numpy
from collections import defaultdict

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

def parse_effects(tar_std, dominant):
	with open("core_features.list.txt", "w") as fout:
		for file in os.listdir("."): 
			if file.endswith(".effect.txt"):

				with open(file, "r") as fin:
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
				if effect_scale > 0.25:
					dominant[title] = effect_scale

def parse_iplots(tar_std, iscales):
	for file in os.listdir("."): 
		if file.endswith(".iplot.txt"):
		
			with open(file, "r") as fin:
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
			
			iscales[(titleRow, titleCol)] = max(interaction_scale1, interaction_scale2)


if len(sys.argv) != 1:
	sys.exit("usage: " + sys.argv[0])

fin = open("log.txt", "r")
flist = open("list.txt", "w")
prev_line = ""
std_val = -1
iscales = {}
dominant = defaultdict(float) #features that have huge (dominant) effects and their effect sizes
strong = defaultdict(float) #features involved in strong interactions and sums of their interactions sizes
weak_n = defaultdict(int) #features involved in weak interactions and number of their weak interactions
weak_sum = defaultdict(float)  #features involved in weak interactions and sum of their interaction sizes
for line in fin:
	cur_line = line.strip()
	if "points in the validation set" in cur_line:
		data = cur_line.split(" ")
		std_val = float(data[-1])
	if cur_line.startswith("ag_interactions"):
		parse_effects(std_val, dominant)
		parse_iplots(std_val, iscales)		
	if cur_line.endswith("is present.") or cur_line.endswith("is absent."):
		data = prev_line.split(" ")
		f1 = data[3]
		f2 = data[5]
		data = cur_line.split()
		if (f1, f2) in iscales:
			strength = float(data[2])
			scale = iscales[(f1,f2)]
			flist.write(f1 + " " + f2 + " " + str(strength) + " " + str(scale) + "\n")
			if scale >= 0.02:
				if strength >= 7.0:
					strong[f1] += scale
					strong[f2] += scale
				elif strength >= 3.0:
					weak_n[f1]+=1
					weak_n[f2]+=1
					weak_sum[f1] += scale
					weak_sum[f2] += scale
	prev_line = cur_line

for f in strong.keys():
	if f in weak_sum:
		strong[f] += weak_sum[f]
		weak_sum.pop(f)
	dominant.pop(f,0)

for f in weak_n.keys():
	if weak_n[f] < 3:
		weak_sum.pop(f,0)
	else:
		dominant.pop(f,0)

		
strong_list = sorted(strong.iteritems(), key = lambda x : x[1], reverse = True)
weak_list = sorted(weak_sum.iteritems(), key = lambda x : x[1], reverse = True)
dominant_list = sorted(dominant.iteritems(), key = lambda x : x[1], reverse = True)
with open("candidates.txt", "w") as fout:
	for s in strong_list:
		fout.write(s[0] + "\ts\n")
	for w in weak_list:
		fout.write(w[0] + "\tw\n")
	for d in dominant_list:
		fout.write(d[0] + "\td\n")
