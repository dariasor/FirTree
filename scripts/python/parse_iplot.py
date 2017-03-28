import sys
import os
import math
# usage: parse_iplot dir split_attr split_val 

path = sys.argv[1] + "/"
splitAttr = sys.argv[2]
splitVal = float(sys.argv[3])

def transpose2D(data):
	nrow = len(data)
	ncol = len(data[0])
	newdata = []
	for i in range(ncol):
		newdata.append([])
	for i in range(ncol):
		for j in range(nrow):
			newdata[i].append(data[j][i])
	return newdata

for file in os.listdir(path):
	if (not file.startswith("chosen")) and file.endswith(".iplot.txt") and (splitAttr in file):
		fileName, fileExtension = os.path.splitext(file)
		fileDens = fileName + '.dens.txt'

		with open(path + file, "r") as fin:
			rawData = fin.readlines()
		with open(path + fileDens, "r") as fin:
			rawDens = fin.readlines()

		titleRow = rawData[1].split("\t")[1].split("\n")[0]
		titleCol = rawData[2].split("\t")[1].split("\n")[0]

		data = []
		dens = []

		for i in range(5, len(rawData)):
			data.append(rawData[i].split("\t"))
		for i in range(4, len(rawDens)):
			dens.append(rawDens[i].split("\t")[1:])
		for i in range(len(data)):
			for j in range(len(data[i])):
				if(data[i][j].strip() == "?"):
					data[i][j] = "Inf"
				data[i][j] = float(data[i][j])
		for i in range(len(dens)):	
			for j in range(len(dens[i])):
				dens[i][j] = float(dens[i][j])
		# print titleCol, " v.s. " , splitAttr
		if(titleCol == splitAttr): 
			# data = np.transpose(data)
			# dens = np.transpose(dens)
			data = transpose2D(data)
			dens = transpose2D(dens)
			titleCol, titleRow = titleRow, titleCol
		fileOut = "chosen." + titleRow + "." + titleCol + ".iplot.txt"
		fileDensOut = "chosen." + titleRow + "." + titleCol + ".iplot.dens.txt"
		
		tmpSum = 0
		for i in range(len(dens)):   # should be just 2 lines
			tmpSum = sum(dens[i])
			for j in range(len(dens[i])):
				dens[i][j] = tmpSum/len(dens[i])
		for i in range(len(dens)):
			for j in range(len(dens[i])):
				data[i+2][j+2] = data[i+2][j+2] * dens[i][j]

		# find the row that represents the splits
		label = 0
		for i in range(len(dens) - 1):
			if (data[i+2][1] < splitVal) and (data[i+3][1] >= splitVal):
				label = i + 1
				break
		# compress the matrix
		# deal with the values
		dataOut = []
		densOut = []
		dataOut.append(data[0])
		dataOut.append(data[1])

		dataOut.append(data[2])
		for i in range(3, 2 + label):
			for j in range(len(data[i])):
				dataOut[2][j] += data[i][j]

		dataOut.append(data[2 + label])
		for i in range(3 + label, len(data)):
			for j in range(len(data[i])):
				dataOut[3][j] += data[i][j]

		data[2][1] = 0
		data[3][1] = 1

		# deal with the density
		densOut.append(dens[0])
		for i in range(1, label):
			for j in range(len(dens[i])):
				densOut[0][j] += dens[i][j]

		densOut.append(dens[label])
		for i in range(label+1, len(dens)):
			for j in range(len(dens[i])):
				densOut[1][j] += dens[i][j]

		# normaize
		for i in range(len(densOut)):
			for j in range(len(densOut[i])):
				if densOut[i][j]>0:
					dataOut[i+2][j+2] /= densOut[i][j]

		# print
		with open(path + fileOut, "w") as fout:
			fout.write("Joint effect table\n")
			fout.write("rows: \t" + titleRow + "\n")
			fout.write("columns: \t" + titleCol + "\n")
			fout.write("" + splitAttr + " < " + str(splitVal) + "\n")
			fout.write("" + splitAttr + " >= " + str(splitVal) + "\n")
			fout.write("First row/column - quantile counts. Second row/column - quantile centers. Ignore four zeros in upper left corner.\n")
			fout.write("\n")
			for i in range(len(dataOut)):
				for j in range(len(dataOut[i])):
					if j>0:
						fout.write("\t")
					if(math.isinf(dataOut[i][j])):
						fout.write("?")
					else:
						fout.write(str(dataOut[i][j]))
				fout.write("\n")

		with open(path + fileDensOut, "w") as fout:
			fout.write("Density table: proportion of data around each quantile point \n")
			fout.write("rows: \t" + titleRow + "\n")
			fout.write("columns: \t" + titleCol + "\n")
			fout.write("\n")
			for i in range(len(densOut)):
				for j in range(len(densOut[i])):
					fout.write("\t")
					fout.write(str(densOut[i][j]))		
				fout.write("\n")
