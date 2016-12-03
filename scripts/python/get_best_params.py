import sys

if len(sys.argv) != 3:
	sys.exit("usage: " + sys.argv[0] + " <log.txt> <performance.txt>")
lines = []
f = open(sys.argv[1], "r")
for line in f:
	lines.append(line.rstrip())
lines.reverse()
count = -1
for i in range(len(lines)):
	line = lines[i]
	if line.startswith("Suggested action: ag_save"):
		count = i
		break
data = lines[count + 4].strip().split(" = ")
n = int(data[1])
data = lines[count + 5].strip().split(" = ")
alpha = data[1]

d = {}
cflag = {}  # flags for converence

f = open(sys.argv[2], 'r')
pLines = f.readlines()
for line in pLines:
	if line.strip() == "":
		break
	else:
		data = line.strip().split()
		if data[1] not in d:
			d[data[1]] = {}
			cflag[data[1]] = {}
		d[data[1]][data[0]] = float(data[2])
		if int(data[3]):
			cflag[data[1]][data[0]] = True
		else:
			cflag[data[1]][data[0]] = False

# build up the overfitting zone
rownames = ["0.5", "0.2", "0.1", "0.05", "0.02", "0.01", "0.005", "0.002", "0.001", "0.0005", "0.0002", "0.0001", "0.00005", "0.00002", "0.00001"]
cut = 0
for cut in range(len(rownames)):
	if not rownames[cut] in d['1']:
		break;
rownames = rownames[0:cut]
if n>= 16:
	colnames = ["1", "2", "3", "4", "6", "8", "16"]
else:
	colnames = ["1", "2", "3", "4", "6", "8"]

# filter out the overfitting cases
overfitting = {}
for i in colnames:
	overfitting[i] = {}
for i in range(len(colnames)):
	for j in range(len(rownames)):
		overfitting[colnames[i]][rownames[j]] = False
		if i > 0:
			if (overfitting[colnames[i-1]][rownames[j]]) or (d[colnames[i]][rownames[j]] < d[colnames[i-1]][rownames[j]]):
				overfitting[colnames[i]][rownames[j]] = True
		if j > 0:
			if (overfitting[colnames[i]][rownames[j-1]]) or (d[colnames[i]][rownames[j]] < d[colnames[i]][rownames[j-1]]):
				overfitting[colnames[i]][rownames[j]] = True
for i in colnames:
	for j in rownames:
		if overfitting[i][j]:
			d[i][j] = 0   # set overfitting values to 0 to avoid selection

bestROC6 = max(d['6'].values())
bestROC8 = max(d['8'].values())
if n >= 16:   # if recommended n is greater than 16, set n = 16
	bestROC16 = max(d['16'].values())
	d16 = d['16']
	bestAlpha = None
	for key in d16:
		if d16[key] == bestROC16:
			bestAlpha = key
			break
	print cflag['16'][bestAlpha]   # print if convergent
	print "-a " + bestAlpha + " -n 16"
elif n >= 8:  # if recommended n is between 8 and 16, select itself
	print cflag[str(n)][alpha]
	print "-a " + alpha + " -n " + str(n)
elif (bestROC8 >= bestROC6 - 0.001) and (bestROC8 > 0):  # if recommended n is too small, choose n = either 6 or 8
	d8 = d['8']
	bestAlpha = None
	for key in d8:
		if d8[key] == bestROC8:
			bestAlpha = key
			break
	print cflag['8'][bestAlpha]
	print "-a " + bestAlpha + " -n 8"
elif bestROC6 > 0:  # 
	d6 = d['6']
	bestAlpha = None
	for key in d6:
		if d6[key] == bestROC6:
			bestAlpha = key
			break
	print cflag['6'][bestAlpha]
	print "-a " + bestAlpha + " -n 6"
else:
	print cflag['6']['0.5']
	print "-a 0.5 -n 6"

