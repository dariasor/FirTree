import sys

if len(sys.argv) != 3 and len(sys.argv) != 4:
	sys.exit("usage: " + sys.argv[0] + " <log.txt> <attr> [<attr.fs>]")
attr = {}
attributes = []
f = open(sys.argv[2], "r")
count = 0
for line in f:
	if line.startswith("contexts:"):
		break
	data = line.rstrip().split(": ")
	attr[data[0]] = count
	count += 1
	attributes.append(data[0])
if len(sys.argv) == 4:
	f = open(sys.argv[3], "r")
	data = line.rstrip().split(": ")
	idx = attr[data[0]]
	print str(idx)
lines = []
f = open(sys.argv[1], "r")
for line in f:
	lines.append(line.rstrip())
lines.reverse()
count = -1
pairs = []
for i in range(len(lines)):
	line = lines[i]
	if line.startswith("ag_interactions"):
		break
	elif line.endswith("is present.") or line.endswith("is absent."):
		infoline = lines[i + 1]
		data = infoline.split(" ")
		v1 = data[3]
		v2 = data[5]
		idx1 = attr[v1] - 1
		idx2 = attr[v2] - 1
		data = line.strip().split()
		imp = float(data[2])
		pairs.append((str(idx1), str(idx2), imp))
sortedPairs = sorted(pairs, key=lambda e:-e[2])

for pair in sortedPairs:
	print pair[0] + " " + pair[1] + " " + str(pair[2]) + " " + attributes[int(pair[0]) + 1] + " " + attributes[int(pair[1]) + 1]
