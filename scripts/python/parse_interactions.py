import sys

if len(sys.argv) != 2:
	sys.exit("usage: " + sys.argv[0] + " <log.txt>")
lines = []
f = open(sys.argv[1], "r")
for line in f:
	lines.append(line.rstrip())
lines.reverse()
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
		data = line.strip().split()
		imp = float(data[2])
		pairs.append((v1, v2, imp))
sortedPairs = sorted(pairs, key=lambda e:-e[2])

for pair in sortedPairs:
	print pair[0] + " " + pair[1] + " " + str(pair[2]);
