import sys

if len(sys.argv) != 2:
	sys.exit("usage: " + sys.argv[0] + " <log.txt>")
lines = []
f = open(sys.argv[1], "r")
for line in f:
	lines.append(line.rstrip())
lines.reverse()
count = -1
for i in range(len(lines)):
	line = lines[i]
	if line.startswith("Average performance:"):
		strs = line.split(", ")
		data = strs[0].split(": ")
		ave = data[1]
		data = strs[1].split(": ")
		std = data[1]
		print "-ave " + ave + " -std " + std
		break

