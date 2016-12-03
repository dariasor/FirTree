import sys

if len(sys.argv) != 2:
	sys.exit("usage: " + sys.argv[0] + " <log.txt>")
lines = []
f = open(sys.argv[1], "r")
for line in f:
	lines.append(line.rstrip())
lines.reverse()
for i in range(len(lines)):
	line = lines[i]
	if line.startswith("Suggested action"):
		strs = line.split(": ")
		idx = strs[1].find(" ")
		action = strs[1][0:idx]
		params = strs[1][(idx + 1):]
		print action + "\n" + params
		break
