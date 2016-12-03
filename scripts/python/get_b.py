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
	if line.endswith("bagging iterations"):
		data = lines[i].strip().split()
		print "-b " + str(int(data[0]) + 40)
		break
