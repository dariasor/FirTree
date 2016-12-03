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
	if line.startswith("ag_save"):
		count = i - 2
		break
data = lines[count].split(" = ")
alpha = data[1]
data = lines[count - 1].split(" = ")
n = data[1]
data = lines[count - 2].split(" ")
b = data[0]
print "-a " + alpha + " -n " + n + " -b " + b
