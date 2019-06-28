import sys

if len(sys.argv) != 2:
	sys.exit("usage: " + sys.argv[0] + " <log.txt> ")
lines = []
f = open(sys.argv[1], "r")
for line in f:
	lines.append(line.rstrip())
lines.reverse()
count = -1
rec_params = {}

for i in range(len(lines)):
	line = lines[i]
	if line.startswith("Suggested action: ag_expand"):
		count = i
		terms = line.split(" ");
		for termNo in range(4, len(terms), 2):
			rec_params[terms[termNo - 1]] = terms[termNo]
		break
data = lines[count + 4].strip().split(" = ")
best_n = data[1]
data = lines[count + 5].strip().split(" = ")
best_alpha = data[1]

do_expand = True
if "-n" in rec_params and int(rec_params["-n"]) > 16:
	del rec_params["-n"]
	if not rec_params:
		print "False"
		print "-n " + best_n + " -a " + best_alpha
		do_expand = False
if do_expand:
	print "True"
	param_str = ""
	for key in rec_params:
		param_str += key + " " + rec_params[key] + " "
	print param_str


