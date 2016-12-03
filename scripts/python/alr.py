import sys
import os

if len(sys.argv) != 4:
	sys.exit("usage: " + sys.argv[0] + " <train> <test> <limit>")
limit = int(sys.argv[3])
out = open("gam.R", "w")
out.write("library(mgcv)\n")
out.write("trainingSet <- read.table(\"" + sys.argv[1] + "\")\n")
out.write("testSet <- read.table(\"" + sys.argv[2] + "\")\n")
out.write("len <- length(names(trainingSet))\n")
out.write("target <- names(trainingSet)[len]\n")

f = open(sys.argv[1], "r")
l = []
for line in f:
	data = line.rstrip().split()
	if len(l) == 0:
		for i in range(len(data) - 1):
			l.append(set())
	for i in range(1, len(data) - 1):
		d = float(data[i])
		l[i].add(d)
length = len(l) + 1
s = set()
for i in range(len(l)):
	if len(l[i]) < limit:
		s.add(i)
formula = "V" + str(length) + "~"
for i in range(length - 1):
	if i == length - 2:
		if i in s:
			formula += "V" + str(i + 1)
		else:
			formula += "s(V" + str(i + 1) + ")"
	else:
		if i in s:
			formula += "V" + str(i + 1) + "+"
		else:
			formula += "s(V" + str(i + 1) + ")+"
out.write("model <- gam(" + formula + ", data = trainingSet, family=binomial(logit))\n")
out.write("pred <- predict(model, testSet)\n")
out.write("prob <- 1 / (1 + exp(-pred))\n")
out.write("write.table(prob, file=\"gam_preds.txt\", row.names=F, col.names=F)\n")
out.write("prob <- round(prob)\n")
out.write("error <- 0.0\n")
out.write("for (i in 1:length(prob)) {\n")
out.write("\tif (prob[i] != testSet[i, target]) error <- error + 1\n")
out.write("}\n")
out.write("error <- error / length(prob)\n")
out.write("error\n")
