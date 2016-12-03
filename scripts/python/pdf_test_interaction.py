import sys
import os

if len(sys.argv) != 4:
	sys.exit("usage: " + sys.argv[0] + " <train> <nval> <R program>")
out = open(sys.argv[3], "w")
out.write("platform=\"linux\"\n")
out.write("rfhome=\"/home/yinlou/rulefit\"\n")
out.write("source(\"/home/yinlou/rulefit/rulefit.r\")\n")
out.write("library(akima, lib.loc=rfhome)\n")
out.write("trainingSet <- read.table(\"" + sys.argv[1] + "\")\n")
out.write("len <- length(names(trainingSet))\n")
out.write("rfmod <- rulefit(trainingSet, len)\n")

f = open(sys.argv[1], "r")
data = f.readline().rstrip().split("\t")
length = len(data)

for i in range(1, length - 1):
	out.write("int2var <- twovarint(" + str(i) + ", c(" + str(i + 1))
	for j in range(i + 1, length - 1):
		out.write("," + str(j + 1))
	out.write("), nval=" + sys.argv[2] + ")\n")
	out.write("int2var\n")
out.flush()
out.close()
