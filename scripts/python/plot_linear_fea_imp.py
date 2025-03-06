#!/usr/bin/env python
from argparse import ArgumentParser
import os.path
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# check if input file exists
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    elif os.path.getsize(arg) == 0:
        parser.error("The file %s is empty!" % arg)
    else:
        return arg

parser = ArgumentParser(description="Plot feature importance for a linear model.")
parser.add_argument("-i", dest="in_filename", required=True,
                    help="path to input file with linear model parameters", metavar="MODEL_FILE_PATH",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument("-o", dest="out_filename", default="linear_feature_importance", 
                    help="(optional): specify an output name for the plot (default: linear_feature_importance)", metavar="PLOT_NAME")
parser.add_argument("-m", dest="model_name", default="", 
                    help="(optional): specify a model name to show on the plot", metavar="MODEL_NAME")

args = parser.parse_args()

with open(args.in_filename) as f:
    lines = f.readlines()[1:] # skip intercept
    feature = [line.split()[0] for line in lines]
    slope = [float(line.split()[1]) for line in lines]
    min_val = [float(line.split()[2]) for line in lines]
    max_val = [float(line.split()[3]) for line in lines]

val_diff = [v_max - v_min for v_max, v_min in zip(max_val, min_val)]
contribution = [s * v for s, v in zip(slope, val_diff)]
min_cont = [s * v for s, v in zip(slope, min_val)]
max_cont = [s * v for s, v in zip(slope, max_val)]

# 10 distinct colors for plotting
colors = ['cornflowerblue','forestgreen','salmon','orange','steelblue','peru','crimson','olivedrab','lightseagreen','orchid']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(feature)):
    x = [0,1]
    y = [min_cont[i], max_cont[i]]
    ax.plot(x, y, c = colors[i],label = str(round(contribution[i],2)) + ", " + feature[i] + " ({}, {})".format(int(min_val[i]), int(max_val[i])))
plt.title("Importance of features in linear model {}".format(args.model_name))
plt.axhline(y = 0, c = 'lightgrey',ls = '--')
plt.xlim(0,1)
plt.xlabel('Feature values normalized to [0,1]')
plt.ylabel('Contribution to model prediction')
handles, labels = ax.get_legend_handles_labels()
# sort labels and handles based on line order 
order = sorted(zip(max_cont, min_cont, handles, labels), key = lambda x: (x[0], x[1]), reverse = True)
sorted_handles = [t[2] for t in order]
sorted_legends = [t[3] for t in order]
plt.legend(sorted_handles, sorted_legends, loc='upper left')
plt.savefig('{}.png'.format(args.out_filename),dpi=200, bbox_inches='tight')





