#!/bin/bash
# Script for moving plot files

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 3 ]
then
	DIR_NAME=$1
	SPLIT_ATTR=$2
	SPLIT_VAL=$3
else
	echo "usage: vis_split <dir> <split_attr> <split_val>"
	exit 1
fi

VIS_SPLIT=$PYTHON_BIN/parse_iplot.py
$PYTHON $VIS_SPLIT $DIR_NAME $SPLIT_ATTR $SPLIT_VAL

