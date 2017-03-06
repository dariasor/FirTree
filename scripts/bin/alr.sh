#!/bin/bash

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 3 ]
then
	TRAIN=$1
	TEST=$2
	LIMIT=$3
else
	echo "usage: gam.sh <train> <test> <limit>"
	exit 1
fi

GAM=$PYTHON_BIN/alr.py

"$PYTHON" $GAM $TRAIN $TEST $LIMIT
time "$R" --slave < gam.R > mgcv.log
