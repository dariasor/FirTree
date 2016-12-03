#!/bin/bash
# Script for splitting the dataset

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 5 ]
then
	DATA=$1
	PREFIX=$2
	PORTION_VAL=$3
	PORTION_TRAIN=$4
	GROUP=$5
else
	echo "usage: rnd.sh <data> <prefix> <portion_validation> <portion_train> <group_col_no>"
	exit 1
fi

$RND --input $DATA --stem $PREFIX --group $GROUP --valid $PORTION_VAL --train $PORTION_TRAIN > /dev/null