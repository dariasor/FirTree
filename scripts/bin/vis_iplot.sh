#!/bin/bash
# Script for interaction detection

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	TRAIN_FILE=$2
	FEATURE1=$3
	FEATURE2=$4
else
	echo "usage: vis_effect.sh <attr> <train> <feature1> <feature2>"
	exit 1
fi

$VIS_IPLOT -v $TRAIN_FILE -r $ATTR_FILE -f1 $FEATURE1 -f2 $FEATURE2 > /dev/null
