#!/bin/bash
# Script for interaction detection

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 3 ]
then
	ATTR_FILE=$1
	TRAIN_FILE=$2
	FEATURE=$3
else
	echo "usage: vis_effect.sh <attr> <train> <feature>"
	exit 1
fi

$VIS_EFFECT -v $TRAIN_FILE -r $ATTR_FILE -f $FEATURE > /dev/null
