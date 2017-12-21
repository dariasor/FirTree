#!/bin/bash
# Script for interaction detection

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

VERSION=`$VIS_IPLOT -version 2> /dev/null | tail -1 | cut -f3 -d ' '`
if [ $(version $VERSION) -lt $(version "2.5.4") ]; then
    echo "Error: Old vis_iplot version. Need TreeExtra version 2.5.4 or higher."
    exit 1
fi

if [ $# -eq 5 ]
then
	ATTR_FILE=$1
	TRAIN_FILE=$2
	FEATURE1=$3
	FEATURE2=$4
    SUFFIX=$5
else
	echo "usage: vis_effect.sh <attr> <train> <feature1> <feature2> <suffix>"
	exit 1
fi

$VIS_IPLOT -v $TRAIN_FILE -r $ATTR_FILE -f1 $FEATURE1 -f2 $FEATURE2 -o $SUFFIX > /dev/null
