#!/bin/bash
# Script for effects visualization

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

VERSION=`$VIS_EFFECT -version 2> /dev/null | tail -1 | cut -f3 -d ' '`
if [ $(version $VERSION) -lt $(version "2.5.4") ]; then
    echo "Error: Old vis_effect version. Need TreeExtra version 2.5.4 or higher."
    exit 1
fi

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	TRAIN_FILE=$2
	FEATURE=$3
    SUFFIX=$4
else
	echo "usage: vis_effect.sh <attr> <train> <feature> <suffix>"
	exit 1
fi

$VIS_EFFECT -v $TRAIN_FILE -r $ATTR_FILE -f $FEATURE -o $SUFFIX -q 30 > /dev/null
