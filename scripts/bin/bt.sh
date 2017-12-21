#!/bin/bash
# Script for fast feature evaluation

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	TRAIN_FILE=$2
	VALID_FILE=$3
	PARAMS=$4
else
	echo "usage: bt.sh <attr> <train> <valid> <other parameters>"
	exit 1
fi

$BT_TRAIN -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE $PARAMS > /dev/null
