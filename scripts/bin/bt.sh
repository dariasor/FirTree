#!/bin/bash
# Script for fast feature evaluation

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

VERSION=`$BT_TRAIN -version 2> /dev/null | tail -1 | cut -f3 -d ' '`
if [ $(version $VERSION) -lt $(version "2.7.0") ]; then
    echo "Error: Old bt_train version. Need TreeExtra version 2.7.0 or higher."
    exit 1
fi

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
