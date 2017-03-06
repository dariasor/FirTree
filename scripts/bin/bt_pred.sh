#!/bin/bash
# Script for bagged tree prediction.

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	TEST_FILE=$2
	OUPUT_FILE=$3
	PARAMS=$4
else
	echo "usage: bt_pred.sh <attr> <test> <output> <other params>"
	exit 1
fi

$BT_PREDICT -p $TEST_FILE -r $ATTR_FILE -o $OUPUT_FILE $PARAMS
