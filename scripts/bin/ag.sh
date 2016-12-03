#!/bin/bash
# Script for running additive groves

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	TRAIN_FILE=$2
	VALID_FILE=$3
	TEST_FILE=$4
else
	echo "usage: ag <attr> <train> <valid> <test>"
	exit 1
fi

PARSE_ACTION=$PYTHON_BIN/parse_action.py
PARSE_PARAMS=$PYTHON_BIN/parse_params.py
PARSE_PERFORMANCE=$PYTHON_BIN/parse_performance.py

START=$(date +%s)
$AG_TRAIN -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE

while :
do
	$PYTHON $PARSE_ACTION log.txt > parsed_action.txt
	ACTION=`head -1 parsed_action.txt`
	PARAMS=`tail -1 parsed_action.txt`
	if [ $ACTION = "ag_save" ]
	then
		$AG_SAVE $PARAMS
		break
	elif [ $ACTION = "ag_expand" ]
	then
		$AG_EXPAND $PARAMS
	fi
done
END=$(date +%s)
DIFF=$(( $END - $START ))

$AG_PREDICT -p $TEST_FILE -r $ATTR_FILE > pred.log
rm -fr AGTemp
tail -1 pred.log
echo "Time: $DIFF seconds."

