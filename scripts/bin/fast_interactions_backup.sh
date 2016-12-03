#!/bin/bash
# Script for interaction detection

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	ATTR_FS_FILE=$2
	TRAIN_FILE=$3
	VALID_FILE=$4
else
	echo "usage: interaction.sh <attr> <attr.fs> <train> <valid>"
	exit 1
fi

START=$(date +%s)

PARSE_ACTION=$PYTHON_BIN/parse_action.py
PARSE_PARAMS=$PYTHON_BIN/parse_params.py
PARSE_PERFORMANCE=$PYTHON_BIN/parse_performance.py
PARSE_INTERACTIONS=$PYTHON_BIN/parse_interactions.py
GET_BEST_PARAMS=$PYTHON_BIN/get_best_params.py
GET_B=$PYTHON_BIN/get_b.py

$FAST_AG_TRAIN -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE -s layered -c roc > /dev/null

SAVED_ONCE="FALSE"

while :
do
	$PYTHON $PARSE_ACTION log.txt > parsed_action.txt
	PARAMS=`tail -1 parsed_action.txt`
	ACTION=`head -1 parsed_action.txt`
	if [ $ACTION = "ag_save" ]
	then
		if [ $SAVED_ONCE = "FALSE" ]
		then
			$PYTHON $GET_B log.txt > b.txt
			B=`head -1 b.txt`
			$FAST_AG_EXPAND $B > /dev/null
			SAVED_ONCE="TRUE"
		else
			$PYTHON $GET_BEST_PARAMS log.txt performance.txt > best_params.txt
			PARAMS=`head -1 best_params.txt`
			$FAST_AG_SAVE $PARAMS > /dev/null
			break
		fi
	elif [ $ACTION = "ag_expand" ]
	then
		$FAST_AG_EXPAND $PARAMS > /dev/null
		SAVED_ONCE="FALSE"
	else
		echo "ag_expand has not finished properly"
		exit 1
	fi
done

$PYTHON $PARSE_PARAMS log.txt > params_fs.txt
PARAMS_FS=`tail -1 params_fs.txt`

$FAST_AG_FS -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE -c roc $PARAMS_FS > /dev/null

$PYTHON $PARSE_PERFORMANCE log.txt > params_interaction.txt
PARAMS_INTERACTION=`tail -1 params_interaction.txt`

$FAST_AG_INTERACTIONS -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FS_FILE $PARAMS_FS -c roc $PARAMS_INTERACTION > /dev/null

$PYTHON $PARSE_INTERACTIONS log.txt > list.txt

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Time: $DIFF seconds" > time_grove.log
rm -fr AGTemp
