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

$AG_TRAIN -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE -s layered -c roc > /dev/null

while :
do
	$PYTHON $PARSE_ACTION log.txt > parsed_action.txt
	PARAMS=`tail -1 parsed_action.txt`
	ACTION=`head -1 parsed_action.txt`
	if [ $ACTION = "ag_save" ]
	then
		$PYTHON $GET_BEST_PARAMS log.txt performance.txt > best_params.txt
		IFCONV=`head -1 best_params.txt`
		PARAMS=`tail -1 best_params.txt`
		if [ $IFCONV = "True" ]
		then
			$AG_SAVE $PARAMS > /dev/null
			break
		else
			$PYTHON $GET_B log.txt > b.txt
			B=`head -1 b.txt`
			$AG_EXPAND $B > /dev/null
		fi
	elif [ $ACTION = "ag_expand" ]
	then
		$AG_EXPAND $PARAMS > /dev/null
	else
		echo "ag_expand has not finished properly"
		exit 1
	fi
done

$PYTHON $PARSE_PARAMS log.txt > params_fs.txt
PARAMS_FS=`tail -1 params_fs.txt`

$AG_FS -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE -c roc $PARAMS_FS > /dev/null

$PYTHON $PARSE_PERFORMANCE log.txt > params_interaction.txt
PARAMS_INTERACTION=`tail -1 params_interaction.txt`

$AG_INTERACTIONS -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FS_FILE $PARAMS_FS -c roc $PARAMS_INTERACTION > /dev/null

$PYTHON $PARSE_INTERACTIONS log.txt > list.txt

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Time: $DIFF seconds" > time_grove.log
rm -fr AGTemp
