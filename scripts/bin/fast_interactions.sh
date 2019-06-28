#!/bin/bash
# Script for interaction detection

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

VERSION=`$AG_TRAIN -version 2> /dev/null | tail -1 | cut -f3 -d ' '`
if [ $(version $VERSION) -lt $(version "2.5.5") ]; then
    echo "Error: Old ag_train version. Need TreeExtra version 2.5.5 or higher."
    exit 1
fi

if [ $# -eq 4 ]
then
	ATTR_FILE=$1
	ATTR_FS_FILE=$2
	TRAIN_FILE=$3
	VALID_FILE=$4
else
	echo "usage: fast_interactions.sh <attr> <attr.fs> <train> <valid>"
	exit 1
fi

START=$(date +%s)

PARSE_ACTION=$PYTHON_BIN/parse_action.py
PARSE_PARAMS=$PYTHON_BIN/parse_params.py
PARSE_PERFORMANCE=$PYTHON_BIN/parse_performance.py
PARSE_INTERACTIONS=$PYTHON_BIN/parse_interactions.py
GET_BEST_PARAMS=$PYTHON_BIN/get_best_params.py

$AG_TRAIN -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE -s layered > /dev/null


while :
do
        $PYTHON $PARSE_ACTION log.txt > parsed_action.txt
        PARAMS=`tail -1 parsed_action.txt`
        ACTION=`head -1 parsed_action.txt`
        if [ $ACTION = "ag_save" ]
        then
                $AG_SAVE $PARAMS > /dev/null
                break
        elif [ $ACTION = "ag_expand" ]
        then
                $PYTHON $GET_BEST_PARAMS log.txt > best_params.txt
                IFCONV=`head -1 best_params.txt`
                BEST_PARAMS=`tail -1 best_params.txt`
                if [ $IFCONV = "True" ]
                then
                    $AG_EXPAND $BEST_PARAMS > /dev/null
                else
                    $AG_SAVE $BEST_PARAMS > /dev/null
                    break
                fi
        else
                echo "ag_train or ag_expand has not finished properly"
                exit 1
        fi
done


$PYTHON $PARSE_PARAMS log.txt > params_fs.txt
PARAMS_FS=`tail -1 params_fs.txt`

$AG_FS -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FILE $PARAMS_FS > /dev/null

$PYTHON $PARSE_PERFORMANCE log.txt > params_interaction.txt
PARAMS_INTERACTION=`tail -1 params_interaction.txt`

$AG_INTERACTIONS -t $TRAIN_FILE -v $VALID_FILE -r $ATTR_FS_FILE $PARAMS_FS $PARAMS_INTERACTION > /dev/null

$PYTHON $PARSE_INTERACTIONS log.txt > list.txt

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Time: $DIFF seconds" > time_grove.log

