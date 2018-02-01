#!/bin/bash
# Script for moving plot files

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

if [ $# -eq 1 ]
then
	DIR_NAME=$1
else
	echo "usage: vis_mv <dir>"
	exit 1
fi

mkdir -p $DIR_NAME
mv *.iplot.txt $DIR_NAME 2> /dev/null
mv *.iplot.dens.txt $DIR_NAME 2> /dev/null
mv *.effect.txt $DIR_NAME 2> /dev/null
mv log.txt $DIR_NAME 2> /dev/null

exit 0 #the script should not fail if there is nothing to move
