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

mkdir $DIR_NAME
mv *.iplot.txt $DIR_NAME
mv *.iplot.dens.txt $DIR_NAME
