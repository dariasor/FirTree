#!/bin/bash
# Script for splitting the dataset

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
source $SCRIPTPATH/env.config

$RND $1 > /dev/null
