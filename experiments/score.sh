#!/bin/bash

if [ $# == 3 ]; then
    SCORES=$3

    if [ -f $SCORES ]; then
    cat /dev/null > $SCORES
    else
    touch $SCORES
    fi
elif [ $# -ne 2 ]; then
    echo "usage: $0 <validation> <predictions> <opt: scores>"
    exit 1
fi

VALIDATION=$1
PREDICTIONS=$2

wget -nc --quiet https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

if [ $# == 3 ]; then
    perl multi-bleu-detok.perl $VALIDATION < $PREDICTIONS >> $SCORES
    cat $SCORES
else
    perl multi-bleu-detok.perl $VALIDATION < $PREDICTIONS
fi
