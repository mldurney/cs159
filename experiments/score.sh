#!/bin/bash

if [ $# -ne 2 ]
then
    echo "usage: $0 <validation> <predictions>"
    exit 1
fi

VALIDATION=$1
PREDICTIONS=$2

wget -nc --quiet https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

perl multi-bleu-detok.perl $VALIDATION < $PREDICTIONS
