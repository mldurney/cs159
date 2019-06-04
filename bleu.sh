#!/bin/bash

if [ $# -ne 6 ]
then
    echo "usage: $0 <from_language> <to_language> <config> <model> <data_dir> <checkpoint_dir>"
    exit 1
fi

FROM_LANG=$1
TO_LANG=$2
CONFIG=$3
MODEL=$4
DATA_DIR=$5
CHECKPOINT_DIR=$6

export CUDA_VISIBLE_DEVICES=0

wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

if [ "$FROM_LANG" == "en" ]; then
  PREDICTIONS=$DATA_DIR/$FROM_LANG-$TO_LANG-sentences_$TO_LANG_predictions.txt
  VALIDATION=$DATA_DIR/$FROM_LANG-$TO_LANG-sentences_$TO_LANG_val.txt
  PREDICTIONS_DECODED=$DATA_DIR/$FROM_LANG-$TO_LANG-sentences_$TO_LANG_predictions.detok.txt
  VALIDATION_DECODED=$DATA_DIR/$FROM_LANG-$TO_LANG-sentences_$TO_LANG_val.detok.txt
else
  PREDICTIONS=$DATA_DIR/$TO_LANG-$FROM_LANG-sentences_$TO_LANG_predictions.txt
  VALIDATION=$DATA_DIR/$TO_LANG-$FROM_LANG-sentences_$TO_LANG_val.txt
  PREDICTIONS_DECODED=$DATA_DIR/$TO_LANG-$FROM_LANG-sentences_$TO_LANG_predictions.detok.txt
  VALIDATION_DECODED=$DATA_DIR/$TO_LANG-$FROM_LANG-sentences_$TO_LANG_val.detok.txt
fi

onmt-main infer \
        --config $CONFIG --auto_config \
        --checkpoint_path=$CHECKPOINT_DIR \
        > $PREDICTIONS

spm_decode --model=$MODEL --input_format=piece \
            < data/$VALIDATION \
            > data/$VALIDATION_DECODED

spm_decode --model=$MODEL --input_format=piece \
            < data/$PREDICTIONS \
            > data/$PREDICTIONS_DECODED

perl multi-bleu-detok.perl $VALIDATION_DECODED < $PREDICTIONS_DECODED
