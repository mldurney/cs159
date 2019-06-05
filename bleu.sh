#!/bin/bash


if [ $# -ne 7 ]
then
    echo "usage: $0 <t [Transformer] | r [RNN]> <from_language> <to_language> <config> <model> <data_dir> <checkpoint_dir>"
    exit 1
fi

if [ $1 == "t" ]; then
    MODEL_TYPE="Transformer"
else
    MODEL_TYPE="NMTBig"
fi
FROM_LANG=$2
TO_LANG=$3
CONFIG=$4
MODEL=$5
DATA_DIR=$6
CHECKPOINT_DIR=$7

SP_PATH=../sentencepiece
export PATH=$SP_PATH:$PATH
export CUDA_VISIBLE_DEVICES=0

wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/input-from-sgm.perl
wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

if [ "$FROM_LANG" == "en" ]; then
  INPUT=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$FROM_LANG_val.txt
  PREDICTIONS=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$TO_LANG\_predictions.txt
  VALIDATION=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$TO_LANG\_val.txt
  PREDICTIONS_DECODED=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$TO_LANG\_predictions.detok.txt
  VALIDATION_DECODED=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$TO_LANG\_val.detok.txt
else
  INPUT=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$FROM_LANG\_val.txt
  PREDICTIONS=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$TO_LANG\_predictions.txt
  VALIDATION=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$TO_LANG\_val.txt
  PREDICTIONS_DECODED=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$TO_LANG\_predictions.detok.txt
  VALIDATION_DECODED=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$TO_LANG\_val.detok.txt
fi

onmt-main infer \
        --model_type=$MODEL_TYPE \
        --config $CONFIG --auto_config \
        --features_file $INPUT \
        --checkpoint_path=$CHECKPOINT_DIR \
        > $PREDICTIONS

spm_decode --model=$MODEL --input_format=piece \
            < $VALIDATION \
            > $VALIDATION_DECODED

spm_decode --model=$MODEL --input_format=piece \
            < $PREDICTIONS \
            > $PREDICTIONS_DECODED

perl multi-bleu-detok.perl $VALIDATION_DECODED < $PREDICTIONS_DECODED
