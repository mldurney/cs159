#!/bin/bash

# SIZE="" # Entire validation set
SIZE="_size_3000" # Reduced validation set

if [ $# -ne 8 ]
then
    echo "usage: $0 <t [Transformer] | r [RNN]> <from_language> <to_language> <config> <model> <data_dir> <checkpoint_dir> <group_name>"
    exit 1
fi

if [ $1 == "t" ]; then
    MODEL_TYPE="Transformer"
    MODEL_NAME="transformer"
else
    MODEL_TYPE="NMTBig"
    MODEL_NAME="rnn"
fi
FROM_LANG=$2
TO_LANG=$3
CONFIG=$4
MODEL=$5
DATA_DIR=$6
CHECKPOINT_DIR=$7
GROUP_NAME=$8

ONMT_EXE=/mnt/c/Users/megan/Anaconda3/Scripts/onmt-main.exe
SP_PATH=../sentencepiece
export PATH=$SP_PATH:$PATH
export CUDA_VISIBLE_DEVICES=0

wget --quiet -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/input-from-sgm.perl
wget --quiet -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

PREDICTIONS_DIR=predictions/$GROUP_NAME/$MODEL_NAME\_$FROM_LANG\_to\_$TO_LANG\_checkpoints$SIZE
SCORES=$PREDICTIONS_DIR/bleu\_scores.txt

if [ ! -d $PREDICTIONS_DIR ]; then
  mkdir -p $PREDICTIONS_DIR
fi

if [ -f $SCORES ]; then
  cat /dev/null > $SCORES
else
  touch $SCORES
fi

if [ "$FROM_LANG" == "en" ]; then
  INPUT=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$FROM_LANG_val$SIZE.txt
  VALIDATION=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$TO_LANG\_val$SIZE.txt
  VALIDATION_DECODED=$DATA_DIR/$FROM_LANG-$TO_LANG\_sentences\_$TO_LANG\_val$SIZE.detok.txt
else
  INPUT=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$FROM_LANG\_val$SIZE.txt
  VALIDATION=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$TO_LANG\_val$SIZE.txt
  VALIDATION_DECODED=$DATA_DIR/$TO_LANG-$FROM_LANG\_sentences\_$TO_LANG\_val$SIZE.detok.txt
fi

if [ ! -f $VALIDATION_DECODED ]; then
  spm_decode --model=$MODEL --input_format=piece \
              < $VALIDATION \
              > $VALIDATION_DECODED \
              2> err_decode1.txt
fi

CHECKPOINT_DIR_WSL=$(wslpath -a $CHECKPOINT_DIR)
for FILE in `ls -v1 $CHECKPOINT_DIR_WSL/*.index`; do
  CHECKPOINT=$(basename $FILE .index)
  CHECKPOINT_ID=${CHECKPOINT#"model."}

  if [ $CHECKPOINT_ID == "ckpt-0" ]; then
    continue
  fi

  if [ ! -f $CHECKPOINT_DIR_WSL/checkpoint\_real ]; then
    mv $CHECKPOINT_DIR_WSL/checkpoint $CHECKPOINT_DIR_WSL/checkpoint\_real
  fi

  {
    echo "model_checkpoint_path: \"$CHECKPOINT\""
    echo "all_model_checkpoint_paths: \"$CHECKPOINT\""
  } > $CHECKPOINT_DIR_WSL/checkpoint

  PREDICTIONS=$PREDICTIONS_DIR/$CHECKPOINT_ID\_predictions$SIZE.txt
  PREDICTIONS_DECODED=$PREDICTIONS_DIR/$CHECKPOINT_ID\_predictions$SIZE.detok.txt

  if [ -f $PREDICTIONS ]; then
    cat /dev/null > $PREDICTIONS
  else
    touch $PREDICTIONS
  fi

  $ONMT_EXE infer \
          --model_type $MODEL_TYPE \
          --config $CONFIG --auto_config \
          --features_file $INPUT \
          --predictions_file $PREDICTIONS \
          --checkpoint_path $CHECKPOINT_DIR \
          2> err_onmt.txt

  spm_decode --model=$MODEL --input_format=piece \
            < $PREDICTIONS \
            > $PREDICTIONS_DECODED\
            2> err_decode2.txt

  perl multi-bleu-detok.perl $VALIDATION_DECODED < $PREDICTIONS_DECODED >> $SCORES

  mv $CHECKPOINT_DIR_WSL/checkpoint\_real $CHECKPOINT_DIR_WSL/checkpoint
done
