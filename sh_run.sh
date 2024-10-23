#!/bin/sh

ROOTDIR=$PWD
DATA_GENERATOR_DIR=$ROOTDIR/projects/training-data-generator
TRAINING_LOGIC_DIR=$ROOTDIR/projects/training-logic

#
#
# build training-data-generator

cd "$DATA_GENERATOR_DIR" || exit 1
# make fclean
make all -j4 || exit 1
cd "$ROOTDIR" || exit 1

#
#
# build training-logic

cd "$TRAINING_LOGIC_DIR" || exit 1
# make fclean
make all -j4 || exit 1
cd "$ROOTDIR" || exit 1

#
#
# generator training data in the asset folder of the training logic

cd "$DATA_GENERATOR_DIR" || exit 1
mkdir -p $TRAINING_LOGIC_DIR/assets/
./bin/exec and > "$TRAINING_LOGIC_DIR/assets/training-data-and-gate.txt"
./bin/exec no > "$TRAINING_LOGIC_DIR/assets/training-data-no-gate.txt"
./bin/exec or > "$TRAINING_LOGIC_DIR/assets/training-data-or-gate.txt"
./bin/exec xor > "$TRAINING_LOGIC_DIR/assets/training-data-xor-gate.txt"
cd "$ROOTDIR" || exit 1

ls -lah "$TRAINING_LOGIC_DIR/assets/"



cd "$TRAINING_LOGIC_DIR" || exit 1
# ./bin/exec "$TRAINING_LOGIC_DIR/assets/training-data-and-gate.txt"
# ./bin/exec "$TRAINING_LOGIC_DIR/assets/training-data-no-gate.txt"
# ./bin/exec "$TRAINING_LOGIC_DIR/assets/training-data-or-gate.txt"
./bin/exec "$TRAINING_LOGIC_DIR/assets/training-data-xor-gate.txt"



