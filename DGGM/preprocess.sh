#!bin/bash

max=1
prefix='data/data'
for i in `seq 0 $max`
do
    dirname="$prefix$i"
    python -W ignore prep.py $dirname
    echo $dirname

    echo "Continue? [n to end, any other character to continue]..."
    read usrinp

    if [ $usrinp = "n" ]; then
    	echo Goodbye.
    	exit 0;
    fi
done
echo 
echo Preprocessing complete.
echo See train_aae_final for training set and test_aae_final for test set.
echo 
echo Goodbye.