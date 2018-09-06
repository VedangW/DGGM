#!bin/bash

# A script to 

# Total number of data folders
max=12

# Prefix to attach
prefix='data/data'

for i in `seq 0 $max`
do
    dirname="$prefix$i"

    # Run py file here
    python -W ignore prep.py $dirname
    echo $dirname

    # Ask user to continue
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