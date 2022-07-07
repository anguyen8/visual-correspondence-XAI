#!/bin/sh

helpFunction()
{
   echo ""
   echo "Usage: $0 -dataset [DATASET] -vis [YES/NO]"
   echo -e "\t-dataset Specify what dataset to test. e.g., imagenet1k-val. More in datasets.py!"
   echo -e "\t-inat Boolean value. Using iNaturalist features for CUB200 or not? \n \
   \t Set to False when running ImageNet datasets."

   exit 1 # Exit script after printing help
}

while getopts "d:v:i:" opt
do
   case "$opt" in
      d ) parameterA="$OPTARG" ;;
      i ) parameterC="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z "$parameterC" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# input dataset and visualization bool value
CUDA_VISIBLE_DEVICES=0,1,2,3, python src/ImageNets_KNN.py "$parameterA" "$parameterC"
