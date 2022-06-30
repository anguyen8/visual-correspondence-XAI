#!/bin/sh

helpFunction()
{
   echo ""
   echo "Usage: $0 -dataset [DATASET] -vis [YES/NO]"
   echo -e "\t-dataset Specify what dataset to test. e.g., imagenet1k-val. More in datasets.py!"
   echo -e "\t-vis Boolean value. Visualize plots or not?"
   echo -e "\t-inat Boolean value. Using iNaturalist features for CUB200 or not? \n \
   \t Set to False when running ImageNet datasets."

   exit 1 # Exit script after printing help
}

while getopts "d:v:i:" opt
do
   case "$opt" in
      d ) parameterA="$OPTARG" ;;
      v ) parameterB="$OPTARG" ;;
      i ) parameterC="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z "$parameterB" ] || [ -z "$parameterC" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# input dataset and visualization bool value
python ImageNets_KNN.py "$parameterA" "$parameterC"
python ImageNets_EMD.py "$parameterA" "$parameterB" "$parameterC"