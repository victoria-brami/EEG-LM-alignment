#! /bin/bash

# conda activate genie3.10

#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_s_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]
cd ../
for list_elt in "OBJECT" "FEELING" "LOCATION"  "MONEY" "NATURE" \
                "QUANTITY" "HOUSE" "MOVE"  \
                 "FOOD"  "PEOPLE" \
                "MODE" "DEVICE"  "TIME" \
                 "DEATH" "BODY" "PEOPLE" "MEDICAL" \
                "MATERIAL" \
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT"
do
  for model in "canine_s" "canine_c" "bert" "bart"
  do
    echo $model
    python3.10 -m kiloword.correlations_over_layers --label_name=$list_elt --model=$model --chunk_size 8 --distance cosine
    python3.10 -m kiloword.correlations_over_layers --label_name=$list_elt  --model=$model --chunk_size 8 --distance l2
#      python3.10 -m kiloword.make_animation --label_name=$list_elt --distance cosine --corr pearson --model=$model
#      python3.10 -m kiloword.make_animation --label_name=$list_elt --distance cosine --corr spearman --model=$model
#      python3.10 -m kiloword.make_animation --label_name=$list_elt --distance l2 --corr pearson --model=$model
#      python3.10 -m kiloword.make_animation --label_name=$list_elt --distance l2 --corr spearman --model=$model
    echo ""
  done
done
