#!/bin/bash

conda activate diffusion-lm

#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_s_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]
cd ../

python3 -m src.compute_correlations tab_name="bert_propn_correlations_nsent_200.csv" word_distance="bert"

for lay_id in 11 10 9 8 7 6 5 4 3 2 1
do
  echo "LAYER ${lay_id}"
  python3 -m src.compute_correlations tab_name="bert_adv_correlations_nsent_200.csv" word_distance="bert" model.layer=${lay_id}
  # python3 -m src.compute_correlations tab_name="bert_adv_correlations_nsent_200.csv" word_distance="bert" model.layer=10
  echo ""
done