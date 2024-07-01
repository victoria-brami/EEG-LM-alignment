#! /bin/bash

#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_c_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

for list_elt in "ABSTRACT"
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT"
do
  for word_dist in "bart" "canine_c_random" "canine_s_random" "bart_random" #"bert" "bert_layer_11"
  do
    echo $word_dist
    python3.10 -m kiloword.get_correlations --word_dist_repr=$word_dist  --use_model_cache
    #python3.10 -m kiloword.get_correlations --word_dist_repr=$word_dist  --use_model_cache --focus_label=$list_elt
  done
done


