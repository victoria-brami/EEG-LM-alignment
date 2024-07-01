#! /bin/bash

# export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_c_random" ]
export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

for list_elt in "ALL"
do
  for word_dist in "bart" "canine_c" "canine_s" "canine_c_random"
  do
    echo $word_dist
    export tab=$word_dist\_$list_elt\_correlations.csv
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab --distance="cosine"
  done
done