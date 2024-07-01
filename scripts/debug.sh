#! /bin/bash


#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_s_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

for list_elt in  "PEOPLE" #"OBJECT" "FEELING" "LOCATION"  "MONEY" "NATURE" \
#                "QUANTITY" "HOUSE" "MOVE"  \
#                 "FOOD" \
#                "MODE" "DEVICE"  "TIME" \
#                 "DEATH" "BODY" "PEOPLE" "MEDICAL" \
#                "MATERIAL"
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT"
do
  for word_dist in "canine_s_layer_1" "canine_s_layer_2" "canine_s_layer_3" "canine_s_layer_4" \
                  "canine_s_layer_5" "canine_s_layer_6" "canine_s_layer_7" "canine_s_layer_8" \
                  "canine_s_layer_9" "canine_s_layer_10" "canine_s_layer_11" "canine_s_layer_12" \
                  "canine_s_layer_13" "canine_s_layer_14" "canine_s_layer_15" "canine_s_layer_16" \
  #"canine_s" "canine_s_random" "canine_c" "canine_s_random"
  do
    echo $word_dist
    python3.10 -m kiloword.get_correlations --word_dist_repr=$word_dist  --use_model_cache --focus_label=$list_elt
    export tab=$word_dist\_$list_elt\_correlations.csv
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab --distance cosine
    echo ""
  done
done