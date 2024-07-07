#! /bin/bash

# export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_c_random" ]
export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

for list_elt in  "MONEY" "NATURE" \
                "QUANTITY" "HOUSE" "MOVE"  \
                 "FOOD"  "PEOPLE" \
                "MODE" "DEVICE"  "TIME" \
                 "DEATH" "BODY" "PEOPLE" "MEDICAL" \
                "MATERIAL" \
#"FEELING" "OBJECT" "FEELING" "LOCATION" "SEPARATION" "POLITICS" "ENTERTAINMENT" "MONEY" "NATURE" \
#                "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT" \
#                "FEELING" "SCIENCE" "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" \
#                "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" \
#                "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" \
#                "MATERIAL" "GOVERN" #"FEELING" #"ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT" "FEELING" "INDUSTRY" "SCIENCE" "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" "MATERIAL" "GOVERN"
# "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" "MATERIAL" "GOVERN"
# "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" "MATERIAL" "GOVERN"
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT""OBJECT" "ABSTRACT" #
do
  for word_dist in "canine_s_layer_3" "canine_s_layer_4" "canine_s_layer_5" "canine_s_layer_6" "canine_s_layer_7"  #"canine_s" "canine_s_random" "canine_c" "canine_c_random"
  do
    echo $word_dist
    python3.10 -m kiloword.get_correlations --word_dist_repr=$word_dist  --use_model_cache --focus_label=$list_elt
    export tab=$word_dist\_$list_elt\_correlations.csv
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab --distance cosine
  done
done