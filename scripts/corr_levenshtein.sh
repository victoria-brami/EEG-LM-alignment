#! /bin/bash

#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_c_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

for list_elt in "OBJECT" "FEELING" "LOCATION"  "MONEY" "NATURE" \
                "QUANTITY" "HOUSE" "MOVE"  \
                 "FOOD" \
                "MODE" "DEVICE"  "TIME" \
                 "DEATH" "BODY" "PEOPLE" "MEDICAL" \
                "MATERIAL"
#"FEELING" "SEPARATION" "POLITICS" "ENTERTAINMENT" "MONEY" "NATURE" \
#                "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT" \
#                "FEELING" "SCIENCE" "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" \
#                "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" \
#                "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" \
#                "MATERIAL" "GOVERN"
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT"
do
  for word_dist in "levenshtein" "levenshtein_ipa"
  do
    echo $word_dist
    python3.10 -m kiloword.get_correlations --word_dist_repr=$word_dist  --use_model_cache --focus_label=$list_elt
    export tab=$word_dist\_$list_elt\_correlations.csv
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab --distance levenshtein-l2
    python3.10 -m kiloword.correlations_visualisation --tab_name=$tab --distance levenshtein-cosine
  done
done
