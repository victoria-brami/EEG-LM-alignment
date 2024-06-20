#! /bin/bash

#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_c_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

for list_elt in "FEELING" "SEPARATION" "POLITICS" "ENTERTAINMENT" "MONEY" "NATURE" \
                "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT" \
                "FEELING" "SCIENCE" "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" \
                "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" \
                "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" \
                "MATERIAL" "GOVERN"
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT"
do
  for word_dist in "bert_layer_11" "bert_layer_10" "bert_layer_9" "bert_layer_8" "bert_layer_7" "bert_layer_6"
  do
    echo $word_dist
    python3.10 -m kiloword.get_correlations --word_dist_repr=$word_dist  --use_model_cache --focus_label=$list_elt
  done
done


# python3.10 -m kiloword.get_correlations --word_dist_repr bert --focus_label OBJECT
#python3.10 -m kiloword.get_correlations --word_dist_repr bert_random  --focus_label OBJECT
#python3.10 -m kiloword.get_correlations --word_dist_repr levenshtein --focus_label OBJECT
#python3.10 -m kiloword.get_correlations --word_dist_repr levenshtein_ipa --focus_label OBJECT

# python3.10 -m kiloword.get_correlations --word_dist_repr bert --focus_label ABSTRACT
#python3.10 -m kiloword.get_correlations --word_dist_repr bert_random --focus_label ABSTRACT
#python3.10 -m kiloword.get_correlations --word_dist_repr levenshtein --focus_label ABSTRACT
#python3.10 -m kiloword.get_correlations --word_dist_repr levenshtein_ipa --focus_label ABSTRACT

echo "\n SCIENCE"
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s --focus_label SCIENCE --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s_random  --focus_label SCIENCE --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c --focus_label SCIENCE --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c_random  --focus_label SCIENCE --use_model_cache
#
#echo "\n INDUSTRY"
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s --focus_label INDUSTRY --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s_random  --focus_label INDUSTRY --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c --focus_label INDUSTRY --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c_random  --focus_label INDUSTRY --use_model_cache
#
## shellcheck disable=SC2028
#echo "\n PHILOSOPHY"
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s --focus_label PHILOSOPHY --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s_random  --focus_label PHILOSOPHY --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c --focus_label PHILOSOPHY --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c_random  --focus_label PHILOSOPHY --use_model_cache
#
## shellcheck disable=SC2028
#echo "\n POLITICS"
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s --focus_label POLITICS --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_s_random  --focus_label POLITICS --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c --focus_label POLITICS --use_model_cache
#python3.10 -m kiloword.get_correlations --word_dist_repr canine_c_random  --focus_label POLITICS --use_model_cache