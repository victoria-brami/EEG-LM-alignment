#! /bin/bash

cd ../
#export WORD_REPR=["canine_s", "canine_s_random", "canine_c", "canine_c_random" ]
#export LIST_LABELS=["ENTERTAINMENT","MONEY","NATURE","QUANTITY","RELIGION","HOUSE","MOVE","SPORT","JUSTICE","INDUSTRY","LANGUAGE","FOOD","MODE","DEVICE","FAMILY","MUSIC","CRIME","CATASTROPHE","ARMY","TIME","SCHOOL","CLEANNESS","DEATH","GLORY","BODY","PEOPLE","MEDICAL","MATERIAL","GOVERN"]

#for list_elt in "FEELING" "SEPARATION" "POLITICS" "ENTERTAINMENT" "MONEY" "NATURE" \
#                "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT" \
#                "FEELING" "SCIENCE" "JUSTICE" "INDUSTRY" "LANGUAGE" "FOOD" \
#                "MODE" "DEVICE" "FAMILY" "MUSIC" "CRIME" "CATASTROPHE" "ARMY" "TIME" \
#                "SCHOOL" "CLEANNESS" "DEATH" "GLORY" "BODY" "PEOPLE" "MEDICAL" \
#                "MATERIAL" "GOVERN"
# "ENTERTAINMENT" "MONEY" "NATURE" "QUANTITY" "RELIGION" "HOUSE" "MOVE" "SPORT" "OBJECT" "ABSTRACT"
#for list_elt in "MONEY" "MUSIC" "NATURE" "QUANTITY" "RELIGION" "DEATH" "HOUSE" "MOVE" "INDUSTRY" "TIME"
#do
#  for word_dist in "bert_layer_11" "bert_layer_10" "bert_layer_9" \
#                    "bert_layer_8" "bert_layer_7" "bert_layer_6" \
#                    "bert_layer_5" "bert_layer_4" "bert_layer_3" \
#                    "bert_layer_2" "bert_layer_1"

conda activate diffusion-lm
# "RELIGION"
export TIMESTEPS=21

for list_elt in  "QUANTITY" #"RELIGION" "INDUSTRY""ABSTRACT"
do
for word_dist in "bert" "bert_layer_11" "bert_layer_10" "bert_layer_9" \
                    "bert_layer_8" "bert_layer_7" "bert_layer_6" \
                    "bert_layer_5" "bert_layer_4" "bert_layer_3" \
                    "bert_layer_2" "bert_layer_1"

do
    echo $word_dist
    python3 -m src.get_correlations \
                      --word_dist_repr=$word_dist  \
                      --use_model_cache \
                      --focus_label=$list_elt \
                      --timesteps=$TIMESTEPS
#                      --save_folder=""
    #
    export tab=$word_dist\_$list_elt\_correlations.csv
    echo ""
#    echo "Reading from tab $tab "

done
    python3.10 -m src.correlations_over_layers  --distance="cosine" --chunk_size 6 --model="bert" --label_name=$list_elt --timesteps=$TIMESTEPS
    python3.10 -m src.correlations_over_layers  --chunk_size 6 --model="bert" --label_name=$list_elt --timesteps=$TIMESTEPS
    python3 -m src.make_animation data=kiloword save_folder="/home/viki/Downloads" distance="cosine" corr="pearson" timesteps=$TIMESTEPS label_name=$list_elt
    python3 -m src.make_animation data=kiloword save_folder="/home/viki/Downloads" distance="l2" corr="pearson" timesteps=$TIMESTEPS label_name=$list_elt
    python3 -m src.make_animation data=kiloword save_folder="/home/viki/Downloads" distance="cosine" corr="spearman" timesteps=$TIMESTEPS label_name=$list_elt
    python3 -m src.make_animation data=kiloword save_folder="/home/viki/Downloads" distance="l2" corr="spearman" timesteps=$TIMESTEPS label_name=$list_elt
done

