# Kiloword
This repository contains the code to generate the Correlations between words' computational representations and Brain signals.

## Usage
Processing and analysing kiloword dataset. The file <a id="./data/words_and_pos.csv">words_and_pos.csv</a> contains
the Part-Of-Speech, the semantic field and the tangibility for each word of the dataset. Please note that the annotation was done manually, and there can be some ambiguities concerning certain words.

## Installation and dependencies
To clone the repo:
```
    git clone https://github.com/victoria-brami/kiloword.git
```
Then installing the packages:
```
    pip install -r requirements.txt
```

## Computing correlations between words representations and brain signals

** [02/08/2024] FOR A MORE RECENT UPDATE OF THE REPO USAGE, CHECK the `docs/` FOLDER **

To compute the Pearson and Spearman correlations between the EEG signals and the pairwise bert distances:
```
    python3.10 -m kiloword.get_correlations --word_dist_repr bert --eeg_path <path_to_eeg_recordings> --timesteps 30 --labels_path <path_to_kiloword_labels> --save_folder <folder_name> --tab_name <csv_correlations_name>
```
To visualize the evolution of the correlations:
```
    python3.10 -m kiloword.correlations_visualisation --tab_name <csv_correlations_name>
```
Here is an example of the different plots generated


<table style="text-align:center;align-items:center;align-self:center;border: none;">
  <tr style="font-size: 12pt; margin-bottom: 4px;">
<td ><b>Pearson Correlation with Trained Bert word Representations</b></td>
<td><b>Pearson Correlation with Random Bert word Representations</b></td>
  </tr>
  <tr style="margin-bottom: 14px;">
    <td><img src = "./assets/topography/pearson_bert_LANGUAGE_correlations.png" width ="600"/></td>
    <td><img src = "./assets/topography/pearson_bert_random_LANGUAGE_correlations.png" width ="600"/></td>
  </tr>
   <tr style="font-size: 12pt; margin-bottom: 4px;">
    <td><b>Pearson Correlation with Levensthein-distance between words</b></td>
    <td><b>Pearson Correlation with Levensthein-distance between IPA words</b></td>
  </tr>
<tr>
    <td><img src = "./assets/topography/pearson_levenshtein_LANGUAGE_correlations.png" width ="600" /></td>
    <td><img src = "./assets/topography/pearson_levenshtein_LANGUAGE_correlations.png" width ="600" /></td>
</tr>
</table>


## References
<a id="https://doi.org/10.1177/0956797615603934">[1]</a> 
Stéphane Dufau, Jonathan Grainger, Katherine J. Midgley, and Phillip J. Holcomb. 
A thousand words are worth a picture: snapshots of printed-word processing in an event-related potential megastudy. 
Psychological Science, 26(12):1887–1897, 2015.

<a id="https://aclanthology.org/2022.acl-long.156">[2]</a> 
Alex Murphy, Bernd Bohnet, Ryan McDonald, and Uta Noppeney. 2022. 
Decoding Part-of-Speech from Human EEG Signals. 
In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2201–2210, Dublin, Ireland. Association for Computational Linguistics.