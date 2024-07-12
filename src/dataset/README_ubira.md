

Number of Unique sentences : 4 479 (complete for at least 3 sessions)

Dataset class:
load a session
in Getitem:
- eeg_input_ids
- Pos
- prev_pos
- next_pos
- word
- sentence id
- sentence content


Plot the dataset label cintents on unique PoS
    
#   EEG Dataset for RSVP-reading a subset of English-Web-Treebank corpus
#### Alex Murphy <AXM1390@student.bham.ac.uk>


## Overall Description:
These files contain preprocessed trial-level EEG data as outlined in the
accompanying paper: "Decoding Part-of-Speech from Human EEG Signals".
These data files haven filtered between [1 40] Hz and downsampled to 250 Hz (from 1 kHz).
Baseline correction has also been applied. The baseline period is 200 ms prior to sentence start.

## Data Acquisition
This data relates to 5.5 iterations of fixed-interval, single-word presentations of a subset of the English Web Treebank Corpus (Bies et al., 2012), released under a Creative Commons Share-Alike license (CC BY-SA 4.0). The EEG data were recorded using 64 Ag/AgCl active actiCAP slim  electrodes arranged in a 10–20 layout (ActiCAP, Brain Products GmbH, Gilching, Germany).


## Dataset Description: 
75 recording sessions of data are present in this data set.
Each recording session has been preprocessed such that all word-level trials presented
to the subject are contained within a window of 1,100 ms centred at the onset of word presentation
on screen, with the prior 200 ms and following 900 ms included in each window of data.
Each word was presented approximately every 240 ms.
Each word-level trial contains the following metadata:

* filename: 	which textfile from the EWT corpus the word derives from
* len:		the length (# characters) of the word
* freq:		Zipf-frequency value from the WordFreq Python package
* sess_id:	session identifier (same as filename, but useful when all data is concatenated)
* prev_pos:	part-of-speech of the previous word in the sentence***
* next_pos:	part-of-speech of the following word in the sentence
* prev_freq:	frequency score of the previous word
* next_freq:	frequency score of the following word
* prev_len: 	length of the previous word (# characters)
* next_len:	length of the following owrd (# characters)
* sent_ident:	a corpus-wise unique sentence identifier

In addition to the Part-of-speech tags are from the Universal Tagset, this dataset recognises two further useful tags in the data:
* "SENT_START"
* "SENT_END" 

These serve to fill prev_pos and next_pos slots for words that occur at the start / end of a sentence, respectively.

## File format:
This dataset employs the use of the .FIF file format for neuroscience data (See: https://mne.tools/0.15/manual/io.html)
The dataset was created using the MNE Python library (Gramfort et al.2013) (https://mne.tools/0.15/index.html).
Installation instructions: https://mne.tools/0.15/install_mne_python.html
Quick start: "pip install mne"

## Loading data:
Using the MNE library, each file can be loaded using the "mne.read_epochs(filename)" command.
The result is saved into an mne.Epochs data structure.
This data structure contains the the preprocessed EEG data in its "._data" attribute
which should be accessed via the ".get_data" function of the mne.Epochs data structure.
Associated metadata (pandas DataFrame) can be accessed via the ".metadata" attribute.

   Example:
   epochs = mne.read_epochs("rsvp_session0_files74-88-epo.fif")
   eeg_data = epochs.get_data() # NumPy array of shape [n_trials, n_electrodes, n_timepoints] with n_electrodes = 64, n_timepoints = 276 and n_trials is variable by each file.
   metadata = epochs.metadata   # Pandas DataFrame

All other information relating to the EEG recording and analysis pipeline is available
according to the extensive documentation on the MNE Python website: https://mne.tools/0.15/documentation.html

#### Example data:
This directory includes the full EEG data and metadata record for one of the sessions. This data can be easily loaded with NumPy and any CSV-reader in order to explore the format of the data. All other sessions follow an identical format. This allows evaluation of the type of data without the need to install MNE-Python and is representative of the entire dataset.

## Citation:
TBD


## License
This dataset is released under the Creative Commons Attribution-ShareAlike 4.0 license (CC BY-SA 4.0).

Further information on this license can be found at https://creativecommons.org/licenses/by-sa/4.0.

## References:

* Ann Bies, Justin Mott, Colin Warner, and Seth Kulick. 2012. English web treebank. Linguistic Data Consortium. (Accessible via: https://universaldependencies.org/treebanks/en_ewt/index.html)
* Alexandre Gramfort, Martin Luessi, Eric Larson, Denis Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti Hamalainen. 2013. "Meg and eeg data analysis with mne-python". Frontiers in Neuroscience, 7:267.

## File List:
This dataset comprises the following files:
* example_eeg_data.npy - Sample data from one EEG recording session
* example_metadata.csv - Sample metadata associated with the above data
* README.txt - A .txt version of this README file
* README.html - HTML version of README file
* LICENSE.txt - Description of the licensing agreement the dataset is released with
* Session0-files0-151-epo.zip
    * rsvp_session0_files0-11-epo.fif - FIF-format of a single EEG recording session
    * rsvp_session0_files105-112-epo.fif - See above (applies to all files below)
    * rsvp_session0_files113-125-epo.fif
    * rsvp_session0_files12-22-epo.fif
    * rsvp_session0_files126-134-epo.fif
    * rsvp_session0_files135-142-epo.fif
    * rsvp_session0_files143-151-epo.fif
    * rsvp_session0_files23-28-epo.fif
    * rsvp_session0_files29-39-epo.fif
    * rsvp_session0_files40-50-epo.fif
    * rsvp_session0_files51-73-epo.fif
    * rsvp_session0_files74-88-epo.fif
    * rsvp_session0_files89-94-epo.fif
    * rsvp_session0_files95-104-epo.fif
* Session1-files0-151-epo.zip
    * rsvp_session1_files0-9-epo.fif
    * rsvp_session1_files10-18-epo.fif
    * rsvp_session1_files106-122-epo.fif
    * rsvp_session1_files123-131-epo.fif
    * rsvp_session1_files132-140-epo.fif
    * rsvp_session1_files141-151-epo.fif
    * rsvp_session1_files19-33-epo.fif
    * rsvp_session1_files34-44-epo.fif
    * rsvp_session1_files45-54-epo.fif
    * rsvp_session1_files55-67-epo.fif
    * rsvp_session1_files68-71-epo.fif
    * rsvp_session1_files72-92-epo.fif
    * rsvp_session1_files93-105-epo.fif
* Session2-files0-151-epo.zip
    * rsvp_session2_files0-13-epo.fif
    * rsvp_session2_files105-113-epo.fif
    * rsvp_session2_files114-127-epo.fif
    * rsvp_session2_files128-139-epo.fif
    * rsvp_session2_files14-29-epo.fif
    * rsvp_session2_files140-151-epo.fif
    * rsvp_session2_files30-36-epo.fif
    * rsvp_session2_files37-42-epo.fif
    * rsvp_session2_files43-48-epo.fif
    * rsvp_session2_files67-75-epo.fif
    * rsvp_session2_files76-88-epo.fif
    * rsvp_session2_files89-97-epo.fif
    * rsvp_session2_files98-104-epo.fif
* Session3-files0-151-epo.zip
    * rsvp_session3_files0-7-epo.fif
    * rsvp_session3_files101-113-epo.fif
    * rsvp_session3_files114-119-epo.fif
    * rsvp_session3_files120-127-epo.fif
    * rsvp_session3_files128-139-epo.fif
    * rsvp_session3_files140-151-epo.fif
    * rsvp_session3_files25-29-epo.fif
    * rsvp_session3_files30-42-epo.fif
    * rsvp_session3_files52-66-epo.fif
    * rsvp_session3_files67-83-epo.fif
    * rsvp_session3_files8-24-epo.fif
    * rsvp_session3_files84-100-epo.fif
* Session4-files0-151-epo.zip
    * rsvp_session4_files0-12-epo.fif
    * rsvp_session4_files102-110-epo.fif
    * rsvp_session4_files111-123-epo.fif
    * rsvp_session4_files124-138-epo.fif
    * rsvp_session4_files13-20-epo.fif
    * rsvp_session4_files139-147-epo.fif
    * rsvp_session4_files148-151-epo.fif
    * rsvp_session4_files21-30-epo.fif
    * rsvp_session4_files31-44-epo.fif
    * rsvp_session4_files45-54-epo.fif
    * rsvp_session4_files55-71-epo.fif
    * rsvp_session4_files72-82-epo.fif
    * rsvp_session4_files83-89-epo.fif
    * rsvp_session4_files90-101-epo.fif
* Session5-files0-151-epo.zip
    * rsvp_session5_files0-8-epo.fif
    * rsvp_session5_files22-37-epo.fif
    * rsvp_session5_files38-50-epo.fif
    * rsvp_session5_files51-61-epo.fif
    * rsvp_session5_files62-68-epo.fif
    * rsvp_session5_files69-81-epo.fif
    * rsvp_session5_files82-94-epo.fif
    * rsvp_session5_files9-21-epo.fif