# Preprocessing the datasets


## 1. Kiloword Dataset


## 2. UBIRA dataset

See <a id="./README_ubira.md">README_ubira</a> file if you want more information about this dataset.

In its raw version, Ubira Dataset requires to relabel all the sentences read across the 6 different sessions so that the common sentences share common IDs. All the functions related to Ubira pre-processing are implemented in `prepare_dataset_files.py`.
For simplicity, we first convert all the fif files to npy and csv.



1. Get all the sentences in every session and create a folder for each of them (Converting by the way from fif to npy and csv): 
    ``` 
    rootpath = /PATH/TO/UBIRA/FOLDER
    list_folders = ["session_0", "session_1", "session_2", 
                    "session_3", "session_4", "session_5"]
    
    for sess_folder in tqdm(list_folders, desc=f"Get all sentences from each session..."):
        # Create csv file containing all the sentences read within a single session
        get_sentences_from_session(rootpath, sess_folder)
        # Filter and keep sentences appearing only ONCE within a single session
        get_unique_and_single_sentences_from_session(rootpath, sess_folder)

    # Get the Unique sentences and Create common sentence id across the different sessions
    # (will be easier later if we want to get the averaged eeg signals over the sessions )
    label_all_sentences_from_sessions(rootpath, use_duplicates=True)

    # Create the labels and signal files for each of those sessions (compute for all the sentences at first,
    # then isolate the duplicate sentences )
    create_sentence_level_files(rootpath, process_duplicates=True)
    ```

[//]: # (<table style="">)

[//]: # (	<tr style="">)

[//]: # (		<td><b></b></td>)

[//]: # (		<td><b></b></td>)

[//]: # (	</tr>)

[//]: # (        <tr style="">)

[//]: # (                <td><b></b></td>)

[//]: # (                <td><b></b></td>)

[//]: # (        </tr>)

[//]: # (</table>)
