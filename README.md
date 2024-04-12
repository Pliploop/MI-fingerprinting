# MI-fingerprinting
Audio fingerprinting for assignment 2 of Music Informatics Coursework - Queen Mary University of London

This project reproduces and adapts implementations of contrastive learning for audio fingerprinting as explored in [1],[2].

[1] Chang, Sungkyun, et al. "Neural audio fingerprint for high-specific audio retrieval based on contrastive learning." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.

[2] Yu, Zhesong, et al. "Contrastive unsupervised learning for audio fingerprinting." arXiv preprint arXiv:2010.13540 (2020).

## environment setup

Two environments are available for this project, one minimal required to run only the inference of the model. To install this environment, run

    pip install -r requirements_minimal.txt

For further experiments such as training and logging, which require wandb and pyorch lightning, a full environment can be installed by running 

    pip install -r requirements_full.txt

I recommend these be installed in a blank virtual environment.

## Running the beat tracker.

All required files should be in the repo to run the audio fingerprinting algorithm, including config and model checkpoints. the audio fingerprinter can be instanciated and run with the following commands:

    from fingerprinting.fingerprinting import fingerPrintBuilder, audioIdentification
    builder = fingerPrintBuilder(path_to_database,path_to_fingerprints)
    matcher = audioIdentification(path_to_fingerprints, path_to_queryset, path_to_output)

by default, these methods load a default checkpoint provided in the file. it is possible to change the encoder and checkpoint by specifying the ```model``` and ```checkpoint``` parameters at instanciation (these should be identical for both classes).

In order, these commands run the fingerprinting algorithm on the database and save the fingerprints to the provided path. Then, the matcher builds the fingerprints for the query_set and matches against the precomputed fingerprints. 

further arguments are available for each class, and are all documented within ```fingerprinting/fingerprinting.py```. 
