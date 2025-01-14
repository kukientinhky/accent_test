

# Counterfactual Explanations for Neural Recommenders
This repository contains data and the implementation of the ACCENT framework for two neural recommenders: Neural Collaborative Filtering (NCF) and Relational Collaborative Filtering (RCF).
Details of ACCENT can be found here: https://dl.acm.org/doi/10.1145/3404835.3463005.

<p align='center'>
	<img src="https://github.com/hieptk/accent/raw/main/accent.png" width=800>
</p>

All results in the paper were produced using the code at tag v1.0.

There are 2 subdirectories ```NCF``` and ```RCF``` which correspond to two recommender models exprimented in the paper.
Each subdirectory contains a ```data``` folder with the necessary data for the model and a ```src``` folder with the source code of all explanation algorithms.
```NCF``` is further divided into 2 subfolders: ```influence``` contains all source code related to the implementation of influence functions,
and ```scripts``` contains the implementation of NCF.

All supplementary materials (data, intermmediate results, pretrained models, virtual machine) can be download [here](https://zenodo.org/record/5150837#.YQZI5FMzZz8).

## Environment
To use this code, the following software are required:
- Python 3.7
- tensorflow 2.2.0
- tensorflow-addons 0.11
- numpy 1.19
- pandas
- scikit-learn
- matplotlib

A virtual machine with all required software can be downloaded [here](https://zenodo.org/record/5150837/files/ubuntu2.ova?download=1). Username: ```accent```, password: ```accent```.

Alternatively, you can follow these steps to setup the environment from a fresh install of Ubuntu 20.04.
1. Download the source code [here](https://github.com/hieptk/accent/archive/refs/heads/main.zip) and unzip.
2. Inside the unzipped folder, run the following command to install all required packages.
```bash
./init
```

## Dataset
We use the popular MovieLens 100K dataset (https://grouplens.org/datasets/movielens/100k/), which contains 100K ratings on a 1 − 5 scale by 943 users on 1682 movies. To conform to the implicit feedback setting in RCF, we binarized ratings to a positive label if it is 3 or above, and a negative label otherwise. We removed all users with < 10 positive ratings or < 10 negative ratings so that the profiles are big and balanced enough for learning discriminative user models. This pruning results in 452 users, 1654 movies, and 61054 interactions in our dataset.

A zip file containing all data can be downloaded [here](https://zenodo.org/record/5150837/files/data.zip?download=1).

Alternatively, to preprocess data from the original MovieLens dataset, follow these steps:
1. Download and unzip the original dataset [here](https://files.grouplens.org/datasets/movielens/ml-100k.zip).
2. Copy file ```u.data``` to ```RCF/data```.
3. Run script to preprocess data
```bash
cd RCF/src
python3 generate_data.py
```
3. New data is written to ```RCF/data/train.csv``` , ```RCF/data/test.csv```, and ```movielens_train.tsv```. Now copy data for NCF to the right directory:
```bash
cp movielens_train.tsv ../../NCF/data
```

### NCF
For NCF, the data is in ```NCF/data/movielens_train.tsv```. 
Each row consists of 4 tab-separated columns, representing an interaction between a user and a movie. The columns are:
- User ID
- Item ID
- Rating (integer, from 1 to 5)
- Timestamp

### RCF
For RCF, the data is in ```RCF/data/train.csv```. 
Each row is a comma-separated triple of a user and two items, where the user liked an item and disliked the other.
- ```user```: User ID
- ```pos_item```: Positive Item
- ```neg_item```: Negative Item

For RCF, metadata (genres, directors, actors) of movies and item-item relations are also required. This data was taken from the original RCF paper (https://arxiv.org/pdf/1904.12796.pdf).

* ```RCF/data/auxiliary-mapping.txt```: this file contains metadata of movies. Each row represents a movie. Each row consists of 4 parts, separated by vertical bars (```|```):
	- Item ID
	- List of genre IDs
	- List of director IDs
	- List of actor IDs
* ```RCF/data/relational_data.csv```: this file contains 97209 item-item relations extracted from ```auxiliary-mapping.txt```. Each row represents a relation, with 5 comma-separated columns:
	* ```head```: the head item
	* ```type```: type of relation (1: same genre, 2: same director, 3: same actor)
	* ```value```: value of the relation (genre/director/actor)
	* ```tail_pos```: the item which has this relation with ```head``` (positive item)
	* ```tail_neg```: the item which does not have this relation with ```head``` (negative item).

## Training Models
### NCF
From the unzipped folder, run the following comands to start training an NCF model.
```bash
cd NCF/src
python3 train.py
```

### RCF
Similarly, an RCF model can be trained by running:
```bash
cd RCF/src
python3 train.py
```
The final model will be saved in a directory named ```pretrain-rcf```. A pretrained model can be downloaded [here](https://zenodo.org/record/5150837/files/pretrain-rcf.zip?download=1).

## Running Experiment
For each algorithm, run the following commands to run the experiment. The script will generate explanations, retrain models, and evaluate results.

### NCF
```bash
cd NCF/src
python3 experiment.py --algo ALGO
```

### RCF
```bash
cd RCF/src
python3 experiment.py --algo ALGO
```
where *ALGO* indicates the explanation algorithm: "attention", "pure_att", "fia", "pure_fia", "accent".
Results will be stored in CSV files: *{ALGO}_{k}.csv* with k = 5, 10, 20.

The percentage of true counterfactual sets and average size will be printed to the console.
A result file has 452 rows (excluding the header), one for each user. Each row has the following columns:
- ```user```: ID of user
- ```item```: ID of the recommendation
- ```topk```: the original top-k items
- ```counterfactual```: the explanation found by the algorithm
- ```predicted_scores```: the predicted scores of the original top-k items
- ```replacement```: the predicted replacement item
- ```actual_scores_0```, ..., ```actual_scores_4```: the actual scores of the top-k after 5 retrains
- ```actual_scores_avg```: the average scores of 5 retrains.

The precomputed result files are uploaded [here](https://zenodo.org/record/5150837/files/results.zip?download=1).

## Reproducing Table 1
To get the counterfactual percentage and average set size as presented in Table 1, run:
```bash
python3 compare.py --file {FILE}
```
where FILE is the result file of the algorithm that you want to evaluate.

## Significance Testing
To perform significance testing between any pair of algorithms, follow these steps:
1. Copy result files produced in the previous steps to the base directory.
2. Run the script:
```bash
python3 compare --file {FILE1} --file2 {FILE2}
```
where FILE1 and FILE2 are the result files of algorithms that you want to compare. For example:
```bash
python3 compare.py --file fia_5.csv --file2 accent_5.csv
```
The script will print the following information:
- The counterfactual percentage of each algorithm
- A contingency table comparing two algorithms
- The p-value of the McNemar's test comparing the counterfactual effect
- The average counterfactual explanation size of each algorithm
- The p-value of the t-test comparing these sizes.
