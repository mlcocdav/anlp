# Lab 8 Instructions

Here are replicated the commands for each section of lab 8, as well as environment setup. 
All of these assume you are in the root of this directory (where the README is).

If any of the commands below are unclear, run `python lab8.py --help`

## Install Requirements
Before installing packages, it is recommended to configure a virtual environment using 
[conda](https://docs.conda.io/en/latest/miniconda.html) or [venv](https://docs.python.org/3/library/venv.html).

### Using conda
Create a conda environment and activate it
```
conda create -n [ENV_NAME] python=3.8
conda activate [ENV_NAME]
```

### Using venv
This command will create the environment in the current folder (you need to have Python 3 installed):
```
python -m venv [ENV_NAME] 
```
Activate the environment:
```
source [ENV_NAME]/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```

## Part 1
Display with all labels:
`python lab8.py --part 1`

Display with just one label:
`python lab8.py --part 1 --ents PERSON`

Display with a subset of labels:
`python lab8.py --part 1 --ents PERSON ORG DATE`

## Part 2

Show entities and standard tokenization:
`python lab8.py --part 2`

Show entities with subword tokenization:
`python lab8.py --part 2 --tokenization subword`

## Part 3
There are no choices in this section, yet :).

Extract entity embeddings and save them to a pkl:
`python lab8.py --part 3`

## Part 4
Train a logistic regression classifier, save it, and output performance:
`python lab8.py --part 4  --classifier_path models/model_average.pkl --corpus data/corpus_average.pkl`

"Train" the three simple baseline classifiers, and output their performance (this does not save it, because it is trivial):
`python lab8.py --part 4 --corpus data/corpus_average.pkl --baseline`


