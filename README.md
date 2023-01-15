# IPEO-Understanding-the-Amazon-from-Space

This project is part of the ENV-540 Image Processing for Earth Observation course at EPFL and aims to predict multiple labels on satellite images of the Amazon river and its surroundings. This is done by training a convolutional neural network on a dataset of 40479 labeled images.

## Authors

Louis Poulain--Auzéau, louis.poulain-auzeau@epfl.ch

Basile Tornare, basile.tornare@epfl.ch

Octavio Profeta, octavio.profetamachon@epfl.ch
## Data
Download the IPEO_Planet_project folder don't already have it from here:  
https://drive.google.com/drive/folders/1tOMxGHMRtY8E1p1NKun6Wi_4DHMmRjAq?usp=sharing 

## Folder setup
The structure should be as follows:
 - Submission_folder  
   - IPEO-Understanding-the-Amazon-from-Space  
     - some code + logs + evaluation.ipynb
   - IPEO_Planet_project  
     - checkpoints  
     - train_labels.csv
     - train-jpg

## Running
### Create a local virtual environment in the venv folder

    python -m venv venv
### Activate the newly created environment

    source venv/bin/activate
### Install requirements

    pip install -r requirements.txt
### Here you need to download the file containing data (cf above for the link)

### cd in the good directory

    cd IPEO-Understanding-the-Amazon-from-Space
### Start a jupyter session

    jupyter notebook  
### Run evaluation.ipynb
### When you are done, close the venv

    deactivate
