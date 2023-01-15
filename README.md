# IPEO-Understanding-the-Amazon-from-Space

## Data
download the chekpoints file, the train-jpg contains all the images provided for the project, download it if you don't already have it
https://drive.google.com/drive/folders/1SS4wRKzELXB4qXrmHg8T_wfbjYX1xdIf?usp=share_link
The structure should be as follows:
 - Submission_folder  
  - IPEO-Understanding-theAmazon-from-Space  
   - some code + logs + evaluation.ipynb
  - IPEO_Planet_project  
   - checkpoints  
   - train_labels.csv
   - train-jpg
## Settings

    # create a local virtual environment in the venv folder
    python -m venv venv
    # activate this environment
    source venv/bin/activate
    # install requirements
    pip install -r requirements.txt
    # Here you need to download the file containing data (cf above for the link)
    
    # cd in the good directory
    cd IPEO-Understanding-the-Amazon-from-Space
    # start a jupyter session
    jupyter notebook  

when you are done, close the venv  

    deactivate
