# DLCV Final Project ( Medical-Imaging )

# How to run your code?
TODO: 
1. Download our models:  
```
bash reproduce_download.sh
```

2. Implement to generate a predicted csv file:  
```
bash reproduce_pred.sh <test_path> <csv_path>   
```
Note that <test_path> is your path to input data. For example, ./Blood_data/test  
Note that <csv_path> is your path to predicted csv file. for example, ./pred.csv  

3. Implement to turn the csv format for kaggle:  
```
bash reproduce_to_kaggle.sh <csv_path> <csv_kaggle_path>  
```
Note that <csv_path> is the path output from 2.  
Note that <csv_kaggle_path> is your path to kaggle format csv file, for example, ./pred_kaggle.csv  

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh

### Evaluation
We will use F2-score to evaluate your model. Please refer to the introduction ppt for more details.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to `requirments.txt` for more details.

You can run the following command to install all the packages listed in `requirements.txt`:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.
