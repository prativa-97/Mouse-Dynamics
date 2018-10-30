# Mouse Dynamics For Imposter Detection
 
  Machine Intelligence and Expert Systems Term Project,

  Autumn Semester, 2018-19,

  Department of Electronics and Electrical Communication Engg,

  IIT Kharagpur


## Group No: 18
- Arka Sourav Karmakar         (15EC10007)
- Nikhil Singh                 (15EC10035)
- Aryendra Kumar               (15EE10008)
- Prativa Das                  (15IE10020)
- Debalina Chowwdhury          (15MF10005)

## Required Python Packages:
  1. `Numpy` 
  2. `Sklearn` 
  3. `pandas`
  4. `time`
  5. `math`

## Steps followed:
###### Dataset Collection:
- Reference:- Folder named **'\data'**
- Collected using the mouse.jar application.
- Continuous data was collected over a period of time by various users


###### Extracting Dataset:
- Reference:- **extractor.py**
- Pre-processes the raw data and transform it to contain hold time and latencies for all combination of keystrokes, 
- The basics of mouse movement: X-coordinate, Y-coordinate, Theta value etc are extracted.


###### `Naive-Bayes Classifier`:-
- Reference:- **main.py**
- Data obtained from extractor.py is used as input.
- Data for each user is assigned a particular class value (0,1,2,..).
- train-test split is done separately for each class to ensure train and test set contain appropriate proportions of each class
- Whole data is then merged, while maintaining the train-test split.
- Gaussian Naive-Bayes Model is implemented on the split X_train, X_test, y_train, y_test
- Accuracy is calculated by taking the ratio of correctly labelled to total points
- To validate the stability of the model, **five fold cross validation** is used.

## Execution of code:-
- Open terminal or ide and run the **main.py** file
- Note: The text files containing the data and python files *should be in same folder*
