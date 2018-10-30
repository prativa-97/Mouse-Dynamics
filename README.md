# Mouse-Dynamics for imposter detection
 
  Machine Intelligence and Expert Systems Term Project,

  Autumn Semester, 2018-19,

  Department of Electronics and Electrical Communication Engg,

  IIT Kharagpur


## Group No:- 18
- Arka Sourav Karmakar         (15EC10007)
- Nikhil Singh                 (15EC10035)
- Aryendra Kumar               (15EE10008)
- Prativa Das                  (15IE10020)
- Debalina Chowwdhury          (15MF10005)

## Required python packages:-
  1.Numpy 

  2.Sklearn 

  3.pandas
  
  4.time
  
  5.math
  

## Execution of code:-
- Open terminal or ide and run the **main.py** file
- Note: The text files containing the data and python files *should be in same folder*

## Steps followed:-
###### Dataset Collection:-
- Reference:- Folder named **'\data'**
- Collected using the mouse.jar application.
- Continuous data was collected over a period of time by various users


###### Extracting Dataset:-
- Reference:- **extractor.py**
- In accordance to the 0th element of a row(i.e.-MM/MC/MR..), we have writtens corresponding codes to extract the feature using 1st to last element of the row.
- The basics of mouse movement: X-coordinate, Y-coordinate, Theta value etc are extracted using the written code


###### Naive- Bayes Classifier:-
- Reference:- **main.py**
- Every user is assigned a particular class (0 to n-1 if n users are there)
- Training and testing data for every user is defined after the complete data is extracted using extractor.py
- X-test, X-train are defined using user data
- y_train and y_test are assigned as the user class
- Data is pre-processed
- Gaussian Naive-Bayes Model is implemented 
- Accuracy is calculated by taking the ratio of correctly labelled to total points
- To validate, **five fold cross validation** is used.

## Execution of code:-
- Open terminal or ide and run the **main.py** file
- Note: The text files containing the data and python files *should be in same folder*

## Results:-
OBTAINED ACCURACY 
 ###### 1.When number of classes(users) is 5:-
           - GaussianNB:- 49.49833% 
           - GaussianNB with 5-fold Cross validation:- 15.798%
     
          
 ###### 2.When number of classes(users) is 8:
           - GaussianNB:-  83.49355%
           - GaussianNB wit 5-fold Cross validation:- 67.2%
     
     
 ###### 3.When number of classes(users) is 9: 
            - GaussianNB:-  94.75419%
            - GaussianNB wit 5-fold Cross validation:- 80.674%
  
## Conclusion:
We can witness a significant increase in the accuracy when the training dataset is increased.


