# Mouse-Dynamics for imposter detection
 
  Machine Intelligence and Expert Systems Term Project,

  Autumn Semester, 2018-19,

  Department of Electronics and Electrical Communication Engg,

  IIT Kharagpur



## Required python packages:-
  1.Numpy 

  2.Sklearn 

  3.pandas
  
 

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
- Pre-processes the raw data and transform it to contain hold time and latencies for all combination of keystrokes, 
- The basics of mouse movement: X-coordinate, Y-coordinate, Theta value etc are extracted.


###### Naive- Bayes Classifier:-
- Reference:- **main.py**
- Data obtained from extractor.py is used as input.
- Data for each user is assigned a particular class value (0,1,2,..).
- train-test split is done separately for each class to ensure train and test set contain appropriate proportions of each class
- Whole data is then merged, while maintaining the train-test split.
- Gaussian Naive-Bayes Model is implemented on the split X_train, X_test, y_train, y_test
- To validate the stability of the model, **five fold cross validation** is used.

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
           - GaussianNB with 5-fold Cross validation:- 67.39%
     
     
 ###### 3.When number of classes(users) is 9: 
            - GaussianNB:-  94.75419%
            - GaussianNB with 5-fold Cross validation:- 80.674%
  
## Conclusion:
We can witness a significant increase in the accuracy when the training dataset is increased.


