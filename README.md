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

## Steps followed:-
###### Dataset Collection:-
- Reference:- Folder named '\data'
- Collected using the mouse.jar application.
- Continuous data was collected over a period of time by various users


###### Extracting Dataset:-
- Reference:- extractor.py
- In accordance to the 0th element of a row(i.e.-MM/MC/MR..), we have writtens corresponding codes to extract the feature using 1st to last element of the row.
- The basics of mouse movement: X-coordinate, Y-coordinate, Theta value etc are extracted using the written code


###### Naive- Bayes Classifier:-
- Reference:- main.py
-  

## Execution of code:-
- Open terminal or ide and run the main.py file
- Note: The text files containing the data and python files should be in same folder
