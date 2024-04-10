# deep-learning-challenge Module 21 Challenge

## Overview
This analysis was commissioned by the Alphabet Soup foundataion for the purpose of creating a classifier model which can correctly predict which funding applications received will have the greatest likelihood of effectively using the funds received, within a population of total funding applications.  The model assesses a dataset of 340000 applications submitted by organizations over the years, consisting of the following fields:  

- EIN and NAME — Identification columns (excluded from the model as they are neither targets nor features)
- APPLICATION_TYPE — Alphabet Soup application type (independent X variable)  
- AFFILIATION — Affiliated sector of industry  (independent X variable)  
- CLASSIFICATION — Government organization classification  (independent X variable)  
- USE_CASE — Use case for funding  (independent X variable)  
- ORGANIZATION — Organization type  (independent X variable)    
- STATUS — Active status  (independent X variable)  
- INCOME_AMT — Income classification  (independent X variable)  
- SPECIAL_CONSIDERATIONS — Special considerations for application  (independent X variable)  
- ASK_AMT — Funding amount requested  (independent X variable)  
- IS_SUCCESSFUL — Was the money used effectively (the target y variable)

## Approach
1. The fields were reviewed and classed as one of: excluded from the model, an independent X variable, or the y variable as indicated above.
2. A frequency distribution of each categorical X variable's values was reviewed and the number of bins was reduced to less than 10 bins per variable, with data from smaller bins consolidated into a single bin named 'other'.
3. The categorical data was converted to numeric data using the pandas get_dummies() method.
4. Both the independent X and dependent y datasets were split into two additional subsets one for training the model and one for test then model. 75% of the data was used for training and 25% was reserved for testing the final model.  
5. In order to remove any imbalance in feature influence in the model, the independent training data was used to create a scaling factor which was applied to both the independent training and independent testing data.
6. After scaling, 75% of the independent variable data was used to train the model, the remaining 25% was used to test the model.
7. 43 features were derived from the data requiring 43 input neurons.
8. A neural network model was created with 2 hidden layers with 30 and 15 neurons respectively with the expectation the 43 inputs could easily be weighted across these layers in a reducing manner.  The 'relu' activation function was selected due to data science convention and trained for 50 epochs. 
## Initial Results
-  The trained model achieved an accuracy of **.7383** in the 49th epoch with the training data.
-  When the resulting model was applied to the testing data, an accuracy of **.7290** was achieved with a loss of **.5508**.
-  As a result this model did not achieve the target model result of 75% accuracy.

## Optimization
The following steps were taken to optimize the model.
1. The get_dummies encoding was replaced with ordinal encoding for INCOME_AMT and SPECIAL_CONSIDERATIONS features to better reflect the ordinal nature of the data as a result the number of features input into the model was reduced from 43 to 34.
2. The number of bins used to classify the CLASSIFICATION and and 'APPLICATION_TYPE were increased to see if they improved the model these levels indicated the original however the results were not less accurate so the optimal model contained the same bin strategy as the base model.
3. The resulting data set was split into two groups for training (75%) and testing (25%) with the same random_state as the intial model to limit variation due to how the training and testing data is split.
4. The data was then run through a Keras Tuner with the following parameters.
    a. Tuner: hyperband
    b. Model: sequential
    c. Activation functions: a choice of one of 'relu', 'leaky_relu', or 'gelu'
    d. Number of units per layer: 1 to 35
    e. Number of hidden layers: 1 to 4 (3 total)
    f. Regularization:  Lasso or L1 regularizer used in an attempt to reduce the number of features used in the model.
    g. Output layer activation function: sigmoid
    h. Maximum Epochs: 30
    i. hyperband interations:2
    j. objective: maximize accuracy.
5. The top two models were reviewed and used to derive a selected **optimized** model with the following features:
    a. Hidden Layers: 3
    b. Layer Activation: relu
    c. Units per layer: 26,31,16,11
    d. Number of Epochs: 50 (to align with base model)
    e. Regularization: L1 

## Optization Results:
-  The optimize model acheived an accuracy of **.7315** in the 28th epoch with the training data.
-  When the resulting model was applied to the testing data, an accuracy of **.7313** was acheived with the loss of **.5711**
-  The results did not meet the objective target of .75 accuracy.

## Summary and Recommendations for Further Analysis:
Comparing the results from the base model vs the "optimal" model there does not appear to be a significant difference in model performance.  The optimal model however does perform better when processing the test data than the base model as evidenced by the testing accuracy exceeding the training accuracy of the model by .0002 (.7315 - .7313).  This is in comparsion to the base model in which the test results lagged behind the training accuracy by .0093 (.7383 - .7290).  This reduction in overfitting is a result of implementing the L1 regularization where as the base model had none.  

## Items to consider to further improve the model:
1. Reducing the number of bins of the categorical features which were adjusted.
2. Consider varying the activation functions per layer.
3. Modify the tuner to allow for varying activation functions per layer
4. Increasing the number of hidden layers assessed by the tuner.

## Note: all files are located in the Main Branch of this repository.
