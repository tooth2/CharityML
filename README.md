# CharityML
## Finding Donors for CharityML
After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. The goal of this project is to evaluate and optimize several different supervised learning algorithms on data collected for the U.S. census to help CharityML identify people most likely to donate to their cause. 

### Project Approach 
In this project several supervised learning algorithms available in sklearn are appried by following steps
- explore the data to learn how the census data is recorded
- apply a series of transformations and preprocessing techniques to manipulate the data into a workable format. 
- evaluate several supervised learners on the data, which is best suited for the solution. 
- optimize the model which was selected 
- explore the chosen model and its performance to see just how well it's performing when considering the data it's given.
- summarize to determine which algorithm provided the highest donation yield while also reducing the total number of letters being sent

### Data Features
The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
    - age: Age. continuous
    - workclass: Working Class, non-numerical(categorical variables) 
    - education: Level of Education, non-numerical(categorical variables) 
    - education-num: Number of educational years completed, continuous
    - marital-status: Marital status, non-numerical(categorical variables) 
    - occupation: Work Occupation, non-numerical(categorical variables) 
    - relationship: Relationship Status, non-numerical(categorical variables) 
    - race: Race, non-numerical(categorical variables) 
    - sex: Sex(Female, Male), non-numerical(categorical variables) 
    - capital-gain: Monetary Capital Gains, continuous
    - capital-loss: Monetary Capital Losses, continuous
    - hours-per-week: Average Hours Per Week Worked, continuous
    - native-country: Native Country, non-numerical(categorical variables) 

**Target Variable**
- `income`: Income Class (<=50K, >50K)
- 
### Dependencies
This project uses the following software and Python libraries:
- [Python 3.6](https://www.python.org/download/releases/3.6/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Jupyter Notebook](http://ipython.org/notebook.html)

### Run
In a terminal or command window, navigate to the project directory and run one of the following commands:
```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

### Project structure
- finding_donors.ipynb: main file 
- census.csv: The project dataset to be loaded in the above notebook.
- visuals.py: Python script to provide supplementary visualizations.

### Project Detail(finding_donors.ipynb)
- Data Exploration: 
    - Number of records: : 45,222
    - Number of individuals with income >$50,000: 11, 208
    - Number of individuals with income <=$50,000: 34, 014
    - Percentage of individuals with income > $50,000: 24.78%
- Data Preprocessing: one-hot encoding for the feature and income data.
    - As for non-numeric features (called categorical variables), by one-hot encoding, these categorical variables were converted. One-hot encoding creates a "dummy" variable for each possible category of each non-numeric feature. 
        - Use pandas.get_dummies() to perform one-hot encoding on the 'features_log_minmax_transform' data.
        - Convert the target label 'income_raw' to numerical entries.
        - Set records with "<=50K" to 0 and records with ">50K" to 1.
- Evaluating Model Performance
    - Naive Predictor Performance: both accuracy and F1 scores of the naive predictor are as follows:
    - Accuracy measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
    - Precision tells us what proportion of messages we classified as spam, actually were spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of
        - [True Positives/(True Positives + False Positives)]
    - Recall(sensitivity) tells us what proportion of messages that actually were spam were classified by us as spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of
        - [True Positives/(True Positives + False Negatives)]
- Applied Model Comparison
    - Decision Tree
        - understandable explanation over the prediction, No assumptions on distribution of data
        - Decision trees are better for categorical data and it deals colinearity better than SVM.
        - Chances for overfitting the model, Prone to outliers.
    - SVM
        - work well with high-dimensional data, can support both linear and non-linear solutions using kernel trick.
        - Does not work well with noisy-data and not efficient with large datasets, training a SVM will be longer to train than a RF
        - Because the problem has complex(14) features and is a binary classification
    - RandomForest
        - same strength with decision tree but can be less prone to overfitting than Decision tree, a more generalized solution
        - reduces overfitting problem in decision trees and also reduces the variance
        - could over-fit with noisy data sets, doesn't work well with sparse data
