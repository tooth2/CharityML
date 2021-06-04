# CharityML
## Finding Donors for CharityML
After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. The goal of this project is to evaluate and optimize several different supervised learning algorithms to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent. 

### Dependencies
This project uses the following software and Python libraries:
- Python 3.7
- NumPy
- pandas
- scikit-learn (v0.17)
- Matplotlib
- Jupyter Notebook.

### Project structure
- finding_donors.ipynb: main file 
- census.csv: The project dataset to be loaded in the above notebook.
- visuals.py: Python script to provide supplementary visualizations.

### Project Detail(finding_donors.ipynb)
- Data Exploration: 
    - Number of records
    - Number of individuals with income >$50,000
    - Number of individuals with income <=$50,000
    - Percentage of individuals with income > $50,000
- Data Preprocessing: one-hot encoding for the feature and income data.
- Evaluating Model Performance
    - Naive Predictor Performance: both accuracy and F1 scores of the naive predictor are as follows:
        - accuracy
        - F1 score 
- Model Comparison
- 
