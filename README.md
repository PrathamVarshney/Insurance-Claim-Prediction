🚑 Insurance Claim Prediction
This project predicts whether an individual will file an insurance claim based on various features such as age, BMI, smoking status, and region. It includes exploratory data analysis (EDA), preprocessing, visualization, and classification using LazyPredict.

📁 Dataset
The dataset used is Insurance.csv, which should be placed in the root directory of the project. It contains the following features:
age: Age of the individual
sex: Gender
bmi: Body Mass Index
children: Number of children
smoker: Smoking status
region: Residential region
charges: Medical costs billed
insuranceclaim: Whether an insurance claim was filed (1) or not (0)

🔧 Requirements
Install the necessary libraries using:
pip install pandas numpy matplotlib seaborn scikit-learn lazypredict

📊 Exploratory Data Analysis (EDA)
The project performs comprehensive EDA including:
Histograms of numerical features
Boxplots to detect outliers
Countplots of categorical variables
Heatmap of feature correlations
Pie chart showing insurance claim distribution
Barplot showing average charges by sex

🧹 Preprocessing
The following preprocessing steps are applied:
Label encoding for categorical variables (sex, smoker, region)
Feature scaling (standardization) of numerical variables (age, bmi, children, charges)
Splitting dataset into training and testing sets (70-30 split)

🤖 Model Training
We use LazyPredict to quickly train and evaluate multiple classification models without manual setup.
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
This provides a comparison table with accuracy, F1 score, training time, etc.

📈 Output Example
LazyPredict outputs a ranked table of classifiers with performance metrics like:
Model	Accuracy	F1 Score	Time Taken
RandomForestClassifier	0.86	0.87	0.12s
LogisticRegression	0.82	0.82	0.03s
...	...	...	...

✅ To Run
python insurance_prediction.py

💡 Future Improvements
Hyperparameter tuning of top models
Use of advanced classifiers like XGBoost or LightGBM
Deployment via Flask or Streamlit
