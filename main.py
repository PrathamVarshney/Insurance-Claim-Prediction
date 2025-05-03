import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lazypredict.Supervised import LazyClassifier
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

data = pd.read_csv("Insurance.csv")

print(data.head())

print("\nDataset Info:\n")
print(data.info())

print("\nStatistical Description:\n")
print(data.describe())
print(data['insuranceclaim'].value_counts)

numerical_data = data.select_dtypes(include='number')
numerical_data.hist(figsize=(10, 8), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=numerical_data, palette='Set2')
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

categorical_data = data.select_dtypes(include='object')
for col in categorical_data.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x=col, palette="Set1")
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=30)
    plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

claim_counts = data['insuranceclaim'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(claim_counts, labels=claim_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title("Insurance Claim Distribution")
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='sex', y='charges', ci=None, palette='Set3')
plt.title("Average Charges by Sex")
plt.show()

le = LabelEncoder()
data['region'] = le.fit_transform(data['region'])
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])

scaler = StandardScaler()
scaled_cols = ['age', 'bmi', 'children', 'charges']
data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

X = data.drop('insuranceclaim', axis=1)
y = data['insuranceclaim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=101)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)