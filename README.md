# TitanicClassification
## Introduction
The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, certain groups of people had a higher likelihood of survival than others. This project is part of a Kaggle challenge, where the goal is to build a predictive model that can determine the probability of survival based on passenger data, such as name, age, gender, and socio-economic class.

## Tools and Libraries
To tackle this challenge, we will use various tools and libraries from the Python ecosystem, including:
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization and exploratory data analysis.
- **Scikit-learn**: For implementing machine learning models and performing feature engineering.
- **Jupyter Notebook**: For interactive coding and analysis.

## Summary of the Proposed Solutions
Our approach to solving this problem involves the following steps:
1. **Data Exploration & Preprocessing**: Cleaning the dataset, handling missing values, and performing exploratory data analysis to understand key patterns.
2. **Feature Engineering**: Creating new features and transforming existing ones to improve model performance.
3. **Model Selection**: Comparing different machine learning algorithms such as Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting.
4. **Hyperparameter Tuning**: Optimizing the selected model to achieve better predictive accuracy.
5. **Evaluation & Interpretation**: Assessing model performance using accuracy, precision, recall, and other relevant metrics.

Through this process, we aim to answer the question: *What sorts of people were more likely to survive the Titanic disaster?* by leveraging data-driven insights and predictive modeling techniques.


## Data Collection
The dataset used for this challenge comes from Kaggle's Titanic competition. It includes two main files:
- **train.csv**: Contains labeled data with survival outcomes.
- **test.csv**: Contains unlabeled data for final model evaluation.
- **gender_submission.csv**: A sample submission file for reference.

The data is loaded using pandas:
```python
import pandas as pd
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
```

## Exploratory Data Analysis (EDA)
EDA was performed to understand patterns and relationships in the dataset. Key steps included:
- Checking for missing values and handling them appropriately:
```python
print(train_data.isnull().sum())  # Identify missing values
```
- Visualizing survival rates based on different features (e.g., gender, class, family size):
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.show()
```
- Creating new features like `AgeGroup`, `FamilySize`, and `IsAlone` to enhance predictive power:
```python
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
```

## Choosing the Best ML Model
Various machine learning models were tested to determine the best approach:
- **Baseline models**: Examined survival probabilities based on simple heuristics (e.g., women were more likely to survive).
- **Random Forest Classifier**: Chosen as the primary model due to its robustness and ability to handle categorical features effectively.
- **SMOTE Oversampling**: Used to address class imbalance in the training dataset.
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```
- **Hyperparameter tuning**: Conducted using GridSearchCV to optimize model performance.
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [5, 10, 15, None],
    'n_estimators': [100, 200, 300],
    'class_weight': [None, 'balanced']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5)
grid_search.fit(X_train_balanced, y_train_balanced)
```

## Model Creation & Deployment
1. **Feature selection**: The final model used features like `Pclass`, `Sex`, and `FamilySize` with one-hot encoding applied where necessary.
```python
features = ["Pclass", "Sex", "FamilySize"]
X = pd.get_dummies(train_data[features])
```
2. **Model training**: A Random Forest Classifier with tuned hyperparameters was trained on balanced data.
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1)
model.fit(X_train_balanced, y_train_balanced)
```
3. **Evaluation**: The model achieved an accuracy of approximately 77.6% on the validation set.
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, model.predict(X_val))
print("Validation Accuracy:", accuracy)
```
4. **Submission**: Predictions were generated on the test dataset and saved as `submission.csv` for Kaggle submission.
```python
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
```

Through this workflow, we effectively built a model that predicts survival probabilities on the Titanic dataset, showcasing the power of machine learning in classification tasks.


## Conclusion
This project demonstrated how machine learning can be used to predict survival on the Titanic dataset. Through feature engineering, data balancing, and hyperparameter tuning, we achieved a reasonable accuracy of 77.6% on the validation set. The model successfully captured key survival factors, such as gender, class, and family size.

## Limitations & Next Steps
### Limitations:
Data Imbalance: Despite using SMOTE, some biases may still exist in the model.
Limited Features: The dataset lacks certain critical survival factors like passenger behavior or location on the ship.
Overfitting Risk: Further validation with different test splits could be useful to confirm generalizability.

### Next Steps:
Try Different Models: Experiment with Gradient Boosting or Neural Networks for potential improvements.
Feature Engineering: Introduce additional features, such as ticket prices or cabin locations, to refine predictions.
Hyperparameter Tuning: Further optimize parameters using techniques like Bayesian Optimization instead of Grid Search.
Ensemble Learning: Combine multiple models to potentially improve overall accuracy.

Through continued iterations, we aim to refine and enhance our predictive capabilities for the Titanic dataset.
