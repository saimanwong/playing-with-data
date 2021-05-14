import random

import pandas as pd
import numpy as np

# visualization
# import seaborn as sns
# import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    combine = [train, test]

    interval = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    for ds in combine:
        ds['Title'] = ds.Name.str.extract(r' ([A-Za-z]+)\.')

        ds['Title'] = ds['Title'].replace('Sir', 'Mr')
        ds['Title'] = ds['Title'].replace('Major', 'Mr')
        ds['Title'] = ds['Title'].replace('Capt', 'Mr')
        ds['Title'] = ds['Title'].replace('Ms', 'Miss')
        ds['Title'] = ds['Title'].replace('Mme', 'Miss')
        ds['Title'] = ds['Title'].replace('Mlle', 'Miss')
        ds['Title'] = ds['Title'].replace('Lady', 'Miss')

        titles = {}
        for k, v in ds['Title'].items():
            if v not in titles:
                titles[v] = 0

            titles[v] += 1

        rare = []
        for k, v in titles.items():
            if v > 7:
                continue
            rare.append(k)

        ds['Title'] = ds['Title'].replace(rare, 'Rare')
        ds['Title'] = ds['Title'].map({
          'Mrs': 0,
          'Miss': 1,
          'Master': 2,
          'Rare': 3,
          'Mr': 4,

        }).astype(int)

        # Fill NA
        none_age = ds[ds['Age'].isna()]
        for i in none_age.index:
            embarked = none_age['Embarked'][i]
            title = none_age['Title'][i]

            mean_age, std_age = (
                ds[
                    (ds['Age'].notna()) &
                    (ds['Embarked'] == embarked) &
                    (ds['Title'] == title)
                ]['Age'].agg(['mean', 'std'])
            )

            min_age = mean_age-std_age
            if min_age < 0:
                min_age = 0
            random_age = random.randint(
                int(min_age),
                int(mean_age+std_age)
            )

            ds.at[i, 'Age'] = random_age

        # Age
        for i in ds['Age'].index:
            age = ds['Age'][i]
            if age >= 0 and age < 16:
                ds.at[i, 'Age'] = 0
                continue

            if age >= 16 and age < 32:
                ds.at[i, 'Age'] = 1
                continue

            if age >= 32 and age < 48:
                ds.at[i, 'Age'] = 2
                continue

            if age >= 48 and age < 64:
                ds.at[i, 'Age'] = 3
                continue

            if age >= 64 and age <= 80:
                ds.at[i, 'Age'] = 4
                continue

        ds['Age*Class'] = ds.Age * ds.Pclass
        # Sex
        ds['Sex'] = ds['Sex'].map({'male': 0, 'female': 1}).astype(int)

        # Custom ticket
        ds['TicketCustom'] = np.where(
            ds['Ticket'].str.contains(r'^\d+'),
            0,
            1,
        )

        # Fare
        for i in ds['Fare'].index:
            fare = ds['Fare'][i]
            for idx in range(len(interval)):
                ds.at[i, 'Fare'] = idx
                if interval[idx] >= fare:
                    break

        # Embarked
        idx = list(ds[ds['Embarked'].isna()].index)
        for i in idx:
            train.at[i, 'Embarked'] = 'S'

        ds['Embarked'] = ds['Embarked'].map(
          {'S': 0, 'C': 1, 'Q': 2}
        ).astype(int)

        # SibSp, Parch
        for i in ds.index:
            sibsp = ds['SibSp'][i]
            parch = ds['Parch'][i]
            company = sibsp + parch

            if company > 0:
                ds.at[i, 'IsAlone'] = 0
                continue

            ds.at[i, 'IsAlone'] = 1

    # Drop(?)
    columns_to_drop = ['SibSp', 'Parch', 'Cabin', 'Ticket', 'Name']
    train = train.drop(columns_to_drop, axis=1)
    test = test.drop(columns_to_drop, axis=1)

    print(train.head())

    print(train.shape, ', '.join(sorted(train.columns.values)))
    print(test.shape, ', '.join(sorted(test.columns.values)))

    # Predict time!
    algos = {
        'Logistic Regression': pd.DataFrame(),
        'Support Vector Machines': pd.DataFrame(),
        'Linear SVC': pd.DataFrame(),
        'k-nearest neighbors algorithm': pd.DataFrame(),
        'Gaussian Naive Bayes': pd.DataFrame(),
        'Perceptron': pd.DataFrame(),
        'Stochastic Gradient Descent': pd.DataFrame(),
        'Decision Tree': pd.DataFrame(),
        'Random Forest': pd.DataFrame(),
    }
    x = train.drop('Survived', axis=1)
    y = train['Survived']

    accs = pd.DataFrame(columns=['Algo', 'Accuracy'])
    for algo in algos:
        pred, acc = predict(algo, x, y, test)
        algos[algo] = pred
        accs = accs.append({'Algo': algo, 'Accuracy': acc}, ignore_index=True)

    print(accs[['Algo', 'Accuracy']].sort_values(
        by='Accuracy',
        ascending=False)
    )

    submission = pd.DataFrame({
      'PassengerId': test['PassengerId'],
      'Survived': algos['Random Forest'],
    })

    submission.to_csv('submission.csv', index=False)


def predict(algo: str, x: pd.DataFrame, y: pd.DataFrame, test: pd.DataFrame):
    algo_obj = None
    if algo == 'Logistic Regression':
        algo_obj = LogisticRegression(max_iter=1000)

    if algo == 'Support Vector Machines':
        algo_obj = SVC()

    if algo == 'Linear SVC':
        algo_obj = LinearSVC()

    if algo == 'k-nearest neighbors algorithm':
        algo_obj = KNeighborsClassifier(n_neighbors=3)

    if algo == 'Gaussian Naive Bayes':
        algo_obj = GaussianNB()

    if algo == 'Perceptron':
        algo_obj = Perceptron()

    if algo == 'Stochastic Gradient Descent':
        algo_obj = SGDClassifier()

    if algo == 'Decision Tree':
        algo_obj = DecisionTreeClassifier()

    if algo == 'Random Forest':
        algo_obj = RandomForestClassifier(n_estimators=100)

    if algo_obj is None:
        return ([], -1)

    algo_obj.fit(x, y)
    pred = algo_obj.predict(test)
    acc = round(algo_obj.score(x, y) * 100, 2)
    return (pred, acc)


if __name__ == "__main__":
    main()
