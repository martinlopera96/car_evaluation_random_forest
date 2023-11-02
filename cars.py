import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier


path = r'C:\Users\marti\Desktop\MARTÍN\DATA SCIENCE\Platzi\ML_projects\decision_trees\cars\car_evaluation.csv'
df = pd.read_csv(path)

df.head(5)
df.describe()
df.info()
print(df.dtypes)
print(df.shape)

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

# All features and the label are object type. The label is 'class'

df['class'].value_counts()
df.isnull().sum()

X = df.drop(columns=['class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Here we transform our features as every one of them is object type

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons','lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X_train, y_train)

y_train_prediction_tree = tree.predict(X_train)
y_test_prediction_tree = tree.predict(X_test)

train_accuracy_tree = accuracy_score(y_train, y_train_prediction_tree)
test_accuracy_tree = accuracy_score(y_test, y_test_prediction_tree)

print('Train accuracy: ', train_accuracy_tree)
print('Test accuracy: ', test_accuracy_tree)

importances = tree.feature_importances_
columns = X.columns

sns.barplot(x=columns, y=importances, palette='bright', saturation=2.0, edgecolor='black', linewidth=2)
plt.title('Feature Importances')
plt.show()

rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(X_train, y_train)

y_train_prediction_rf = rf.predict(X_train)
y_test_prediction_rf = rf.predict(X_test)

train_accuracy_rf = accuracy_score(y_train, y_train_prediction_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_prediction_rf)

print('Train accuracy: ', train_accuracy_rf)
print('Test accuracy: ', test_accuracy_rf)

feature_scores = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_scores, y=feature_scores.index, palette='bright')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.show()

cm = confusion_matrix(y_test, y_test_prediction_rf)
print('Confusion Matrix\n\n', cm)

print(classification_report(y_test, y_test_prediction_rf))

n_estimators_list = [10, 20, 30, 40, 50]
loss_values = []

for n_estimators in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    rf.fit(X_train, y_train)

    y_test_probabilities = rf.predict_proba(X_test)
    loss = log_loss(y_test, y_test_probabilities)
    loss_values.append(loss)

plt.plot(n_estimators_list, loss_values, marker='o')
plt.xlabel(f'Number of trees')
plt.ylabel('Log Loss')
plt.title('Log Loss vs. Number of Trees')
plt.grid(True)
plt.show()
