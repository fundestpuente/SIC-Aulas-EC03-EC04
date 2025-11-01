import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

base_tree = DecisionTreeClassifier(random_state=SEED)
param_grid = {
    "max_depth": [None, 6, 8, 10, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}
grid = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
print("Mejores hiperparámetros:", grid.best_params_)

y_pred = best_tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
print(f"\nAccuracy (test): {acc:.4f}")
print(f"F1-macro (test): {f1m:.4f}\n")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción"); plt.ylabel("Real")
plt.title("Matriz de confusión - Árbol de Decisión")
plt.tight_layout()
plt.show()

