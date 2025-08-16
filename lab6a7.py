#importing necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Loading dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# 2 features for visualization
X = df.iloc[:, [0, 1]]  
Y = df['LABEL']

#splitting data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#training decision tree 
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, Y_train)

#plotting decision boundary 
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')


plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, cmap='coolwarm', edgecolor='k', label='Train')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=Y_test, cmap='coolwarm', marker='x', label='Test')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Tree Decision Boundary')
plt.legend()
plt.show()

