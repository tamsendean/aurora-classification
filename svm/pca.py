import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

file = "histogram.txt"
dataset = pd.read_csv(file)
# delete unnecessary column
dataset.pop(dataset.columns[768])
print(dataset.head())

# get non-target columns
X = dataset.drop(dataset.columns[0], 1)
y = dataset[dataset.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train with 10 components 
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# variance ratios for each component
variance = pca.explained_variance_ratio_
print(variance)

# using random forest classifier ?? for predictions
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('Accuracy: ' + accuracy_score(y_test, y_pred))

comps = pca.components_.shape[0]

for x in range(10):
  values = [(np.argsort(np.abs(pca.components_[i])))[767-x] for i in range(comps)]

  features = values
  print(str(features))

  feature_names = [x for x in range(0, 767)]
  
  # get names
  important_names = [feature_names[features[i]] for i in range(comps)]
  list = {'{}'.format(i): important_names[i] for i in range(comps)}

  df = pd.DataFrame(list.items())
  print(df)