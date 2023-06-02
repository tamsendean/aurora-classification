import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data = 'histogram.txt'
X = np.genfromtxt(data, delimiter=' ', usecols=range(1, 767))
y = np.genfromtxt(data, delimiter=' ', usecols=[0], dtype=np.int64)

# split the data into a training set and a test set
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

clf = svm.SVC(kernel='linear')
clf.fit(trainX, trainY)
y_pred = clf.predict(testX)
accuracy = accuracy_score(testY, y_pred)

print('accuracy: ', accuracy)
print('y pred: ', y_pred)