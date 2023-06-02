import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

mu = 0.15
error_arr = []
accuracy_arr = []

#trains the model with training data
def train(yes_train, no_train, yes_aurora, no_aurora, num_batch_size, num_epoch, weight):
  for x in range(num_epoch):
    for x in range(int(num_batch_size/2)):
      new_data = np.insert(np.array(yes_train[x][1:-2], dtype = float), 0, 1)
      activation = np.sum(np.multiply(new_data, weight))
      if x == 0:
        error_arr.append(activation-float(yes_train[x][0]))
      prediction = 1 if activation > 0 else -1
      if prediction != int(yes_train[x][0]):
        error = float(yes_train[x][0]) - activation
        W_t = weight
        W_t = np.multiply(0.01,(np.multiply(error, new_data))) # adjust weight w.r.t. mu 
        weight = np.add(weight, W_t)
      new_data = np.insert(np.array(no_train[x][1:-2], dtype = float), 0, 1)
      activation_no = np.sum(np.multiply(new_data, weight))
      if x == 0:
        error_arr.append(activation_no-float(no_train[x][0]))
      prediction = 1 if activation_no > 0 else -1
      if prediction != int(no_train[x][0]):
        error_no = float(no_train[x][0]) - activation_no
        W_t_no = weight
        W_t_no = np.multiply(0.01,(np.multiply(error_no, new_data))) # adjust weight w.r.t. mu
        weight = np.add(weight, W_t_no)
    validate(yes_aurora, no_aurora, weight)
    yes_train=shuffle(yes_train)
    no_train=shuffle(no_train)
  return weight

#tests the model on validation data
def validate(yes_aurora, no_aurora, weight):
  accuracy = 0.0
  size = len(yes_aurora)
  for x in range(size):
    new_data = np.insert(np.array(yes_aurora[x][1:-2], dtype = float), 0, 1)
    activation = np.sum(np.multiply(new_data, weight))
    prediction = 1 if activation > 0 else -1
    if prediction == int(yes_aurora[x][0]):
      accuracy += 1.0
    new_data = np.insert(np.array(no_aurora[x][1:-2], dtype = float), 0, 1)
    activation = np.sum(np.multiply(new_data, weight))
    prediction = 1 if activation > 0 else -1
    if prediction == int(no_aurora[x][0]):
      accuracy += 1.0
  
  accuracy_arr.append(accuracy / (size * 2))
  return accuracy / (size * 2)

weight=np.random.random_sample([769])
all_img = []
f = open('histogram.txt')
for line in f:
    all_img.append(line.strip().split(" "))
f.close()

yes_img = all_img[0:1000]
no_img = all_img[1000:2001]
unknown_img = all_img[2001:]

yes_img = shuffle(yes_img, random_state = 0)
no_img = shuffle(no_img, random_state = 0)

yes_train = yes_img[0:int(1000 * 0.8)]
yes_test= yes_img[int(1000 * 0.8): 1000]
no_train = no_img[0: int((2001 - 1000) * .8)]
no_test = no_img[int((2001 - 1000) * .8): 2001]

batch_size = 20
epochs = 100

weight = train(yes_train, no_train, yes_test, no_test, batch_size, epochs, weight)
print("A:  " + str(validate(yes_test, no_test, weight)))

def not_known(data, weight):
  for x in range(len(data)):
    new_data = np.insert(np.array(data[x][1:-2], dtype = float), 0, 1)
    activation = np.sum(np.multiply(new_data, weight))
    prediction = 1 if activation > 0 else -1
    print("Image: " + str(data[x][769:])  + " P: " + str(prediction))

not_known(unknown_img, weight)

plt.xlabel= "Epochs"
plt.ylabel= "Error"
plt.title="Epochs vs. Error and Accuracy"
plt.scatter([int(n/2) for n in range(len(error_arr))], error_arr)
plt.scatter([n for n in range(len(accuracy_arr))], accuracy_arr, c='r')
plt.show()