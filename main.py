#importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
print(tf.__version__)



# make numpy printouts easier to read.
np.set_printoptions(precision = 3, suppress = True)



# raw dataset is downloaded from UCI repo and split using a comma delimiter
url2Repo = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url2Repo, names = column_names, na_values = '?', comment = '\t', sep = ' ', skipinitialspace = True)



# preprocessing layer on dataset
dataset = raw_dataset.copy()
dataset.tail()

# find unknown values
dataset.isna().sum()
# and remove them
dataset = dataset.dropna()

# convert numbers in country column to their corresponding country name
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# get dummies
dataset = pd.get_dummies(dataset, prefix = '', prefix_sep = '')
dataset.tail()

# divide dataset into training and testing then divide each into features and labels
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# plot dataset for inspection
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind = 'kde')

# data normalization
train_dataset.describe().transpose()[['mean', 'std']]
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))



# build and compile model
model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1)])

model.compile(loss = 'mean_absolute_error', optimizer = tf.keras.optimizers.Adam(0.001))

# summarise model
model.summary()

# train model
hist = model.fit(train_features, train_labels, validation_split = 0.2, verbose = 0, epochs = 100)

#plot loss
plt.plot(hist.history['loss'], label = 'loss')
plt.plot(hist.history['val_loss'], label = 'val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)

# collect test results
test_results['model'] = model.evaluate(test_features, test_labels, verbose=0)

# evaluate test performance
pd.DataFrame(test_results, index = ['Mean absolute error [MPG]']).T



# making predictions with the model
test_predictions = model.predict(test_features).flatten()

# plotting predictions vs real values
a = plt.axes(aspect = 'equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# error distribution of predictions
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
