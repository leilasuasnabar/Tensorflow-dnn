#
# Author: Leila Suasnabar
# Date: November 29th, 2017
#
# Note: Some messages (that may look like errors) will show up after running
# the program, this is expected. This will take a few minutes to run.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Creat a folder for each trial
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error creating directory ' + directory)

if not os.path.isdir('./DNN_Files'):
    createFolder('./DNN_Files')
    if not os.path.isfile('./DNN_Files/counterTrial.txt'):
        np.savetxt('./DNN_Files/counterTrial.txt',[0])
        
numTrial = int(np.loadtxt('./DNN_Files/counterTrial.txt'))+1
directory = './DNN_Files/Trial%s'%str(numTrial)
createFolder(directory)
np.savetxt('./DNN_Files/counterTrial.txt',[round(numTrial)]) 

# Save parameters in text file
def createLog(directory, upperLimit, testSize, steps, predInfo, features, model):

    logFile = directory + '/log.txt'

    # Creat and write in file
    f = open(logFile, 'w')
    f.write('==== Deep Neural Network Regression model log==== \n\n')
    f.write('Trial: %s\n\n'%str(numTrial))

    f.write('Features used: ' + str(list(features)) )
    f.write('\nUpper limit on dataset: %d\n'%upperLimit)
    f.write('Number of steps: %d\n'%steps)
    testSize *= 100
    f.write('Training set size percentage: %.2f\n'%(100-testSize))
    f.write('   - Data points: %d\n'%predInfo[6])
    f.write('Testing set size percentage: %.2f\n'%testSize)
    f.write('   - Data points: %d\n\n'%predInfo[7])

    # Write prediction related outputss and model parameters
    params = model.get_params()['params']
    f.write('--Model Parameters--')
    f.write('\nHidden Neurons: \n' + str(params['hidden_units']))
    f.write('\nActivation Function: \n' + str(params['activation_fn']))
    f.write('\nOptimizer: \n' + str(params['optimizer']))
    f.write('\nLearning Rate: '+str(LEARNING_RATE))

    f.write('\n\nOnly relevant if using Momentum Optimizer:')
    f.write('\nMomentum: ' + str(MOMENTUM))
    f.write('\nNesterov: ' + str(MOMENTUM_MODE))

    f.write('\n\nOnly relevant if using adding dropout:')
    f.write('\nDropout values: ' + str(DROPOUT))
    
    f.write('\n\nFinal Loss on the testing set: %f\n'%predInfo[0])
    f.write('MSE Test Data: %f\n'%predInfo[1])
    f.write('Average percentage error on test set: %f\n'%predInfo[2])
    f.write('Accuracy based on predicted value being within a range of its true value\n')
    f.write('   Within 5 percent: %f <- To plot\n'%predInfo[3])
    f.write('   Within 10 percent: %f\n'%predInfo[4])
    f.write('   Within 15 percent: %f\n'%predInfo[5])

    # Write full model parameters
    f.write('\n\nFull Model Info: \n\n' + str(regressor.get_params(deep=False).items()))  
    f.close()


# Import data
# Change file if using the full set
data = pd.read_csv('Diamonds_dataset_full.csv',index_col=0)

# Add histogram of whole dataset, save image
if numTrial == 1:
    plt.grid()
    plt.hist(np.asarray(data['Price']),bins = 100)
    plt.title('Histogram of Full Dataset')
    plt.xlabel('Price of Diamond')
    plt.savefig('./DNN_Files/Hist_FullDataset.jpg')
    plt.close()
    #plt.show()

#-------------------------------------------------------------------------
#------------------------------To Modify----------------------------------
#-------------------------------------------------------------------------   
# Use only data with prices lower than a set priced
# Values can be changed
upperLimit = 200000 # Where to cut the tail, look at the whole histogram to
                    # see where is a good place to cut
                    # can try 50000,10000,20000,1000000
data = data[data.Price<=upperLimit]
testSize = 0.33 # Test split percentage from total new set
steps = 2000 
#-------------------------------------------------------------------------

# Add histogram of particular trial
plt.grid()
plt.hist(np.asarray(data['Price']),bins = 20)
plt.title('Histogram of modified data')
plt.xlabel('Price of Diamond')
plt.savefig(directory+'/Histogram%d.jpg'%numTrial)
plt.close()
#plt.show()

# List of features
COLUMNS = data.columns
FEATURES = COLUMNS.drop('Price')
PRICE = "Price"

# Splitting data into a train and test sets 
data_prices = data.Price
x_train, x_test, y_train, y_test = train_test_split(data[FEATURES] , data_prices, test_size=testSize, random_state=42)
y_train = pd.DataFrame(y_train, columns = [PRICE])
y_test = pd.DataFrame(y_test, columns = [PRICE])

# Modify price column, using its log10 result - useful since we're dealing
# with a large range of numbers
y_true = y_test.copy()
#y_test = np.sqrt(y_test)
#y_train = np.sqrt(y_train)
base = 10
y_test = np.log(y_test)/np.log(base)
y_train = np.log(y_train)/np.log(base)


# Reset indices
x_train.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
x_test.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

# Separate features - first numerical then categorical
train_numerical = x_train.select_dtypes(exclude=['object']).copy()
train_categoric = x_train.select_dtypes(include=['object']).copy()
train = train_numerical.merge(train_categoric, left_index = True, right_index = True) 

test_numerical = x_test.select_dtypes(exclude=['object']).copy()
test_categoric = x_test.select_dtypes(include=['object']).copy()
test = test_numerical.merge(test_categoric, left_index = True, right_index = True) 

# Features names
FEATURES_CAT = train_categoric.columns
FEATURES_NUM = train_numerical.columns

# Normalize
col_train_num = list(train_numerical.columns)
col_train_cat = list(train_categoric.columns)

mat_x_train_num = np.matrix(train_numerical)
mat_x_test_num = np.matrix(test_numerical)
mat_y_train = np.array(y_train)
mat_y_test = np.array(y_test)

norm_x_train_num = MinMaxScaler()
norm_x_train_num.fit(mat_x_train_num)

norm_y_train = MinMaxScaler()
norm_y_train.fit(mat_y_train)

x_train_num_scaled = pd.DataFrame(norm_x_train_num.transform(mat_x_train_num),columns = col_train_num)
y_train_num_scaled  = pd.DataFrame(norm_y_train.transform(mat_y_train),columns = [PRICE])

x_test_num_scaled = pd.DataFrame(norm_x_train_num.transform(mat_x_test_num),columns = col_train_num)
y_test_num_scaled  = pd.DataFrame(norm_y_train.transform(mat_y_test),columns = [PRICE])

# Setting up the numerical and categorical features
engineered_features = []

for continuous_feature in FEATURES_NUM:
    engineered_features.append(
        tf.contrib.layers.real_valued_column(continuous_feature))


for categorical_feature in FEATURES_CAT:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        categorical_feature, hash_bucket_size=1000)

    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,combiner="sum"))        

# Set up training and testing set containing all features
training_set = x_train_num_scaled.copy()
training_set[PRICE] = y_train_num_scaled.copy()
training_set[FEATURES_CAT] = train_categoric.copy()

testing_set = x_test_num_scaled.copy()
testing_set[PRICE] = y_test_num_scaled.copy()
testing_set[FEATURES_CAT] = test_categoric.copy()

def input_fn(data, training = True):
    continuous_cols = {k: tf.constant(data[k].values) for k in FEATURES_NUM}
    
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(data[k].size)], 
        values = data[k].values, 
        dense_shape = [data[k].size, 1]) 
        for k in FEATURES_CAT}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
    
    if training == True:
        # Converts the label column into a constant Tensor.
        label = tf.constant(data[PRICE].values)

        # Returns the feature columns and the label.
        return feature_cols, label
    
    # Returns the feature columns 
    return feature_cols

def leaky_relu(x):
    return tf.nn.relu(x) - 0.01 * tf.nn.relu(-x)

# Model
tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

#-------------------------------------------------------------------------
#------------------------------To Modify----------------------------------
#-------------------------------------------------------------------------   
# Parameters can be changed, change activation function, add regularizer, change number of hidden layers
# and number of neurons per layer

# Set Learning rate to a constant value
# Try at each then plot accuracies
LEARNING_RATE = 0.1 # 0.1,0.01,0.001,0.0001, etc

# If using MomemtumOptimizer
MOMENTUM = 0.95 # Change (0-0.9) range
MOMENTUM_MODE = True # set to True or False
# If using dropout
DROPOUT = 0.1 # Change (0-0.9) range

# Training the model using a deep nn regressor
regressor = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features,
                                        # These can be modified
                                        activation_fn = tf.nn.relu, # tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, leaky_relu
                                        hidden_units=[128, 256, 256], #Play with the values and layers quantities

                                        # Optimizers: GradientDescentOptimizer, AdadeltaOptimizer, AdagradOptimizer,RMSPropOptimizer
                                              # Optional: MomentumOptimizer(learning_rate= LEARNING_RATE, momentum = MOMENTUM, use_nesterov = MOMENTUM_MODE)
                                          #optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
                                        optimizer = tf.train.MomentumOptimizer(learning_rate = LEARNING_RATE, momentum = MOMENTUM, use_nesterov = MOMENTUM_MODE)
                                          , dropout = DROPOUT
                                          , model_dir = directory + '/model' # Run in trial folder: tensorboard --logdir ./model
                                          )
#-------------------------------------------------------------------------

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn = lambda: input_fn(training_set) , steps=steps)

# Evalute model
eval_model = regressor.evaluate(input_fn=lambda: input_fn(testing_set, training = True), steps=1)
loss = eval_model["loss"]
#print("Final Loss on the testing set: {0:f}".format(loss))

# Find predicted values
y = regressor.predict(input_fn=lambda: input_fn(testing_set))
y_pred = list(itertools.islice(y, testing_set.shape[0]))
y_pred = norm_y_train.inverse_transform(np.array(y_pred).reshape(len(y_pred),1))
y_pred = y_pred.reshape(y_pred.shape[0])
#y_true = np.asarray(y_test).reshape(y_pred.shape)
y_true = np.asarray(y_true).reshape(y_pred.shape)
y_pred = base**y_pred
#y_pred = y_pred**2

# Find MSE (probably won't be included in the slides/report)
MSEscore = metrics.mean_squared_error(y_pred,y_true)
#print('MSE Test Data: {0:f}'.format(MSEscore))

# Find average percentage error
error = np.mean(100*abs(y_true - y_pred)/y_true)
print('Average percentage error on Test Data: {0:f}'.format(error))

# Find accuracy, check percentage of predicted values that are within
# 5,10 and 15% range within the true values
acc1 = y_pred[(y_pred>y_true*0.95) & (y_pred<y_true*1.05)]
acc1 = 100*acc1.shape[0]/y_true.shape[0]
acc2 = y_pred[(y_pred>y_true*0.90) & (y_pred<y_true*1.10)]
acc2 = 100*acc2.shape[0]/y_true.shape[0]
acc3 = y_pred[(y_pred>y_true*0.85) & (y_pred<y_true*1.15)]
acc3 = 100*acc3.shape[0]/y_true.shape[0]

# Prediction information (loss, mse, error, accuracy) and other
predInfo = np.array([loss, MSEscore, error, acc1, acc2, acc3,y_train.shape[0],y_test.shape[0]])

# Plot predicted vs true values
fig, ax = plt.subplots(figsize=(50, 40))
plt.scatter(y_pred, y_true, s=100)
plt.xlabel('Predicted Values', fontsize = 50)
plt.ylabel('True Values', fontsize = 50)
plt.title('Predicted Vs True values on dataset Test', fontsize = 50)
# Find linear regression on predicted output
m,b = np.polyfit(y_pred,y_true,1)
ax.plot(y_pred,m*y_pred+b,'-')
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)

plt.savefig(directory+'/Prediction%d.jpg'%numTrial)
#plt.show()

# Save to log.txt
createLog(directory, upperLimit, testSize, steps, predInfo, FEATURES_NUM.append(FEATURES_CAT), regressor)
