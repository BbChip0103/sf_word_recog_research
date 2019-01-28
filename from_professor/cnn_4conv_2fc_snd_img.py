# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 05:20:23 2018

@author: root
"""

import numpy as np
import tensorflow as tf
import os.path
import scipy.io as sio

train_data = np.load('/users/jhlee/data/img_snd/train_data.npz')
validation_data = np.load('/users/jhlee/data/img_snd/validation_data.npz')
test_data = np.load('/users/jhlee/data/img_snd/test_data.npz')

img_dim_orig = (99, 257)
img_dim_crop = (99, 257)

#X_train = test_data['X_test'].reshape(-1, img_dim_orig[0], img_dim_orig[1])[:,:img_dim_crop[0],:img_dim_crop[1]].reshape(-1,np.prod(img_dim_crop))
#y_train = test_data['y_test']
#X_valid = validation_data['X_validation'].reshape(-1, img_dim_orig[0], img_dim_orig[1])[:,:img_dim_crop[0],:img_dim_crop[1]].reshape(-1,np.prod(img_dim_crop))
#y_valid = validation_data['y_validation']

X_train = train_data['X_train']
y_train = train_data['y_train']
X_valid = validation_data['X_validation']
y_valid = validation_data['y_validation']
X_test = test_data['X_test']
y_test = test_data['y_test']


#X_tr = X_train.reshape(-1, img_dim[0], img_dim[1])

def batch_generator(X, y, batch_size=64, 
                    shuffle=False, random_seed=None):
    
    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])
        
        
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = (X_valid - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_valid, X_test

        
## wrapper functions 

def conv_layer(input_tensor, name,
               kernel_size, n_output_channels, 
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        ## get n_input_channels:
        ##   input tensor shape: 
        ##   [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1] 


        weights_shape = (list(kernel_size) + 
                         [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor, 
                            filter=weights,
                            strides=strides, 
                            padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases, 
                              name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv, name='activation')
        print(conv)
        
        return conv
    

def fc_layer(input_tensor, name, 
             n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, 
                                      shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_units]))
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases,
                              name='net_pre-activation')
        print(layer)
        if activation_fn is None:
            return layer
        
        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer


def build_cnn():
    ## Placeholders for X and y:
    tf_x = tf.placeholder(tf.float32, shape=[None, np.prod(img_dim_crop)],
                          name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None],
                          name='tf_y')

    # reshape x to a 4D tensor: 
    # [batchsize, width, height, 1]
    tf_x_image = tf.reshape(tf_x, shape=[-1, img_dim_crop[0], img_dim_crop[1], 1],
                            name='tf_x_reshaped')
    ## One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=16,
                             dtype=tf.float32,
                             name='tf_y_onehot')

    ## 1st layer: Conv_1
    print('\nBuilding 1st layer: ')
    conv1 = conv_layer(tf_x_image, name='conv_1',
                    kernel_size=(5, 5), 
                    padding_mode='VALID',
                    n_output_channels=8)
    ## MaxPooling
    conv1_pool = tf.nn.max_pool(conv1, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')
    ## 2n layer: Conv_2
    print('\nBuilding 2nd layer: ')
    conv2 = conv_layer(conv1_pool, name='conv_2', 
                    kernel_size=(5,5), 
                    padding_mode='VALID',
                    n_output_channels=16)
    ## MaxPooling 
    conv2_pool = tf.nn.max_pool(conv2, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')

    ## 3rd layer: Conv_3
    print('\nBuilding 3rd layer: ')
    conv3 = conv_layer(conv2_pool, name='conv_3', 
                    kernel_size=(5,5), 
                    padding_mode='VALID',
                    n_output_channels=32)
    ## MaxPooling 
    conv3_pool = tf.nn.max_pool(conv3, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')

    ## 4th layer: Conv_4
    print('\nBuilding 4th layer: ')
    conv4 = conv_layer(conv3_pool, name='conv_4', 
                    kernel_size=(5,5), 
                    padding_mode='VALID',
                    n_output_channels=64)
    ## MaxPooling 
    conv4_pool = tf.nn.max_pool(conv4, 
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], 
                             padding='SAME')

    ## 1st FC layer: Fully Connected
    print('\nBuilding 1st FC layer:')
    fc1 = fc_layer(conv4_pool, name='fc_1',
                  n_output_units=1024, 
                  activation_fn=tf.nn.relu)

    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob, 
                            name='fc1_dropout_layer')

    ## 1st FC layer: Fully Connected
    print('\nBuilding 1st FC layer:')
    fc2 = fc_layer(fc1_drop, name='fc_2',
                  n_output_units=512, 
                  activation_fn=tf.nn.relu)

    ## Dropout
#    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob, 
                            name='fc2_dropout_layer')

    ## 4th layer: Fully Connected (linear activation)
    print('\nBuilding 4th layer:')
    output_layer = fc_layer(fc2_drop, name='output_layer',
                  n_output_units=16, 
                  activation_fn=None)

    ## Prediction
    predictions = {
        'probabilities' : tf.nn.softmax(output_layer, name='probabilities'),
        'labels' : tf.cast(tf.argmax(output_layer, axis=1), tf.int32,
                           name='labels')
    }
    
    ## Visualize the graph with TensorBoard:

    ## Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=output_layer, labels=tf_y_onehot),
        name='cross_entropy_loss')

    ## Optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss,
                                   name='train_op')

    ## Computing the prediction accuracy
    correct_predictions = tf.equal(
        predictions['labels'], 
        tf_y, name='correct_preds')

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32),
        name='accuracy')

    
def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in {} at epoch# {}'.format(path, epoch))
    saver.save(sess, os.path.join(path,'cnn-model.ckpt'),
               global_step=epoch)

    
def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
            path, 'cnn-model.ckpt-%d' % epoch))

    
def train(sess, training_set, validation_set=None, test_set=None, 
          initialize=True, epochs=20, shuffle=True,
          dropout=0.5, random_seed=None, path='./model'):

    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []
    train_acc, valid_acc, test_acc = [], [], []

    ## initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed) # for shuflling in batch_generator
    for epoch in range(1, epochs+1):
        batch_gen = batch_generator(
                        X_data, y_data, batch_size=64, 
                        shuffle=shuffle)
        avg_loss = 0.0
        avg_acc = 0.0
        for i,(batch_x,batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x, 
                    'tf_y:0': batch_y, 
                    'fc_keep_prob:0': dropout}
            loss, _ = sess.run(
                    ['cross_entropy_loss:0', 'train_op'],
                    feed_dict=feed)
            avg_loss += loss
            avg_acc += sess.run('accuracy:0', feed_dict=feed)
            
        train_acc.append(avg_acc / (i+1))
        training_loss.append(avg_loss / (i+1))
        print('Epoch %02d Training Avg. Loss: %7.3f, Train Acc: %7.3f,' % (
            epoch, avg_loss, train_acc[-1]), end=' ')
          
            
        if validation_set is not None:
            batch_gen_valid = batch_generator(validation_set[0], validation_set[1], \
                        batch_size=64, shuffle=shuffle)
            avg_acc =0.0
            for j, (batch_valid_x, batch_valid_y) in enumerate(batch_gen_valid):

                feed = {'tf_x:0': batch_valid_x, 'tf_y:0': batch_valid_y, \
                        'fc_keep_prob:0':1.0}
                avg_acc += sess.run('accuracy:0', feed_dict=feed)

            valid_acc.append(avg_acc / (j+1))
            print(' Validation Acc: %7.3f,' % valid_acc[-1])


        if test_set is not None:
            batch_gen_test = batch_generator(test_set[0], test_set[1], \
                        batch_size=64, shuffle=shuffle)
            avg_acc = 0.0
            for j, (batch_test_x, batch_test_y) in enumerate(batch_gen_test):

                feed = {'tf_x:0': batch_test_x, 'tf_y:0': batch_test_y, \
                        'fc_keep_prob:0':1.0}
                avg_acc += sess.run('accuracy:0', feed_dict=feed)

            test_acc.append(avg_acc / (j+1))
            print(' Test Acc: %7.3f' % test_acc[-1])
                
#            feed = {'tf_x:0': validation_set[0],
#                    'tf_y:0': validation_set[1],
#                    'fc_keep_prob:0':1.0}
#            valid_acc = sess.run('accuracy:0', feed_dict=feed)
#            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()

        if epoch % 10 == 0:
            save(saver, sess, epoch=epoch, path=path)
            sio.savemat(path+"/accuracy.mat", mdict={'training_loss': training_loss, \
            'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc})

    save(saver, sess, epoch=epoch, path=path)
    sio.savemat(path+"/accuracy.mat", mdict={'training_loss': training_loss, \
    'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc})
           
            
def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test, 
            'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
        
        
        
## Define hyperparameters
learning_rate = 1e-4
random_seed = 123

np.random.seed(random_seed)


## create a graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

    ## saver:
    saver = tf.train.Saver()


## crearte a TF session 
## and train the CNN model
##os.environ["CUDA_VISIBLE_DEVICES"]="7"

#config = tf.ConfigProto(device_count = {'GPU': 7}, log_device_placement=True)

#with tf.Session(graph=g, config=config) as sess:
with tf.Session(graph=g) as sess:
    
    max_epoch = 100  
    result_dir = './model/cnn_4conv_2fc'
    
    train(sess, 
          training_set=(X_train_centered, y_train), 
          validation_set=(X_valid_centered, y_valid), 
            test_set=(X_test_centered, y_test), 
          initialize=True,
          random_seed=123, epochs=max_epoch, path=result_dir)
#    save(saver, sess, epoch=max_epoch, path=result_dir)




'''
## create a default Graph
g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

    ## saver:
    saver = tf.train.Saver()

## create a new session 
## and restore the model
with tf.Session(graph=g2) as sess:
    load(saver, sess, 
         epoch=3, path='./model/cnn_5conv_2fc/')
    
    preds = predict(sess, X_test_centered, 
                    return_proba=False)

    print('Test Accuracy: %.3f%%' % (100*
                np.sum(preds == y_test)/len(y_test)))
 '''             


















