import tensorflow as tf
#tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io,transform
import os
import random
from skimage.color import rgb2gray

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
            if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                         for f in os.listdir(label_directory)
                            if f.endswith(".ppm")]
        for f in file_names:
             images.append(data.imread(f))
             labels.append(int(d))
    return images, labels

def run_tensor_flow(images,labels):
    images28 = [transform.resize(image, (28,28)) for image in images]
    images28 = [np.reshape(image,2352) for image in images28]
    images28_orig = [transform.resize(image, (28,28)) for image in images]

    sample_indexes = random.sample(range(len(images28)),15) #the number shows number of images to show as evaluation
    sample_images = [images28[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

    x= tf.placeholder(dtype=tf.float32, shape=[None,2352])
    y= tf.placeholder(dtype=tf.int32,shape=[None])
    
    images_flat = tf.contrib.layers.flatten(x)
    
    W= tf.Variable(tf.random_normal([2352,62]))
    b= tf.Variable(tf.random_normal([62]))
    logits = tf.matmul(x,W)+b
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    correct_pred = tf.argmax(logits,1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
   

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(400):
        print('EPOCH', i)
        _, accuracy_val,pred = sess.run([train_step, accuracy,correct_pred],feed_dict={x: images28 , y: labels})
        if i % 10 == 0:
            print('Step {:5d}: training accuracy {:g}'.format(i,accuracy_val))

    fig = plt.figure(figsize=(20,15))
    for i in range(len(sample_indexes)):
        cor = sample_labels[i]
        plt.subplot(5,3,1+i)
        plt.axis('off')
        color = 'green' if cor == pred[i] else 'red'
        plt.text(40,10,"Correct: {0}\nTensor:{1}".format(cor,pred[i],fontsize = 5 ,color = 'red'))
        plt.imshow(images28_orig[i],cmap="nipy_spectral")
    plt.show()
    plt.savefig('sample_evaluation.png')

    sess.close()

ROOT_PATH="/data/examples/bel_tsc_tnsr"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
run_tensor_flow(images,labels)
