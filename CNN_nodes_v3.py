import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import csv
import cv2
import os as os

# define shuffle function
def shuffle_a_list(list_in, permutation):
    
    shuffle_list = []
    for i in range(0, len(permutation)): 
        shuffle_list.append(list_in[permutation[i]])
    
    return shuffle_list

#create_placeholders
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name='inputs_placeholder')
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='labels_placeholder')
    
    return X, Y

def initialize_parameters():
    
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed=0) )
    
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed=0) )
    
    parameters = {"W1": W1, "W2": W2} #"W3": W3, "W4": W4}#, "W5": W5} #, "W6": W6} 
    return parameters

def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # CONV2D: stride of 1, padding 'SAME', 64 
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME', 128
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
   
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 5 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    FC1=4096
    Z6 = tf.contrib.layers.fully_connected(P2, FC1, activation_fn=None)
    #FC2=4096
    #Z7 = tf.contrib.layers.fully_connected(Z6, FC2, activation_fn=None)
    FC2=1000
    Z7 = tf.contrib.layers.fully_connected(Z6, FC2, activation_fn=None)
    num_outputs = 5
    Z = tf.contrib.layers.fully_connected(Z7, num_outputs, activation_fn=None)
    ### END CODE HERE ###
    print(Z)
    return Z

# Loss function
def compute_cost(Z, Y):
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y)
    cost = tf.reduce_mean(cost)
    print(cost)
    return cost

def model(learning_rate = 0.009,num_epochs = 100, minibatch_size = 64, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    n_H0 = 224
    n_W0 = 224
    n_C0 = 3
    n_y = 5                         
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    # Saver
    saver = tf.train.Saver()

    # PREPARE DATA
    MAX_IMAGES = 22
    Y_train = np.zeros((MAX_IMAGES,5), float)
    
    m = 0
    image_name_list = []
    
    with open('/data/vernica_data/Toy_labels.csv') as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name_list.append(row['image'])
            if (row['level'] == '0'):
                Y_train[m,:] = np.array([1, 0, 0, 0, 0])
            if (row['level'] == '1'):
                Y_train[m,:] = np.array([0, 1, 0, 0, 0])
            if (row['level'] == '2'):
                Y_train[m,:] = np.array([0, 0, 1, 0, 0])
            if (row['level'] == '3'):
                Y_train[m,:] = np.array([0, 0, 0, 1, 0])
            if (row['level'] == '4'):
                Y_train[m,:] = np.array([0, 0, 0, 0, 1])  
            m = m + 1
    print(m)
    print(Y_train)
    #Y_train = np.delete(Y_train, np.linspace(m, MAX_IMAGES-1, num = MAX_IMAGES-m), 0)
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        seed=3
        #print (m)
        for epoch in range(num_epochs):
        
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            
            # DEBUGGING
            #num_minibatches = 2

            seed = seed + 1
            
            # THIS PART IS SIMILAR TO random_mini_batches
            np.random.seed(seed)

            permutation = (np.random.permutation(m))
            shuffled_image_name_list = shuffle_a_list(image_name_list, permutation)
            shuffled_Y = Y_train[permutation,:]

            #minibatch_X = np.zeros((minibatch_size,512,512,3))
            minibatch_X = np.zeros((minibatch_size,224,224,3))

            for batch in range(num_minibatches):
                    
                minibatch_Y = shuffled_Y[ batch * minibatch_size : (batch+1)*minibatch_size ,:]
                lower_limit=batch * minibatch_size
                
                upper_limit=(batch+1)*minibatch_size-1
                
                for i in range(lower_limit,upper_limit):# I WANT i RUN FROM batch * batch_size TO (batch+1)*batch_size-1
                                
                    # HOW TO READ IMAGE FROM THE FOLDER CONTAINING ALL IMAGES
                    image_path= "/data/vernica_data/Toy_images/" + shuffled_image_name_list[i] + ".jpeg"
                    image = cv2.imread(image_path)
                    #print(image.size)
                    image = cv2.resize(image, (224,224))
                    #print(image.shape)
                    minibatch_X[i - batch * minibatch_size ,:,:,:] = image
                
                # runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).

                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:np.float32(minibatch_X),Y:np.float32(minibatch_Y)})
                #print(Z3.eval())
                
                minibatch_cost += temp_cost / num_minibatches        
    
            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                
            #if epoch % 1 == 0:
             #   checkpoint_file = os.path.join("/data/vernica_data/model_v3/", 'checkpoint')
              #  saver.save(sess, checkpoint_file, global_step = epoch)
                    
                
            # This is the end of all minibatches
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

            # code for validation accuracy
            
        # This is the end of epoch
        # Calculate accuracy of training set here

        predict_op = tf.argmax(Z, 1)
            #print(predict_op)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1), name='prediction')
            #print(correct_prediction)
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy') #"float"
        
        #if epoch % 1 == 0:
         #   checkpoint_file = os.path.join("/data/vernica_data/model_v4/",'checkpoint')            
         #   saver.save(sess, checkpoint_file, global_step = epoch)
        
        minibatch_size = 2
        num_minibatches = int(m/minibatch_size)
        train_accuracy = 0
        #minibatch_X = np.zeros((minibatch_size,512,512,3))
        minibatch_X = np.zeros((minibatch_size,224,224,3))
        for batch in range(num_minibatches):
            minibatch_Y = Y_train[ batch * minibatch_size : (batch+1)*minibatch_size ,:]
            lower_limit=batch * minibatch_size
            upper_limit=(batch+1)*minibatch_size-1
            for i in range(lower_limit,upper_limit):
                image_path= "/data/vernica_data/Toy_images/" + image_name_list[i] + ".jpeg"
                image = cv2.imread(image_path)
               # print(image.shape)
                image = cv2.resize(image, (224,224))
                #print(image.shape)
                minibatch_X[i - batch * minibatch_size ,:,:,:] = image
           
        # Calculate accuracy on the test set
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #"float"
            batch_accuracy = accuracy.eval({X: minibatch_X, Y: minibatch_Y})
            print ("Accuracy of batch %i: %f" % (batch, batch_accuracy))
            train_accuracy += batch_accuracy
        
        total_train_accuracy=train_accuracy/num_minibatches
        
        print("Training Accuracy = " + str(total_train_accuracy))
        if epoch % 1 == 0:
            checkpoint_file = os.path.join("/data/mmanning/vtest/vmodel/", 'checkpoint')
            saver.save(sess, checkpoint_file, global_step = epoch)
    # This is the end of tf.Session()
    # plot the cost
   # plt.plot(np.squeeze(costs))
   # plt.ylabel('cost')
   # plt.xlabel('iterations (per tens)')
   # plt.title("Learning rate =" + str(learning_rate))
   # plt.show()

    
        print(' y shape: ' + str( Y.shape) )
        print('y: ' + str(Y[0]))


        values, indices = tf.nn.top_k(Y, 5)
 
        print(' values shape: ' + str(values.shape))
        print('values: ' + str(values[0]))


        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant([str(i) for i in range(10)]))

        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        prediction_classes = table.lookup(tf.to_int64(indices))



        # Build the signature_def_map.
        classification_inputs = tf.saved_model.utils.build_tensor_info(
            serialized_tf_example)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
            prediction_classes)
        classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)


        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores
                },
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(Y)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))


        # try to save the model with SavedModelBuilder
        export_path = '/data/mmanning/bitnamitfs/vjain-model-data/'
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
               'predict_images':
                   prediction_signature,
                   tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                   classification_signature,
            },
            legacy_init_op=legacy_init_op)
        builder.save()
                
    return parameters, cost, total_train_accuracy
    
    close()

# RUNNING CNN
learning_rate = 0.001
num_epochs = 1
minibatch_size = 2
print_cost = True

parameters, cost, total_train_accuracy = model(learning_rate,num_epochs, minibatch_size, print_cost)

#print(type(parameters))

#print(parameters):wq
print([n.name for n in tf.get_default_graph().get_operations()])
