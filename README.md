# tensorflowserveGrpcTest
Learning TFServing and gRPC clients




https://github.com/bitnami/bitnami-docker-tensorflow-serving

de761375-8d25-4e32-8f26-918de2c8338c


[mmanning@vjain bitnamitfs]$ docker run  -v /home/mmanning/bitnamitfs/:/bitnami bitnami/tensorflow-serving


[mmanning@vjain bitnamitfs]$ docker network create app-tier --driver bridge
[mmanning@vjain bitnamitfs]$ docker run  -v /home/mmanning/bitnamitfs/:/bitnami --network app-tier bitnami/tensorflow-serving


docker run -it --rm --volume /home/vernica/model-data/:/bitnami/model-data  --network app-tier  bitnami/tensorflow-inception:latest inception_client --server=tensorflow-serving:8500  --image=/home/vernica/images/18059_left.jpeg


docker run -it --rm --volume /home/vernica/:/bitnami/  --network app-tier  bitnami/tensorflow-inception:latest inception_client --server=mmmanning/tensorflow-serving:8500  --image=/bitnami/images/18059_left.jpeg







TO GENERATE THE PROTOBUFS

MUST LINK TF TO TF-serving
(py35) [mmanning@vjain mmanning]$ ls -al serving/
total 60
...
lrwxrwxrwx.  1 mmanning mmanning    37 Jul 22 14:47 tensorflow -> /data/mmanning/tensorflow/tensorflow/


Using the bitnami container

My working version: [mmanning@vjain ~]$ docker run -it mmanning/tensorflow-serving bash
from inside the container
root@96b5362d6b12:/# cat /opt/bitnami/tensorflow-serving/bin/tensorflow-serving.sh shows:
--model_config_file="/opt/bitnami/tensorflow-serving/conf/tensorflow-serving.conf"

which is linked to the mapped drive /bitnami/
root@96b5362d6b12:/opt/bitnami/tensorflow-serving# ls -al conf
lrwxrwxrwx. 1 root root 32 Jul 12 06:07 conf -> /bitnami/tensorflow-serving/conf

on the dsvm use the /data/mmanning/bitnamitfs folder
so put the config in /data/mmanning/bitnamitfs/tensorflow-serving/config


This runs the simple model from the Medium post:
docker run  -v /data/mmanning/bitnamitfs/:/bitnami mmanning/tensorflow-serving


/opt/bitnami/tensorflow-serving/conf/tensorflow-serving.conf


BUILD MNIST MODEL ON DSVM
 https://www.tensorflow.org/serving/serving_basic

(py35) [mmanning@vjain mmanning]$ python ./vtest/serving/tensorflow_serving/example/mnist_saved_model.py .buildmnist/mnist_model
Training model...
Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
2018-07-26 05:59:09.890646: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-07-26 05:59:09.914479: E tensorflow/stream_executor/cuda/cuda_driver.cc:406] failed call to cuInit: CUDA_ERROR_UNKNOWN
2018-07-26 05:59:09.914567: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:145] kernel driver does not appear to be running on this host (vjain): /proc/driver/nvidia/version does not exist
training accuracy 0.9092
Done training!
Exporting trained model to b'.buildmnist/mnist_model/1'
Done exporting!


MOVE RESULTING .pb an variables dir to /data/mmanning/bitnamitfs/mnist-model-data


(py35) [mmanning@vjain mmanning]$ !cat
cat /data/mmanning/bitnamitfs/tensorflow-serving/conf/tensorflow-serving.conf
model_config_list: {
config: {
name: "simple-model",
base_path: "/bitnami/model-data",
model_platform: "tensorflow",
}
config: {
name: "mnist-model",
base_path: "/bitnami/mnist_model/",
model_platform: "tensorflow",
}
}

******************************************************
(py35) [mmanning@vjain bitnamitfs]$ docker run -p 8500:8500 -p 8501:8501  -v /data/mmanning/bitnamitfs/:/bitnami mmanning/tensorflow-serving
******************************************************

RUN THE CLIENT IN /data/mmanning/vtest/client/predict_client

driver program is example.py for simple-model_dir


(py35) [mmanning@vjain predict_client]$ python ./example.py
2018-07-26 07:14:52,392 - DEBUG - ProdClient - -------- Initilize  the log file -------------
2018-07-26 07:14:52,392 - INFO - ProdClient - Sending request to tfserving model
2018-07-26 07:14:52,392 - INFO - ProdClient - Host: localhost:8500
2018-07-26 07:14:52,392 - INFO - ProdClient - Model name: simple-model
2018-07-26 07:14:52,392 - INFO - ProdClient - Model version: 1
2018-07-26 07:14:52,397 - DEBUG - ProdClient - Establishing insecure channel took: 0.004400968551635742
2018-07-26 07:14:52,397 - DEBUG - ProdClient - Creating stub took: 7.724761962890625e-05
2018-07-26 07:14:52,397 - DEBUG - ProdClient - Creating request object took: 1.9788742065429688e-05
2018-07-26 07:14:52,397 - DEBUG - ProdClient - Making tensor protos took: 0.00012564659118652344
2018-07-26 07:14:52,405 - DEBUG - ProdClient - Actual request took: 0.007737398147583008 seconds
2018-07-26 07:14:52,405 - DEBUG - util - Key: add, shape: [1]
2018-07-26 07:14:52,405 - INFO - ProdClient - Got predict_response with keys: ['add']
2018-07-26 07:14:52,405 - INFO - __main__ - Prediction: {'add': 12}
(py35) [mmanning@vjain predict_client]


CREATING THE MNIST CLIENT
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py


(py35) [mmanning@vjain predict_client]$ pwd
/data/mmanning/vtest/client/predict_client
(py35) [mmanning@vjain predict_client]$ python ./mnist_example.py
2018-07-26 08:38:35,251 - DEBUG - ProdClient - -------- Initilize  the log file -------------
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2018-07-26 08:38:36,825 - INFO - ProdClient - Sending request to tfserving model
2018-07-26 08:38:36,826 - INFO - ProdClient - Host: localhost:8500
2018-07-26 08:38:36,826 - INFO - ProdClient - Model name: mnist-model
2018-07-26 08:38:36,826 - INFO - ProdClient - Model version: 1
2018-07-26 08:38:36,831 - DEBUG - ProdClient - Establishing insecure channel took: 0.004760265350341797
2018-07-26 08:38:36,831 - DEBUG - ProdClient - Creating stub took: 5.650520324707031e-05
2018-07-26 08:38:36,831 - DEBUG - ProdClient - Creating request object took: 2.9087066650390625e-05
2018-07-26 08:38:36,839 - DEBUG - ProdClient - Actual request took: 0.007582187652587891 seconds
2018-07-26 08:38:36,839 - DEBUG - util - Key: scores, shape: [1, 10]
2018-07-26 08:38:36,840 - INFO - ProdClient - Got predict_response with keys: ['scores']
2018-07-26 08:38:36,841 - INFO - __main__ - Prediction: {'scores': array([[  2.04608095e-05,   1.72720882e-09,   7.74099099e-05,
          3.64777376e-03,   1.25222709e-06,   2.27521577e-05,
          1.14668754e-08,   9.95974720e-01,   3.68832661e-05,
          2.18785644e-04]])}


Run the inception model until Vernice's model is ready:
https://github.com/bitnami/bitnami-docker-tensorflow-inception

mediumexample
https://medium.com/epigramai/tensorflow-serving-101-pt-1-a79726f7c103


GRPC articles:
https://medium.com/@KailaGaurav/deploying-object-detection-model-with-tensorflow-serving-part-3-6a3d59c1e7c0



TENSORBOARD
work mac: /Users/mmanning/Dev/code/mnist_model/

(tensorflow) mmanning:~/Dev/code/mnist_model/2:$pwd
/Users/mmanning/Dev/code/mnist_model/2
(tensorflow) mmanning:~/Dev/code/mnist_model/2:$python /Users/mmanning/anaconda3/envs/tensorflow/lib/python3.6/site-packages/tensorflow/python/tools/import_pb_to_tensorboard.py --model_dir ./classify_mnist_graph_def.pb --log_dir .
2018-07-26 12:25:21.331018: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Model Imported. Visualize by running: tensorboard --logdir=.
(tensorflow) mmanning:~/Dev/code/mnist_model/2:$tensorboard --logdir=.
TensorBoard 1.9.0 at http://Maurices-MBP:6006 (Press CTRL+C to quit)


