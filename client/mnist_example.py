import logging

from mnist_client import ProdClient

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# Make sure you have a model running on localhost:9000
host = 'localhost:8500'
model_name = 'mnist-model'
#model_name = 'saved-model'
model_version = 1

client = ProdClient(host, model_name, model_version)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

req_data = mnist.test.images[0]

prediction = client.predict(req_data, request_timeout=30)
logger.info('Prediction: {}'.format(prediction))
