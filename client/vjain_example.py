import logging

from mnist_client import ProdClient

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# Make sure you have a model running on localhost:9000
host = 'localhost:8500'
model_name = 'vjain-model'
#model_name = 'saved-model'
model_version = 1

client = ProdClient(host, model_name, model_version)


from PIL import Image
req_data = Image.open("/data/mmanning/vtest/client/predict_client/10000_left.jpeg")

prediction = client.predict(req_data, request_timeout=30)
logger.info('Prediction: {}'.format(prediction))
