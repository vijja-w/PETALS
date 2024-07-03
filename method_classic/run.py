from method_classic.utils import data_init
from method_classic.predictor import Predictor

config = Predictor.default_config

train_dataloader = data_init('train', config)
batch = next(iter(train_dataloader))
images = batch[0]
predictor = Predictor(config)
predictor.train(images)
predictor.test()
