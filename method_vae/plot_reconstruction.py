from predictor import Predictor

from torchvision.utils import make_grid
from method_vae.utils import data_init
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


config = Predictor.default_config


project_name = 'PETALS'
expt_name = f'vae_1'
grid_nrow = 3
n_images = 9


# load model
predictor = Predictor.load(expt_name, project_name)
predictor.model.eval()


# run model
train_dataloader = data_init('train', predictor.config)
x, labels = next(iter(train_dataloader))
z, mu, logvar = predictor.model.encode(x)
x_pred = predictor.model.decode(z)


# plot
plt.subplot(1, 2, 1)
plt.imshow(make_grid(x[0:n_images], nrow=grid_nrow).permute(1, 2, 0))
plt.axis('off')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(make_grid(x_pred[0:n_images], nrow=grid_nrow).permute(1, 2, 0))
plt.axis('off')
plt.title('Reconstruction')
