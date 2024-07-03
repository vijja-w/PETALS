import torch
torch.set_float32_matmul_precision('high')

import numpy as np
from PIL import Image
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import Scale, Frame
from pydantic import BaseModel
from typing import Tuple


class ColourThreshold:
    class Config(BaseModel):
        lower_threshold: Tuple[float, float, float] = (0.1159, 0.4194, 0.4196)
        upper_threshold: Tuple[float, float, float] = (0.1935, 1.0000, 0.8706)
        model_name: str = 'colour_threshold'

    default_config = Config().dict()

    def __init__(self, config):
        _ = self.Config(**config).dict()

        self.lower_threshold = torch.tensor(config['lower_threshold'])
        self.upper_threshold = torch.tensor(config['upper_threshold'])

        self.pixel_values = [self.lower_threshold.numpy(), self.upper_threshold.numpy()]

    def __call__(self, images):
        pgreen = self.get_pgreen(images).unsqueeze(-1)
        z = torch.cat((torch.zeros_like(pgreen), pgreen), dim=1)
        return z

    def get_pgreen(self, images):
        mask = self.get_mask(images)
        pgreen = mask.reshape(mask.shape[0], -1).sum(dim=1)/(mask.shape[-1]*mask.shape[-2])
        return pgreen

    def get_mask(self, images):
        if len(images.shape) != 4:
            raise ValueError(f'Expect images to be shape: [N, 3, height, width]')

        images = self.rgb2hsv(images)
        mask = (images >= self.lower_threshold.view(1, 3, 1, 1)).all(dim=1) & (images <= self.upper_threshold.view(1, 3, 1, 1)).all(dim=1)
        return mask

    def segment_image(self, images):
        mask = self.get_mask(images)
        visualization = np.zeros_like(mask)
        visualization[mask] = (255, 255, 255)
        return Image.fromarray(visualization, 'RGB')

    def rgb2hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def on_click(self, event):
        x, y = int(event.xdata), int(event.ydata)
        pixel_value = self.images[self.img_idx, :, y, x]
        pixel_value = self.rgb2hsv(pixel_value.view(1, 3, 1, 1))
        pixel_value = pixel_value.view(3).numpy()
        self.pixel_values.append(pixel_value)
        self.lower_threshold = torch.tensor(np.array(self.pixel_values).min(axis=0))
        self.upper_threshold = torch.tensor(np.array(self.pixel_values).max(axis=0))
        self.update_slider_position()
        self.remask()

    def prev_image(self):
        self.img_idx = (self.img_idx - 1) % self.n_images
        self.update_image()

    def next_image(self):
        self.img_idx = (self.img_idx + 1) % self.n_images
        self.update_image()

    def remask(self):
        self.mask = (self.images_hsv >= self.lower_threshold.view(1, 3, 1, 1)).all(dim=1) & (self.images_hsv <= self.upper_threshold.view(1, 3, 1, 1)).all(dim=1)
        self.update_image()

    def clear(self):
        self.pixel_values = []
        self.lower_threshold = torch.tensor([0., 0., 0.])
        self.upper_threshold = torch.tensor([0., 0., 0.])
        self.mask = (self.images >= self.lower_threshold.view(1, 3, 1, 1)).all(dim=1) & (
                    self.images <= self.upper_threshold.view(1, 3, 1, 1)).all(dim=1)
        self.update_image()
        self.update_slider_position()

    def close_plot(self):
        self.root.quit()
        self.root.destroy()
        print(f'lower_threshold = {self.lower_threshold}')
        print(f'upper_threshold = {self.upper_threshold}')

    def undo(self):
        del self.pixel_values[-1]
        self.lower_threshold = torch.tensor(np.array(self.pixel_values).min(axis=0))
        self.upper_threshold = torch.tensor(np.array(self.pixel_values).max(axis=0))
        self.remask()

    def update_image(self):
        image = self.images[self.img_idx]
        m = self.mask[self.img_idx]
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[0].axis('off')
        self.axs[1].axis('off')
        self.axs[0].imshow(np.transpose(image, (1, 2, 0)))
        self.axs[1].imshow(m)
        self.fig.suptitle(f'image: {self.img_idx}')
        self.canvas.draw()

    def update_lower_h(self, value):
        new_value = float(value)
        self.lower_threshold[0] = new_value
        self.remask()

    def update_lower_s(self, value):
        new_value = float(value)
        self.lower_threshold[1] = new_value
        self.remask()

    def update_lower_v(self, value):
        new_value = float(value)
        self.lower_threshold[2] = new_value
        self.remask()

    def update_upper_h(self, value):
        new_value = float(value)
        self.upper_threshold[0] = new_value
        self.remask()

    def update_upper_s(self, value):
        new_value = float(value)
        self.upper_threshold[1] = new_value
        self.remask()

    def update_upper_v(self, value):
        new_value = float(value)
        self.upper_threshold[2] = new_value
        self.remask()

    def update_slider_position(self):
        self.lower_h_slider.set(self.lower_threshold[0].item())
        self.lower_s_slider.set(self.lower_threshold[1].item())
        self.lower_v_slider.set(self.lower_threshold[2].item())
        self.upper_h_slider.set(self.upper_threshold[0].item())
        self.upper_s_slider.set(self.upper_threshold[1].item())
        self.upper_v_slider.set(self.upper_threshold[2].item())

    def set_hsv(self):
        self.pixel_values.append(self.lower_threshold.numpy())
        self.pixel_values.append(self.upper_threshold.numpy())

    def train(self, images):
        self.images = images
        self.images_hsv = self.rgb2hsv(images)
        self.n_images = images.shape[0]
        self.img_idx = 0
        self.mask = (self.images_hsv >= self.lower_threshold.view(1, 3, 1, 1)).all(dim=1) & (self.images_hsv <= self.upper_threshold.view(1, 3, 1, 1)).all(dim=1)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.axs[0].axis('off')
        self.axs[1].axis('off')
        self.root = tk.Tk()
        self.root.wm_title("Colour Threshold Training")

        left_frame = Frame(self.root)
        left_frame.pack(side='left')
        right_frame = Frame(self.root)
        right_frame.pack(side='right')

        lower_h_label = tk.Label(right_frame, text="Lower H")
        lower_h_label.pack()
        self.lower_h_slider = Scale(right_frame, from_=0, to=1, orient="horizontal", length=200,
                                            resolution=0.005, command=self.update_lower_h)
        self.lower_h_slider.set(self.lower_threshold[0].item())
        self.lower_h_slider.pack()

        lower_s_label = tk.Label(right_frame, text="Lower S")
        lower_s_label.pack()
        self.lower_s_slider = Scale(right_frame, from_=0, to=1, orient="horizontal", length=200,
                                            resolution=0.005, command=self.update_lower_s)
        self.lower_s_slider.set(self.lower_threshold[1].item())
        self.lower_s_slider.pack()

        lower_v_label = tk.Label(right_frame, text="Lower V")
        lower_v_label.pack()
        self.lower_v_slider = Scale(right_frame, from_=0, to=1, orient="horizontal", length=200,
                                    resolution=0.005, command=self.update_lower_v)
        self.lower_v_slider.set(self.lower_threshold[2].item())
        self.lower_v_slider.pack()

        upper_h_label = tk.Label(right_frame, text="Upper H")
        upper_h_label.pack()
        self.upper_h_slider = Scale(right_frame, from_=0, to=1, orient="horizontal", length=200,
                                            resolution=0.005, command=self.update_upper_h)
        self.upper_h_slider.set(self.upper_threshold[0].item())
        self.upper_h_slider.pack()

        upper_s_label = tk.Label(right_frame, text="Upper S")
        upper_s_label.pack()
        self.upper_s_slider = Scale(right_frame, from_=0, to=1, orient="horizontal", length=200,
                                            resolution=0.005, command=self.update_upper_s)
        self.upper_s_slider.set(self.upper_threshold[1].item())
        self.upper_s_slider.pack()

        upper_v_label = tk.Label(right_frame, text="Upper V")
        upper_v_label.pack()
        self.upper_v_slider = Scale(right_frame, from_=0, to=1, orient="horizontal", length=200,
                                    resolution=0.005, command=self.update_upper_v)
        self.upper_v_slider.set(self.upper_threshold[2].item())
        self.upper_v_slider.pack()

        set_hsv_button = tk.Button(right_frame, text="set_hsv", command=self.set_hsv)
        set_hsv_button.pack()

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack()

        self.update_image()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        prev_button = tk.Button(left_frame, text="Previous Image", command=self.prev_image)
        prev_button.pack(side='left')

        next_button = tk.Button(left_frame, text="Next Image", command=self.next_image)
        next_button.pack(side='left')

        close_button = tk.Button(left_frame, text="Done", command=self.close_plot)
        close_button.pack(side='right')

        clear_button = tk.Button(left_frame, text="Clear", command=self.clear)
        clear_button.pack(side='right')

        undo_button = tk.Button(left_frame, text="Undo", command=self.undo)
        undo_button.pack(side='right')

        self.root.mainloop()
