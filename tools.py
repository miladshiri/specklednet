## Project: SpeckleDNet
## Developer: Milad Shiri
## @2020

import numpy as np
from PIL import Image


def construct_input_data(xm, xc, y, main_dims, comp_dims):
    Xm = np.empty((len(xm),
                   main_dims[0],
                   main_dims[1],
                   1))
    Xc = np.empty((len(xc),
                   comp_dims[0],
                   comp_dims[1],
                   1))
    Y = np.empty((len(y),))

    for i, item in enumerate(list(zip(xm, xc, y))):
        xm_image = Image.fromarray(item[0])
        xc_image = Image.fromarray(item[1])
        Xm [i,:, :, 0] = np.array(xm_image.resize((main_dims[0],
                                                  main_dims[1])))/255.0
        Xc [i,:, :, 0] = np.array(xc_image.resize((comp_dims[0],
                                                  comp_dims[1])))/255.0
        Y [i] = item[2]
    return Xm, Xc, Y

