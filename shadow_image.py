import numpy as np
from tqdm import trange
from matplotlib import pyplot as pl


s = np.load("shadow_image.npy")
z = np.load("height_image.npy")
    
fg = pl.figure(1, (19.2, 10.8))
ax = fg.add_axes([0.025, 0.01, .93, .99])
cx = fg.add_axes([0.96, 0.04, .01, .93])
im = ax.imshow(s, cmap=pl.cm.binary,
                   origin="lower")
im = ax.imshow(z, alpha=0.6,
                   cmap=pl.cm.magma_r,
                   origin="lower")
fg.colorbar(im, cax=cx).set_label("height [px]")
pl.savefig("shadow_image.png")
