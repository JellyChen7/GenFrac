import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
import imageio


f = h5py.File('denoising100.h5','r')
f.keys()                            #可以查看所有的主键
x = f['x_array'][:]                    #取出主键为data的所有的键值
f.close()

x_posterior = x[0, :, 0, :, :].squeeze()

lines = [
    ((80.48, 12.80), (115.20, 47.52)),  # 第一条线
    ((68.06, 45.44), (12.80, 100.70)),  # 第二条线
    ((78.10, 76.04), (115.20, 113.15))   # 第三条线
]

x_mean = np.mean(x_posterior, axis=0)

plt.imshow(x_mean, vmin=0, vmax=0.3, cmap='GnBu')

for line in lines:
    start, end = line
    plt.plot([start[0], end[0]], [start[1], end[1]], color='k', marker=None, linestyle='--', linewidth=0.5)
plt.savefig('x_dist.png', dpi=600)




indexes = [1, 3, 5, 7, 10, 15, 20, 50, 60, 80, 100, 150, 200, 400, 600, 800, 1000]

for i in indexes:
    sampled_step = x[i-1, 1, 0, :, :].squeeze()
    plt.imshow(sampled_step, vmin=0, vmax=1, cmap='GnBu')
    plt.xlabel('Denoising step: {}'.format(i))
    plt.savefig('./denoising/sampled_step{}.png'.format(i), dpi=600)
    np.savetxt('./denoising/sampled_step{}.txt'.format(i), sampled_step.flatten())

gif_images = []
for i in reversed(indexes):
    gif_images.append(imageio.imread('./denoising/sampled_step{}.png'.format(i)))
imageio.mimsave("test.gif",gif_images,fps=5)

