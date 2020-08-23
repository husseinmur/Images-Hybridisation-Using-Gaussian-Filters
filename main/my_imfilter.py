import numpy as np
import matplotlib.pyplot as plt
import cv2

def cross_correlation_2d(image, kernel):
    x, y = kernel.shape
    a, b = image.shape[0], image.shape[1]
    out = np.zeros(image.shape)
    flat_kernel = []
    for row in kernel:
        for i in row:
            flat_kernel.append(i)
    channels = image.shape[2] if image.ndim == 3 else 1
    padding_size = (a+x-1,b+y-1,channels)
    padding = np.zeros(padding_size, dtype=image.dtype)
    coords = [[x//2,x//2+a],[y//2,y//2+b]]
    padding[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1]] = image[:, :, np.newaxis] if channels == 1 else image
    for i in range(a):
        for j in range(b):
           out[i, j] = flat_kernel @ padding[i:i+x, j:j+y].reshape((x*y, channels))
    return out

def convolve_2d(image, kernel):
    kernel = kernel[:,::-1]
    kernel = kernel[::-1]
    return cross_correlation_2d(image, kernel)

def gaussian_blur_kernel_2d(size,sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp((-1*((x-(size[0]-1)/2)**2+(y-(size[1]-1)/2)**2))/(2*sigma**2)), (size[0], size[1]))
    kernel /= np.sum(kernel)
    return kernel

def low_pass(size, image, sigma):
    return convolve_2d(image, gaussian_blur_kernel_2d(size,sigma))

def high_pass(size, image, sigma):
    return image - low_pass(size, image, sigma)

def create_hybrid_image(image1, image2, size, sigma):
    image1 = cv2.resize(image1,(200,200))
    image1 = image1.astype(np.float32) / 255.0
    image2 = cv2.resize(image2,(200,200))
    image2 = image2.astype(np.float32) / 255.0
    out1 = 1.4*low_pass(size, image1, sigma) + 0.6*high_pass(size, image2,sigma)
    out2 = 1.4*low_pass(size, image2, sigma) + 0.6*high_pass(size, image1,sigma)
    return (out1*255).clip(0, 255).astype(np.uint8), (out2*255).clip(0, 255).astype(np.uint8)