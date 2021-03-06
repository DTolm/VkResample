[![Build Status](https://travis-ci.com/DTolm/VkResample.svg?branch=main)](https://travis-ci.com/DTolm/VkResample)
# VkResample - Vulkan upscaler based on VkFFT library
VkResample is a demonstration that FFT upscaling and transition to the frequency domain can be done on GPUs in real-time. VkResample is based on VkFFT library (https://github.com/DTolm/VkFFT).

## What VkResample does
Upscaling and supersampling became hot topics in the modern GPU world. There are a lot of options available already, starting from the simplest nearest neighbor algorithm, where each pixel is split in multiple, and coming to Nvidia's DLSS technology, which uses trained neural networks. One of such options is a Fast Fourier Transform (FFT) based upscaling.

FFT is an essential algorithm in image processing. It is used for convolution calculations (see: convolution theorem), filtering (high- and low-bandwidth passes), denoising (Wiener filter). FFT can be used as well for Lanczos upsampling, or in other words, for convolutions with sinc window. In the frequency domain, sinc window corresponds to a step-wise function, which is then multiplied with a centered FFT of an image, padded with zeros to the required size and transformed back to the spatial domain to retrieve upscaled image.

The main time consuming part, that was previously immensely limiting FFT-based algorithms, was forward and inverse FFTs themselves. The computational cost of them was simply too high to be performed in real-time. However, modern advances in general purpose GPU computing allow for efficient parallelization of FFT, which is done in a form of Vulkan FFT library - VkFFT. It can be used as a part of a rendering process to perform frequency based computations on a frame before showing it to the user.

VkResample uses various optimizations available in VkFFT package, such as R2C/C2R mode and native zero padding support, which greatly reduce the amount of memory transfers and computations. With them enabled, it is possible to upscale 2048x1024 image to 4096x2048 in under 2ms on Nvidia GTX 1660Ti GPU. Measured time covers command buffer submission and execution, which include data transfers to the chip, FFT algorithm, modifications in frequency domain and inverse transformation with its own data trasnfers. Support for arbitrary resolutions will be added in one of the next updates of VkFFT.

After upscaling, VkResample does a sharpening filter pass (implementation, similar to FidelityFX-CAS), which improves the final image quality.

Possible improvements to this algorithm can include: implementing the Discrete Cosine Transform, which is better suited for real-world images; using additional data from previous frames and/or motion vectors; more low-precision tests and optimizations; using deep learning methods in the frequency domain. As of now, VkResample is more of a proof of concept that can be greatly enchanced in the future.

VkResample supports upscaling with an arbitrary non-integer factor (one condition is that output image dimensions must be representabale as multiplication of 2s, 3s and 5s).

Below you can find a collection of screenshots details comparison from Cyberpunk 2077 game upscaled 2x using nearest neighbor method (NN), FFT method + sharpener(FFT) and rendered in native resolution (Native). All of the images can be found in the samples folder as well.

![alt text](https://github.com/DTolm/VkResample/blob/main/samples/close_people.png?raw=true)
![alt text](https://github.com/DTolm/VkResample/blob/main/samples/distant_people.png?raw=true)
![alt text](https://github.com/DTolm/VkResample/blob/main/samples/skyscraper.png?raw=true)
![alt text](https://github.com/DTolm/VkResample/blob/main/samples/trees.png?raw=true)
![alt text](https://github.com/DTolm/VkResample/blob/main/samples/car.png?raw=true)

## Future release plan
 - ##### Planned
    - Different frequency domain kernel experiments
	- Better precision utilization - reading data in uint8, calculations in half precision
	- DCT support, which will reduce typical FFT boundary problems

## Installation
Sample CMakeLists.txt file configures project based on VkResample.cpp file, VkFFT library, stb_image library, half library and glslang compiler. Vulkan 1.0 is required. Windows executable is also available.

## Command-line interface
VkResample has a command-line interface with the following set of commands:\
-h: print help\
-devices: print the list of available GPU devices\
-d X: select GPU device (default 0)\
-u X: specify upscale factor (float, make sure that upscaled image can be represented as a multiplication of 2s, 3s, 5s and 7s)\
-p X: specify precision (0 - single, 1 - double, 2 - half, default - single)\
-s X: specify sharpening factor, range 0.0-0.2 (default 0.2) \
-n X: specify how many times to perform upscale. This removes dispatch overhead and will show the real application performance (default 1)\
Single image mode:\
	-i NAME: specify input png file path\
	-o NAME: specify output png file path (default X_X_upscale.png)\
Batched mode:\
	-ifolder X: specify input folder plus file prefix, like inp/img\
	-ofolder X: specify output folder plus file prefix, like outp/img\
	-numfiles X: specify how many images to upscale. They should have names like prefix + 000001.png with numbers padded with zeros to six digits. Temporary limitation.\
	-numthreads X: specify how many threads to launch. Used to speed up png reads\
		
The simplest way to launch a 2x upscaler will be: -i no_upscaling.png -u 2
A 2x upscaler in half-precision batched mode with 16 threads can be launched as: -ifolder inp -ofolder outp -numthreads 16 -numfiles 200 -u 2 -p 2

## Contact information
Initial version of VkResample is developed by Tolmachev Dmitrii\
E-mail 1: <dtolm96@gmail.com>
