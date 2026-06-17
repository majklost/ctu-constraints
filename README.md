# Different architectures and how to make them work

## AIC-Net

We run AIC-Net on a system running Debian GNU/Linux 11, with Python 3.12.4, PyTorch 2.3.1, and CUDA 12.4. For a full list of software packages and version numbers, see the Conda environment file environment.yml.

## HybridGNet - postpone it

In case the installation fails, you can build your own enviroment.

Conda dependencies:
-PyTorch 1.10.0
-Torchvision
-PyTorch Geometric
-Scipy
-Numpy
-Pandas
-Scikit-learn
-Scikit-image

Pip dependencies:
-medpy==0.4.0
-opencv-python==4.5.4.60

## SGDIR - alternative to voxelmorph formulation

- <https://github.com/mattkia/SGDIR>
