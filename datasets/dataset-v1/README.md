# Dataset v1


This dataset is a synthetic dataset and it has following features:
- First, images of (360, 4096) are generated for each detector, L1 and H1.
- Abs function is applied to the images.
- Images are then transformed to (360, 256, 2): Max function is applied each 16 pixels in the time axis. This is done to reduce the size of the dataset. 
- The last dimension is for the H1 and L1, respectively.
- It has 1024 images of signals and 1024 images of noise.

## Generation parameters

- depth: 10-25.