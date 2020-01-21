# What is 3D-GAN?
> Generator and Discriminator networks both use 3D convolutional layers, instead of using 2D convolutions. If provided with
enough data, it can learn to generate 3D shapes with good visual quality.

# What is 3D convolutions?
3D convolution operations apply a 3D filter to the input data along the three directions, which are x, y, and z. This operation 
creates a stacked list of 3D feature maps. The shape of the output is similar to the shape of a cube or a cuboid.

![3D Convolutions](images/3D_Convolutions.png)

The highlighted part of the left cube is the input data. The kernel is in the middle, with a shape of (3, 3, 3). The block on the right-hand is the output of the convolution operation.

# The architecture of a 3D-GAN
Both of the networks in a 3D-GAN are deep convolutional neural networks. The generator network is, as usual, an upsampling network. It upsamples a noise vector (a vector from probabilistic latent space) to generate a 3D image with a shape that is similar to the input image in terms of its length, breadth, height, and channels. The discriminator network is a downsampling network. Using a series of 3D convolution operations and a dense layer, it identifies whether the input data provided to it is real or fake.