import tensorflow as tf
import sys

class ResNet():
    def __init__(self, training=True):
        self.training = training

    def add_residual_block(self, inp, block_number, in_channels, out_channels):
        block_number = str(block_number) #This was used for providing a unqiue name to each layer.
        skip = tf.identity(input)
        x = inp

        if in_channels != out_channels:
            #TODO: perform 1x1 convolution to match output dimensions for skip connection
           inp = tf.layers.conv2d(inp, in_channels, kernel_size=1, stride=1)

        #TODO: Implement one residual block (Convolution, batch_norm, relu)
        ...
        inp = tf.layers.conv2d(inp, out_channels, kernel_size=3, stride=1)
        inp = tf_contrib.layers.batch_norm(inp)
        inp = tf.nn.relu(x)

        #TODO: Add the skip connection and ReLU the output
        ...
        return tf.nn.relu(inp + x)

    def forward(self, data):
        #TODO: 64 7x7 convolutions followed by batchnorm, relu, 3x3 maxpool with stride 2
        ...
        out = data

        for _ in range(64):
            out = add_convolution(out, 7, 112, 112, 'SAME')
        out = tf_contrib.layers.batch_norm(out)
        out = tf.nn.relu(out)
        out = tf.nn.maxpool(out, 3, stride = 2)


        #TODO: Add residual blocks of the appropriate size. See the diagram linked in the README for more details on the architecture.
        # Use the add_residual_block helper function
        
        out = add_residual_block(out, 1, 112, 56)
        out = add_residual_block(out, 2, 56, 28)
        out = add_residual_block(out, 3, 28, 14)
        out = add_residual_block(out, 4, 14, 7)

        #TODO: perform global average pooling on each feature map to get 4 output channels
        ...
        out = tf.reduce_mean(out, axis=[1, 2])
        logits = out
        return logits

    def add_convolution(self,
                        inp,
                        name,
                        filter_size,
                        input_channels,
                        output_channels,
                        padding):
        #TODO: Implement a convolutional layer with the above specifications
        ...

        return nn.Conv2d(inp, filter_size, input_channels, output_channels, padding) 

