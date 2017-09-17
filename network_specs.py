import numpy as np
import tensorflow as tf

from inputstate import *
from layers import *
import cPickle

class network:
    def __init__(self, batch_size = 1, rng = None, load_file = None):
        #Create a single value if rng is not defined
		if(not rng): rng = np.random.RandomState(None):

        #Position Matrix
        self.input = tf.placeholder(shape=[None,None,None,None],dtype=tf.float32,name='input')

        self.batch_size = batch_size

        #Layer 0
        layer0_D3 = 12
		layer0_D5 = 20
		layer0 = xnetConvLayer(
			rng,
			self.input,
			(batch_size, num_channels, input_size, input_size),
			layer0_D5,
			layer0_D3
		)

        #Layer 1
        layer1_D3 = 16
		layer1_D5 = 16
		layer1 = xnetConvLayer(
			rng,
			self.layer0.output,
			(batch_size, layer0_D3+layer0_D5, input_size, input_size),
			layer1_D5,
			layer1_D3
		)

        #Layer 2
        layer2_D3 = 20
		layer2_D5 = 12
		layer2 = xnetConvLayer(
			rng,
			self.layer1.output,
			(batch_size, layer1_D3+layer1_D5, input_size, input_size),
			layer2_D5,
			layer2_D3
		)

        #Layer 3
        layer3_D3 = 24
		layer3_D5 = 8
		layer3 = xnetConvLayer(
			rng,
			self.layer2.output,
			(batch_size, layer2_D3+layer2_D5, input_size, input_size),
			layer3_D5,
			layer3_D3
		)

        #Layer 4
        layer4_D3 = 28
		layer4_D5 = 4
		layer4 = xnetConvLayer(
			rng,
			self.layer3.output,
			(batch_size, layer3_D3+layer3_D5, input_size, input_size),
			layer4_D5,
			layer4_D3
		)

        #Layer 5
        layer5_D3 = 28
		layer5_D5 = 4
		layer5 = xnetConvLayer(
			rng,
			self.layer4.output,
			(batch_size, layer4_D3+layer4_D5, input_size, input_size),
			layer5_D5,
			layer5_D3
		)

        #Layer 6
        layer6_D3 = 32
		layer6_D5 = 0
		layer6 = xnetConvLayer(
			rng,
			self.layer5.output,
			(batch_size, layer5_D3+layer5_D5, input_size, input_size),
			layer6_D5,
			layer6_D3
		)

        #Layer 7
        layer7_D3 = 32
		layer7_D5 = 0
		layer7 = xnetConvLayer(
			rng,
			self.layer6.output,
			(batch_size, layer6_D3+layer6_D5, input_size, input_size),
			layer7_D5,
			layer7_D3
		)

        #Layer 8
        layer8_D3 = 32
		layer8_D5 = 0
		layer8 = xnetConvLayer(
			rng,
			self.layer7.output,
			(batch_size, layer7_D3+layer7_D5, input_size, input_size),
			layer8_D5,
			layer8_D3
		)

        #Layer 9
        layer9_size = 5000
		layer9 = xnetFullyConnectedLayer(
			rng,
            input = layer8.output.flatten(2),
            number_in = (layer8_D3+layer8_D5)*input_size*input_size,
            number_out = layer9_size
		)

        #Layer 10
		layer10 = xnetSigmoidLayer(
			rng,
            input = layer8.output.flatten(2),
            number_in = layer9_size,
            number_out = boardsize*boardsize
		)

        self.output = 2*layer10.output-1

        self.params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params+ layer7.params + layer8.params + layer9.params + layer10.params

		self.mem_size = layer1.mem_size + layer2.mem_size + layer3.mem_size + layer4.mem_size + layer5.mem_size + layer6.mem_size + layer7.mem_size + layer8.mem_size + layer9.mem_size + layer10.mem_size






