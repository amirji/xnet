import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class xnetConvLayer:
    def __init__(self, rng, input, input_shape, num_D5_filters, num_D3_filters, params = None):

        W3_bound = np.sqrt(6. / (7*(input_shape[1] + num_D3_filters)))
        W5_bound = np.sqrt(6. / (19*(input_shape[1] + num_D5_filters)))


        if(params):
        	self.W3_values = params[1]
        else:
	        self.W3_values = theano.shared(
	            np.asarray(
	                rng.uniform(
	                    low=-W3_bound,
	                    high=W3_bound,
	                    size=(num_D3_filters,input_shape[1],7)
	                ),
	                dtype=theano.config.floatX
	            ),
	            borrow = True
	        )
