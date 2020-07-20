##This script uses the dense layer to produce outputs
import numpy as np
from dense_layer import Dense_Layer

NO_OF_FEATURES = 4

input_batch = [[1, 2, 3, 2.5],
               [2, 5, 1, 2],
               [-1.5, 2.7, 3.3, -0.8]
              ]
#use the dense layer
layer_1 = Dense_Layer(NO_OF_FEATURES, 3)
output = layer_1.forward_pass(input_batch)
output = layer_1.activation_relu(output)
print("Output is: ", output)