## SECTION ONE - WHY USE BATCHES
# It makes use of GPU rather than the CPU since a 
#batch can run parrallel calculations using GPU.
# Batches also helps with generalization 

## SECTION TWO - CODING A 4 INPUT WITH 1 (3 NEURON HIDDEN LAYER)
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, 1, 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]
weights = np.array(weights).T
biases = [2, 3, 0.5]

output = np.dot(inputs, weights) + biases
print("The output of 3 batches is: ", output)