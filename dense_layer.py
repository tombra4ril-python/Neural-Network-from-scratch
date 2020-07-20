import numpy as np

class Dense_Layer:
  def __init__(self, input_features, neurons):
    #producing the shape this way, you do not have to do transpose
    np.random.seed(0)
    self.weights = 0.1 * np.random.randn(input_features, neurons)
    self.baises = np.zeros((1, neurons))
  
  def forward_pass(self, input_batch):
    output = np.dot(input_batch, self.weights) + self.baises
    return output
  
  def activation_relu(self, inputs):
    output = np.maximum(0, inputs)
    return output