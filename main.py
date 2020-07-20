## Section 1
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias

print("Output: ", output)

## Section 2
#for a 4 input 3 Neurons and 1 output neural network
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]
          ]
biases = [2, 3, 0.5]
output = []

for neuron_weights, neuron_bias in zip(weights, biases):
  value = 0
  for weight, input in zip(neuron_weights, inputs):
    value += input * weight
  output.append(value + neuron_bias)

print("Section 2: ", output)