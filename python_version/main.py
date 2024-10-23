


import math
from typing import Any, List












def activations_tanh_activate(x: float):
  return math.tanh(x) # [-1.0..1.0]

def activations_tanh_derive(x: float):
  return 1.0 - math.tanh(x * x) # [-1.0..1.0]










from abc import ABC

class AbstractClassNeuron(ABC):
  _outputValue: float = 0
  _gradientOutputValue: float = 0
  _isBias: bool





class Connection:
  input: AbstractClassNeuron
  output: AbstractClassNeuron
  weight: float

  def __init__(self, input: AbstractClassNeuron, output: AbstractClassNeuron):
    self.input = input
    self.output = output
    self.weight = random.random() - random.random() # [-1..1]






class Neuron(AbstractClassNeuron):

  _inputConnections: List[Connection]
  _outputConnections: List[Connection]
  _outputValue: float
  _gradientOutputValue: float
  _isBias: bool

  def __init__(self, isBias: bool):

    self._inputConnections = []
    self._outputConnections = []
    self._outputValue = 0
    self._gradientOutputValue = 0

    self._isBias = isBias
    if self._isBias is True:
      self._outputValue = 1

  def feedForward(self):

    # is bias -> skip
    if self._isBias is True:
      return

    # is part of the input layer -> skip
    if len(self._inputConnections) == 0:
      return

    sum = 0

    for inputConnection in self._inputConnections:
      sum += inputConnection.input._outputValue * inputConnection.weight

    self._outputValue = activations_tanh_activate(sum)

  def calculateGradient_output(self, targetValue: float):

    # is not part of the output layer -> crash
    if len(self._inputConnections) == 0 or len(self._outputConnections) != 0:
      raise TypeError("not part of a output layer")

    delta = targetValue - self._outputValue
    self._gradientOutputValue = delta * activations_tanh_derive(self._outputValue)

  def calculateGradient_hidden(self):

    # is not part of a hidden layer -> crash
    if self._isBias is False and len(self._inputConnections) == 0:
      raise TypeError(f"not part of a hidden layer (no inputs)")
    if len(self._outputConnections) == 0:
      raise TypeError(f"not part of a hidden layer (no outputs)")

    sum = 0

    for outputConnection in self._outputConnections:
      if outputConnection.output._isBias:
        continue
      sum += outputConnection.weight * outputConnection.output._gradientOutputValue

    self._gradientOutputValue = sum * activations_tanh_derive(self._outputValue)

  def updateInputWeights(self):
    # The weights to be updated are in the Connection container
    # in the neurons in the preceding layer

    k_learningRate = 0.15

    for inputConnection in self._inputConnections:
      inputConnection.weight += k_learningRate * inputConnection.input._outputValue * self._gradientOutputValue













k_minSamples = 100



import random

class NeuralNetwork:

  _layers: List[List[Neuron]] = []
  _error: float = 0
  _recentAvgError: float = 0


  def __init__(self, topology: List[float]):

    if (len(topology) < 2):
      raise TypeError("invalid amount of layers")

    for (index, totalNeurons) in enumerate(topology):
      if totalNeurons <= 0:
        raise TypeError(f"invalid amount of layer neurons in layer {index} ({totalNeurons})")

    self._buildTheLayers(topology)
    self._connectTheLayers()


  def _buildTheLayers(self, topology: List[float]):

    for ii in range(len(topology)):

      newLayer: List[Neuron] = []

      for _ in range(topology[ii]):
        newLayer.append(Neuron(False))

      isLastLayer = (ii + 1 == len(topology))
      if isLastLayer is False:
        newLayer.append(Neuron(True))

      self._layers.append(newLayer)

  def _connectTheLayers(self):

    for ii in range(len(self._layers) - 1):

      prevLayer = self._layers[ii]
      nextLayer = self._layers[ii + 1]

      for prevNeuron in prevLayer:
        for nextNeuron in nextLayer:

          if nextNeuron._isBias:
            continue

          newConnection = Connection(prevNeuron, nextNeuron)

          prevNeuron._outputConnections.append(newConnection)
          nextNeuron._inputConnections.append(newConnection)

  def feedForward(self, inputValues: List[float]) -> List[float]:

    # skip the bias neuron -> -1
    total_input_neurons = len(self._layers[0]) - 1

    if len(inputValues) != total_input_neurons:
      raise TypeError("invalid amount of input values")

    # set the input values
    for (index, value) in enumerate(inputValues):
      self._layers[0][index]._outputValue = value

    for layer in self._layers:
      for neuron in layer:
        neuron.feedForward()

    # for neuron in self._layers[0]:
    #   print(f"neuron {neuron._outputValue}")


    outputValues: List[float] = []

    outputLayer = self._layers[-1]
    for neuron in outputLayer:
      outputValues.append(neuron._outputValue)

    return outputValues

  def backProp(self, targetValues: List[float]):

    outputLayer = self._layers[-1]

    if len(targetValues) != len(outputLayer):
      raise TypeError("invalid amount of input values")

    #
    # error
    #

    self._error = 0

    for (index, outputNeuron) in enumerate(outputLayer):
      delta = targetValues[index] - outputNeuron._outputValue
      self._error += delta * delta

    self._error /= len(outputLayer) # get average error squared
    self._error = math.sqrt(self._error) # RMS

    self._recentAvgError = (self._recentAvgError * k_minSamples + self._error) / (k_minSamples + 1.0)

    #
    # gradients
    #

    for (index, outputNeuron) in enumerate(outputLayer):
      outputNeuron.calculateGradient_output(targetValues[index])

    # from last to first hidden layer(s)
    for ii in range(len(self._layers) - 2, 0, -1):
      for neuron in self._layers[ii]:
        neuron.calculateGradient_hidden()

    #
    # update inputs
    #

    for currLayer in self._layers:
      for neuron in currLayer:
        neuron.updateInputWeights()

  def debug(self):
    for (index1, layer) in enumerate(self._layers):
      print(f'layer {index1}')
      print(f'  neurons: {len(layer)}')
      for neuron in layer:
        print(f'            _outputValue: {neuron._outputValue}')
        print(f'    _gradientOutputValue: {neuron._gradientOutputValue}')
        print(f'          _inputConnections: {len(neuron._inputConnections)}')
        print(f'         _outputConnections: {len(neuron._outputConnections)}')



















class TrainingData:

  input: List[float]
  output: List[float]

  def __init__(self, input: List[float], output: List[float]):
    self.input = input
    self.output = output





trainingData: List[TrainingData] = [
  TrainingData([0,0], [0]),
  TrainingData([1,0], [1]),
  TrainingData([0,1], [1]),
  TrainingData([1,1], [0]),
]

neuralNetwork = NeuralNetwork([2,3,1])

training_progress: List[float] = []
training_error: List[float] = []

trainingPass = 0

while trainingPass < 100000:

  currData = trainingData[trainingPass % len(trainingData)]

  trainingPass += 1

  results = neuralNetwork.feedForward(currData.input)
  neuralNetwork.backProp(currData.output)

  # neuralNetwork.debug()

  print(f"{trainingPass} error: {neuralNetwork._error} recentAvgError: {neuralNetwork._recentAvgError}, input {currData.input}, output {results}")

  training_progress.append(neuralNetwork._recentAvgError)
  training_error.append(neuralNetwork._error)

  if (
    # we need enough sample data for the average, here 100 samples
    trainingPass > 100 and
    # is the average error acceptable?
    neuralNetwork._recentAvgError < 0.05
  ):
    print("\naverage error acceptable -> break")
    break

print("DONE")

print("PRINT FINAL RESULT")

def reduce_to_two_decimal(value: float):
  return float("{:.2f}".format(value))

for currData in trainingData:

  inputs = list(map(reduce_to_two_decimal, currData.input))
  expected = list(map(reduce_to_two_decimal, currData.output))

  results = neuralNetwork.feedForward(currData.input)
  results = list(map(reduce_to_two_decimal, results))

  print(f">  inputs: {inputs}, expected: {expected}, result: {results}")





import matplotlib.pyplot as plt

epochs = range(len(training_progress))
plt.figure()
plt.plot(epochs, training_error, 'r', label='last_err')
plt.plot(epochs, training_progress, 'b', label='avg_err')
plt.title('errors')
plt.legend()

# will pause until closed by the user
plt.show(block=True)



