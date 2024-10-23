





const _clampValue = (val: number, minVal: number, maxVal: number) =>
  Math.min(Math.max(val, minVal), maxVal);

const activations = {
  tanh: {
    activation: (x: number) => {
      return Math.tanh(x); // [-1.0..1.0]
    },
    derivative: (x: number) => {
      return 1.0 - Math.tanh(x * x);
      // return 1.0 - x * x; // faster, small loss of accuracy
    }
  },
  relu: {
    activation: (x: number) => {
      // return Math.max(0, x);
      return _clampValue(Math.max(0, x),   -1.0, 1.0); // [-1..1]
    },
    derivative: (x: number) => {
      return x < 0 ? 0 : 1;
    }
  },
  leakyRelu: {
    activation: (x: number) => {
      // return Math.max(x * 0.1, x); // [-0.1..x]
      return _clampValue(Math.max(x * 0.1, x),   -1.0, 1.0); // [-1..1]
    },
    derivative: (x: number) => {
      return x < 0 ? -0.1 : 1;
    }
  },
}












interface Synapse {
  input: Neuron;
  output: Neuron;
  weight: number;
}


class Neuron {

  public _inputSynapses: Synapse[] = [];
  public _outputSynapses: Synapse[] = [];
  public _outputValue: number = 0;
  private _gradientOutputValue: number = 0;
  public _isBias: boolean;

  constructor(isBias: boolean) {
    this._isBias = isBias;

    if (this._isBias) {
      this._outputValue = 1;
    }
  }

  feedForward() {

    if (this._isBias) {
      this._outputValue = 1;
      return;
    }

    if (this._inputSynapses.length === 0) {
      return;
    }

    let sum = 0;
    this._inputSynapses.forEach((inputSynapse) => {
      sum += inputSynapse.input._outputValue * inputSynapse.weight;
    });
    this._outputValue = activations.tanh.activation(sum);
  }

  calculateGradient_output(targetValue: number) {
    const delta = targetValue - this._outputValue;
    this._gradientOutputValue = delta * activations.tanh.derivative(this._outputValue);
  }

  calculateGradient_hidden() {
    let sum = 0;
    this._outputSynapses.forEach((outputSynapse) => {
      if (outputSynapse.output._isBias) {
        return;
      }
      sum += outputSynapse.weight * outputSynapse.output._gradientOutputValue;
    });

    this._gradientOutputValue = sum * activations.tanh.derivative(this._outputValue);
  }

  updateInputWeights() {
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    const k_learningRate = 0.15;

    this._inputSynapses.forEach((inputSynapse) => {
      inputSynapse.weight += k_learningRate * inputSynapse.input._outputValue * this._gradientOutputValue;
    });
  }

}














const k_minSamples = 200;


class NeuralNetwork {

  public _layers: Neuron[][] = [];
  public _error: number = 0;
  public _recentAvgError: number = 0;

  constructor(topology: number[]) {

    if (topology.length < 2) {
      throw new Error("invalid amount of layers");
    }
    topology.forEach((totalNeurons, index) => {
      if (totalNeurons === 0) {
        throw new Error(`invalid amount of layer neurons in layer ${index}`);
      }
    });

    this._buildTheLayers(topology);
    this._connectTheLayers();
  }

  private _buildTheLayers(topology: number[]) {

    for (let ii = 0; ii < topology.length; ++ii) {

      const newLayer: Neuron[] = [];

      for (let jj = 0; jj < topology[ii]; ++jj) {
        newLayer.push(new Neuron(false));
      }

      const isLastLayer = (ii + 1 === topology.length);
      if (!isLastLayer) {
        newLayer.push(new Neuron(true));
      }

      this._layers.push(newLayer);
    }
  }

  private _connectTheLayers() {
    for (let ii = 0; ii + 1 < this._layers.length; ++ii) {

      const prevLayer = this._layers[ii];
      const nextLayer = this._layers[ii + 1];

      for (const prevNeuron of prevLayer) {
        for (const nextNeuron of nextLayer) {

          if (nextNeuron._isBias) {
            continue;
          }

          const newSynapse: Synapse = {
            input: prevNeuron,
            output: nextNeuron,

            weight: Math.random(), // [0..1]

            // // relu friendly weights
            // weight: 0.2 + Math.random() * 0.8, // [0.2..1]
          };

          prevNeuron._outputSynapses.push(newSynapse);
          nextNeuron._inputSynapses.push(newSynapse);
        }
      }
    }
  }

  feedForward(inputValues: number[]): number[] {

    if (inputValues.length !== this._layers[0].length - 1) {
      throw new Error("invalid amount of input values");
    }

    // set the input values
    inputValues.forEach((value, index) => {
      this._layers[0][index]._outputValue = value;
    });

    this._layers.forEach((layer) => {
      layer.forEach(neuron => neuron.feedForward());
    });

    const outputLayer = this._layers[this._layers.length - 1];
    return outputLayer.map(neuron => neuron._outputValue);
  }

  backProp(targetValues: number[]) {

    const outputLayer = this._layers[this._layers.length - 1];

    if (targetValues.length !== outputLayer.length) {
      throw new Error("invalid amount of input values");
    }

    //
    // error
    //

    this._error = 0;

    outputLayer.forEach((outputNeuron, index) => {
      const delta = targetValues[index] - outputNeuron._outputValue;
      this._error += delta * delta;
    });
    this._error /= outputLayer.length; // get average error squared
    this._error = Math.sqrt(this._error); // RMS

    this._recentAvgError = (this._recentAvgError * k_minSamples + this._error) / (k_minSamples + 1.0);

    //
    // gradients
    //

    outputLayer.forEach((neuron, index) => {
      neuron.calculateGradient_output(targetValues[index]);
    });

    // from last to first hidden layer(s)
    for (let ii = this._layers.length - 2; ii > 0; --ii) {
      this._layers[ii].forEach(neuron => neuron.calculateGradient_hidden());
    }

    //
    // update inputs
    //

    this._layers.forEach((currLayer) => {
      currLayer.forEach(neuron => neuron.updateInputWeights());
    });

  }


}

















const showVector = (vector: number[]) => {
  return `[${vector.map(value => value.toFixed(3)).join(', ')}]`;
}

// learn the XOR logic gate (the hardest of all)
const trainingData: { in: number[], out: number[] }[] = [
  { in: [0,0], out: [0] },
  { in: [1,0], out: [1] },
  { in: [0,1], out: [1] },
  { in: [1,1], out: [0] },
];


const neuralNetwork = new NeuralNetwork([2,3,1]);

let trainingPass = 0;

while (trainingPass < 100000) {

  const currData = trainingData[trainingPass % trainingData.length];

  trainingPass++;

  const results = neuralNetwork.feedForward(currData.in);
  neuralNetwork.backProp(currData.out);

  console.log(
    trainingPass,
    `error: ${neuralNetwork._error.toFixed(3)}`,
    `recentAvgError: ${neuralNetwork._recentAvgError.toFixed(3)}`,
    `input: ${showVector(currData.in)}`,
    `output: ${showVector(results)}`);


  if (
    // we need enough sample data for the average, here 100 samples
    trainingPass > 100 &&
    // is the average error acceptable?
    neuralNetwork._recentAvgError < 0.05
  ) {
    console.log("\naverage error acceptable -> break");
    break;
  }
}

// console.log("neuralNetwork._layers", neuralNetwork._layers);

console.log("DONE");

console.log("PRINT FINAL RESULT")

for (const currData of trainingData) {

  const results = neuralNetwork.feedForward(currData.in);

  console.log(`input: ${currData.in.join(", ")}, expected: ${currData.out.join(", ")}, result: ${results.join(", ")}`);
}

