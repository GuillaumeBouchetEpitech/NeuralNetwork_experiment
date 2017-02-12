
'use strict'

//
//
// NEURON

// ctor/dtor

function Neuron (numOutputs, myIndex) {

	this._m_layer_index = myIndex;
	this._m_arr_outputWeights = [];

    for (var i = 0; i < numOutputs; ++i)
        this._m_arr_outputWeights.push({
        	  weight: this._randomWeight()
        	, deltaWeight: 0
        });

	// other attr

    this._m_outputVal = 0.0;
    this._m_gradient = 0.0; // used by the backpropagation
};

// static attr

Neuron.prototype._k_eta = 0.15
Neuron.prototype._k_alpha = 0.5

// static inline method(s)

Neuron.prototype._transferFunction = function(x)
{
    return Math.tanh(x); // tanh - output range [-1.0..1.0]
}

Neuron.prototype._transferFunctionDerivative = function(x)
{
    return (1.0 - x * x); // tanh derivative
}

Neuron.prototype._randomWeight = function()
{
    return Math.random();
}

// private method(s)

Neuron.prototype._sumDOW = function(arr_nextLayer)
{
    var sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    var num_neuron = (arr_nextLayer.length - 1); // exclude bias neuron

    for (var n = 0; n < num_neuron; ++n)
        sum += this._m_arr_outputWeights[n].weight * arr_nextLayer[n]._m_gradient;

    return sum;
}

// public method(s)

Neuron.prototype.feedForward = function(arr_prevLayer)
{
    var sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (var n = 0; n < arr_prevLayer.length; ++n)
        sum += arr_prevLayer[n].getOutputVal() * arr_prevLayer[n]._m_arr_outputWeights[this._m_layer_index].weight;

    this._m_outputVal = this._transferFunction(sum);
}

Neuron.prototype.calcOutputGradients = function(targetVal)
{
    var delta = targetVal - this._m_outputVal;
    this._m_gradient = delta * this._transferFunctionDerivative(this._m_outputVal);
}

Neuron.prototype.calcHiddenGradients = function(arr_nextLayer)
{
    var dow = this._sumDOW(arr_nextLayer);
    this._m_gradient = dow * this._transferFunctionDerivative(this._m_outputVal);
}

Neuron.prototype.updateInputWeights = function(arr_prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (var n = 0; n < arr_prevLayer.length; ++n)
    {
        var neuron = arr_prevLayer[n];
        var oldDeltaWeight = neuron._m_arr_outputWeights[this._m_layer_index].deltaWeight;

        var newDeltaWeight =
            // Individual input, magnified by the gradient and train rate:
            this._k_eta
            * neuron.getOutputVal()
            * this._m_gradient
            // Also add momentum = a fraction of the previous delta weight;
            + this._k_alpha
            * oldDeltaWeight
            ;

        neuron._m_arr_outputWeights[this._m_layer_index].deltaWeight = newDeltaWeight;
        neuron._m_arr_outputWeights[this._m_layer_index].weight += newDeltaWeight;
    }
}

// getter/setter

Neuron.prototype.setOutputVal = function(val) { this._m_outputVal = val; }
Neuron.prototype.getOutputVal = function() { return this._m_outputVal; }

Neuron.prototype.getOutputWeight = function() { return this._m_arr_outputWeights; }

// NEURON
//
//


module.exports = Neuron;
