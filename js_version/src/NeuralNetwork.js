
'use strict'


var Neuron = require("./Neuron.js");


//
//
// NET

function NeuralNetwork (arr_topology) { // ctor/dtor

	// attr -> error

    this._m_error = 0.0;
    this._m_recentAvgError = 0.0;

	// attr

	this._m_arr_layers = []; // m_layers[layerNum][neuronNum]

	if (arr_topology.length == 0)
		throw new Error('(arr_topology.length == 0)');

    for (var i = 0; i < arr_topology.length; ++i)
    {
        var num_neuron = arr_topology[i];

		if (num_neuron <= 0) // no empty layer
			throw new Error('(num_neuron <= 0)');

        var arr_new_layer = [];

        //
        //

        var is_last_layer = (i == (arr_topology.length - 1));

        // 0 output if on the last layer
        var numOutputs = ((is_last_layer) ? (0) : (arr_topology[i + 1]));

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (var j = 0; j < (num_neuron + 1); ++j) // add a bias neuron
            arr_new_layer.push( new Neuron(numOutputs, j) );

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        var bias_neuron = arr_new_layer[ arr_new_layer.length - 1 ];
        bias_neuron.setOutputVal(1.0);
        // bias_neuron._m_outputVal = 1.0;

        //
        //

        this._m_arr_layers.push(arr_new_layer);
    }
};

// static attr -> error

NeuralNetwork.prototype._k_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over

// public method(s)

NeuralNetwork.prototype.feedForward = function(arr_inputVals)
{
	if (arr_inputVals.length != (this._m_arr_layers[0].length - 1)) // exclude bias neuron
		throw new Error('(arr_inputVals.length != (m_arr_layers[0].length - 1))');

    // Assign (latch) the input values into the input neurons
    for (var i = 0; i < arr_inputVals.length; ++i)
        this._m_arr_layers[0][i].setOutputVal(arr_inputVals[i]);
        // this._m_arr_layers[0][i]._m_outputVal = arr_inputVals[i];


    // console.log(this._m_arr_layers[0]);

    // forward propagate
    for (var i = 1; i < this._m_arr_layers.length; ++i) // exclude input layer
    {
        var arr_prevLayer = this._m_arr_layers[i - 1];
        var arr_currLayer = this._m_arr_layers[i];

        var num_neuron = (arr_currLayer.length - 1); // exclude bias neuron
        for (var n = 0; n < num_neuron; ++n)
            arr_currLayer[n].feedForward(arr_prevLayer);
    }
}

NeuralNetwork.prototype.backProp = function(arr_targetVals)
{
    //
    // error

    // Calculate overall net error (RMS of output neuron errors)

    var arr_outputLayer = this._m_arr_layers[this._m_arr_layers.length - 1];
    this._m_error = 0.0;

    for (var n = 0; n < arr_outputLayer.length - 1; ++n)
    {
        var delta = arr_targetVals[n] - arr_outputLayer[n].getOutputVal();
        this._m_error += delta * delta;
    }
    this._m_error /= (arr_outputLayer.length - 1); // get average error squared
    this._m_error = Math.sqrt(this._m_error); // RMS

    // Implement a recent average measurement

    this._m_recentAvgError =
            (this._m_recentAvgError * this._k_recentAvgSmoothingFactor + this._m_error)
            / (this._k_recentAvgSmoothingFactor + 1.0);

    // error
    //


    //
    // Gradients

    // Calculate output layer gradients

    for (var n = 0; n < (arr_outputLayer.length - 1); ++n)
        arr_outputLayer[n].calcOutputGradients(arr_targetVals[n]);

    // Calculate hidden layer gradients

    for (var i = (this._m_arr_layers.length - 2); i > 0; --i)
    {
        var arr_hiddenLayer = this._m_arr_layers[i];
        var arr_nextLayer = this._m_arr_layers[i + 1];

        for (var n = 0; n < arr_hiddenLayer.length; ++n)
            arr_hiddenLayer[n].calcHiddenGradients(arr_nextLayer);
    }

    // Gradients
    //


    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (var i = (this._m_arr_layers.length - 1); i > 0; --i)
    {
        var arr_currLayer = this._m_arr_layers[i];
        var arr_prevLayer = this._m_arr_layers[i - 1];

        for (var n = 0; n < (arr_currLayer.length - 1); ++n) // exclude bias
            arr_currLayer[n].updateInputWeights(arr_prevLayer);
    }
}

NeuralNetwork.prototype.getResults = function(arr_resultVals)
{
    arr_resultVals.length = 0;

    var arr_outputLayer = this._m_arr_layers[ this._m_arr_layers.length - 1 ];

    // exclude last neuron (bias neuron)
    var total_neuron = (arr_outputLayer.length - 1);

    for (var n = 0; n < total_neuron; ++n)
        arr_resultVals.push(arr_outputLayer[n].getOutputVal());
}

NeuralNetwork.prototype.getError = function() { return this._m_error; }
NeuralNetwork.prototype.getRecentAverageError = function() { return this._m_recentAvgError; }


NeuralNetwork.prototype.getWeights = function()
{
    var out_arr_weight = [];
    for (var i = 0; i < this._m_arr_layers.length - 1; ++i)
    {
        var curr_layer = this._m_arr_layers[i];

        var num_neurons = (curr_layer.length - 1); // exluding bias neuron

        for (var j = 0; j < num_neurons; ++j)
        {
            var curr_neuron = curr_layer[j];

            var arr_weight = curr_neuron.getOutputWeight();

            for (var k = 0; k < arr_weight.length; ++k)
                out_arr_weight.push( arr_weight[k].weight );
        }
    }

    return out_arr_weight;
}

// NET
//
//


module.exports = NeuralNetwork;

