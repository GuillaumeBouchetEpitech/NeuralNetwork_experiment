

'use strict'





//
//
// TrainingData

// var sample = require('../../src/trainsample/out_xor.txt');

var fs = require('fs');

function TrainingData (str_file) {

	var sample = fs.readFileSync('../src/trainsample/out_xor.txt', 'utf8');

	sample = sample.trim();
	this._arr_lines = sample.split('\n');
};

TrainingData.prototype.isEof = function() {
	return (this._arr_lines.length == 0)
}

TrainingData.prototype._extract_from_data = function (str_pattern, arr_output) {

	arr_output.length = 0; // clear array

	var str_value = this._arr_lines.shift().trim();
	var arr_elements = str_value.split(' ');

	if (arr_elements.length > 0 &&
		arr_elements[0] == str_pattern)
	{
		for (var i = 1; i < arr_elements.length; ++i)
			arr_output.push( parseFloat(arr_elements[i]) );
	}

	return arr_output.length;
}

TrainingData.prototype.getTopology = function(arr_topology) {
	return this._extract_from_data('topology:', arr_topology);
}

TrainingData.prototype.getNextInputs = function(arr_inputVals) {
	return this._extract_from_data('in:', arr_inputVals);
}

TrainingData.prototype.getTargetOutputs = function(arr_targetOutputVals) {
	return this._extract_from_data('out:', arr_targetOutputVals);
}

// TrainingData
//
//



















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

// NEURON
//
//















//
//
// NET

function Net (arr_topology) { // ctor/dtor

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

Net.prototype._k_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over

// public method(s)

Net.prototype.feedForward = function(arr_inputVals)
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

Net.prototype.backProp = function(arr_targetVals)
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

Net.prototype.getResults = function(arr_resultVals)
{
    arr_resultVals.length = 0;

    var arr_outputLayer = this._m_arr_layers[ this._m_arr_layers.length - 1 ];

    // exclude last neuron (bias neuron)
    var total_neuron = (arr_outputLayer.length - 1);

    for (var n = 0; n < total_neuron; ++n)
        arr_resultVals.push(arr_outputLayer[n].getOutputVal());
}

Net.prototype.getError = function() { return this._m_error; }
Net.prototype.getRecentAverageError = function() { return this._m_recentAvgError; }


// NET
//
//















//
//
// MAIN

function showVectorVals(prefix, arr_values)
{
    var str_msg = prefix + " ";
    for (var i = 0; i < arr_values.length; ++i)
        str_msg += arr_values[i] + " ";

    console.log(str_msg);
}

{
	var trainData = new TrainingData();


	var arr_topology = [];
	trainData.getTopology(arr_topology);
	console.log('arr_topology=', arr_topology);

	var myNet = new Net(arr_topology);


	var arr_inputVals = [];
	var arr_resultVals = [];
	var arr_targetVals = [];
    var trainingPass = 0;


	// while (!trainData.isEof() && trainingPass < 5)
	while (!trainData.isEof())
	{
		// trainData.getNextInputs(arr_input);
		// console.log('arr_input=', arr_input);

		// trainData.getTargetOutputs(arr_output);
		// console.log('arr_output=', arr_output);


        ++trainingPass;
        console.log();
        console.log("Pass " + trainingPass);

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(arr_inputVals) != arr_topology[0])
            break;

        showVectorVals("Inputs:", arr_inputVals);
        myNet.feedForward(arr_inputVals);

        // Collect the net's actual output results:
        myNet.getResults(arr_resultVals);
        showVectorVals("Outputs:", arr_resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(arr_targetVals);
        showVectorVals("Targets:", arr_targetVals);
        if (arr_targetVals.length != arr_topology[arr_topology.length - 1])
			throw new Error('(arr_targetVals.length != arr_topology.back())');


        myNet.backProp(arr_targetVals);

        // Report how well the training is working, average over recent samples:
        console.log("Net current error: " + myNet.getError());
        console.log("Net recent average error: " + myNet.getRecentAverageError());

        if (trainingPass > 100 && myNet.getRecentAverageError() < 0.05)
        {
            console.log("average error acceptable -> break");
            break;
        }
	}

	console.log();
	console.log("Done");
	console.log();

    if (arr_topology.length < 2 ||
        arr_topology[0] != 2)
    {
		console.log("Unexpected topology, no test");
    }
    else
    {
		console.log("TEST");
		console.log();

        var dblarr_test = [ [0,0], [0,1], [1,0], [1,1] ];

        for (var i = 0; i < 4; ++i)
        {
            arr_inputVals.length = 0;
            arr_inputVals.push(dblarr_test[i][0]);
            arr_inputVals.push(dblarr_test[i][1]);

            myNet.feedForward(arr_inputVals);
            myNet.getResults(arr_resultVals);

            showVectorVals("Inputs:", arr_inputVals);
            showVectorVals("Outputs:", arr_resultVals);

			console.log();
        }

		console.log("/TEST");
    }
}

// MAIN
//
//

