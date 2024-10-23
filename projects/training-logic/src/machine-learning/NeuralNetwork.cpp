
#include "NeuralNetwork.hpp"

#include <cassert>


double NeuralNetwork::k_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over

NeuralNetwork::NeuralNetwork(const std::vector<uint32_t>& arr_topology)
    :   m_error(0.0),
        m_recentAvgError(0.0)
{
    assert( !arr_topology.empty() ); // no empty topology

    RandomNumberGenerator rng;
    rng.ensureRandomSeed();

    for (uint32_t ii = 0; ii < arr_topology.size(); ++ii)
    {
        uint32_t totalNeurons = arr_topology[ii];

        assert( totalNeurons > 0 ); // no empty layer

        totalNeurons += 1; // add a bias neuron

        m_arr_layers.push_back(t_Layer());

        t_Layer& arr_new_layer = m_arr_layers.back();
        arr_new_layer.reserve(totalNeurons); // pre-allocate

        const bool is_last_layer = ((ii + 1) == arr_topology.size());

        // 0 output if on the last layer
        const uint32_t numOutputs = ((is_last_layer) ? (0) : (arr_topology[ii + 1]));

        // We have a new layer, now fill it with neurons (+ the extra bias neuron)
        for (uint32_t jj = 0; jj < totalNeurons; ++jj)
        {
            arr_new_layer.emplace_back(numOutputs, jj, rng);
        }

        // Force the bias node's output to 1.0
        // -> it was the last neuron pushed in this layer
        Neuron& bias_neuron = arr_new_layer.back();
        bias_neuron.setOutputVal(1.0);
    }
}

void NeuralNetwork::feedForward(const t_vals &inputVals)
{
    assert( inputVals.size() == (m_arr_layers[0].size() - 1) ); // exclude bias neuron

    // Assign (latch) the input values into the input neurons
    for (uint32_t ii = 0; ii < inputVals.size(); ++ii)
    {
        m_arr_layers[0][ii].setOutputVal(inputVals[ii]);
    }

    // forward propagate
    // -> start at 1 -> exclude input layer
    for (uint32_t ii = 1; ii < m_arr_layers.size(); ++ii)
    {
        t_Layer& prevLayer = m_arr_layers[ii - 1];
        t_Layer& currLayer = m_arr_layers[ii];

        const uint32_t num_neuron = uint32_t(currLayer.size()) - 1; // exclude bias neuron
        for (uint32_t jj = 0; jj < num_neuron; ++jj)
        {
            currLayer[jj].feedForward(prevLayer);
        }
    }
}

void NeuralNetwork::backProp(const t_vals &arr_targetVals)
{
    //
    // error

    // Calculate overall net error (RMS of output neuron errors)

    t_Layer &outputLayer = m_arr_layers.back();
    m_error = 0.0;

    // exclude bias neuron
    const uint32_t numOutputs = uint32_t(outputLayer.size()) - 1;

    for (uint32_t ii = 0; ii < numOutputs; ++ii)
    {
        const double delta = arr_targetVals[ii] - outputLayer[ii].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= (outputLayer.size() - 1); // get average error squared
    m_error = std::sqrt(m_error); // RMS

    // Implement a recent average measurement

    m_recentAvgError =
            (m_recentAvgError * k_recentAvgSmoothingFactor + m_error)
            / (k_recentAvgSmoothingFactor + 1.0);

    // error
    //


    //
    // Gradients

    // Calculate output layer gradients
    for (uint32_t ii = 0; ii < numOutputs; ++ii)
    {
        outputLayer[ii].calcOutputGradients(arr_targetVals[ii]);
    }

    // Calculate hidden layer gradients

    // apply -2 to remove the input and output layer
    // possibly 0
    const uint32_t numHidden = uint32_t(m_arr_layers.size()) - 2;

    for (uint32_t ii = numHidden; ii > 0; --ii)
    {
        t_Layer &arr_currLayer = m_arr_layers[ii];
        const t_Layer &arr_nextLayer = m_arr_layers[ii + 1];

        for (auto& currNeuron : arr_currLayer)
        {
            currNeuron.calcHiddenGradients(arr_nextLayer);
        }
    }

    // Gradients
    //


    // For all layers from outputs to first hidden layer,
    // update connection weights

    const uint32_t numInputAndHidden = numHidden + 1;

    for (uint32_t ii = numInputAndHidden; ii > 0; --ii)
    {
        t_Layer &arr_prevLayer = m_arr_layers[ii - 1];
        const t_Layer &arr_currLayer = m_arr_layers[ii];

        // exclude last neuron (bias neuron)
        const uint32_t sizeLayer = uint32_t(arr_currLayer.size()) - 1;

        for (uint32_t jj = 0; jj < sizeLayer; ++jj)
        {
            arr_currLayer[jj].updateInputWeights(arr_prevLayer);
        }
    }
}

void NeuralNetwork::getResults(t_vals &arr_resultVals) const
{
    arr_resultVals.clear();

    const t_Layer& outputLayer = m_arr_layers.back();

    // exclude last neuron (bias neuron)
    const uint32_t total_neuron = uint32_t(outputLayer.size()) - 1;

    arr_resultVals.reserve(total_neuron); // pre-allocate

    for (uint32_t ii = 0; ii < total_neuron; ++ii)
    {
        arr_resultVals.push_back(outputLayer[ii].getOutputVal());
    }
}
