// neural-net-tutorial.cpp
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529


#include "./TrainingData.hpp"

#include "./utilities/RandomNumberGenerator.hpp"

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <cmath>
// #include <fstream>
#include <sstream>








struct SynapseConnection
{
    double m_weight;
    double m_deltaWeight = 0;

    SynapseConnection(double weight)
        : m_weight(weight)
    {}
};
using SynapseConnections = std::vector<SynapseConnection>;


class Neuron; // forward declaration

using t_Layer = std::vector<Neuron>;









//
//
// NEURON

class Neuron
{
private: // static attr
    static double k_eta;   // [0.0..1.0] overall net training rate
    static double k_alpha; // [0.0..n] multiplier of last weight change (momentum)

private: // static inline method(s)
    static inline double transferFunction(double x)
    {
        return tanh(x); // tanh - output range [-1.0..1.0]
    }

    static inline double transferFunctionDerivative(double x)
    {
        return (1.0 - x * x); // tanh derivative
    }

    // static inline double randomWeight(void)
    // {
    //     return rand() / double(RAND_MAX);
    // }

private: // attr
    double          m_outputVal;
    SynapseConnections   m_outputWeights;
    uint32_t        m_layer_index;
    double          m_gradient; // used by the backpropagation

public: // ctor/dtor
    Neuron(uint32_t numOutputs, uint32_t myIndex, RandomNumberGenerator& rng);

private: // private method(s)
    double  sumDOW(const t_Layer &arr_nextLayer) const;

public: // public method(s)
    void    feedForward(const t_Layer& arr_prevLayer);
    void    calcOutputGradients(double targetVal);
    void    calcHiddenGradients(const t_Layer& arr_nextLayer);
    void    updateInputWeights(t_Layer& arr_prevLayer);

public: // getter/setter
    inline void     setOutputVal(double val) { m_outputVal = val; }
    inline double   getOutputVal(void) const { return m_outputVal; }
};

double Neuron::k_eta = 0.15;  // overall net learning rate, [0.0..1.0]
double Neuron::k_alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(uint32_t numOutputs, uint32_t myIndex, RandomNumberGenerator& rng)
    : m_layer_index(myIndex)
{
    if (numOutputs == 0) {
        return;
    }

    m_outputWeights.reserve(numOutputs);
    for (uint32_t ii = 0; ii < numOutputs; ++ii)
    {
        m_outputWeights.emplace_back(rng.getRangedValue(0.0f, 1.0f));
    }
}

double Neuron::sumDOW(const t_Layer &arr_nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    const uint32_t num_neuron = uint32_t(arr_nextLayer.size()) - 1; // exclude bias neuron

    for (uint32_t ii = 0; ii < num_neuron; ++ii)
    {
        sum += m_outputWeights[ii].m_weight * arr_nextLayer[ii].m_gradient;
    }

    return sum;
}

void Neuron::feedForward(const t_Layer &arr_prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (uint32_t ii = 0; ii < arr_prevLayer.size(); ++ii)
    {
        sum += arr_prevLayer[ii].getOutputVal() * arr_prevLayer[ii].m_outputWeights[m_layer_index].m_weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const t_Layer &arr_nextLayer)
{
    double dow = sumDOW(arr_nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(t_Layer &arr_prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (Neuron &neuron : arr_prevLayer)
    {
        auto& outSynapse = neuron.m_outputWeights[m_layer_index];

        const double oldDeltaWeight = outSynapse.m_deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                k_eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + k_alpha
                * oldDeltaWeight
                ;

        outSynapse.m_deltaWeight = newDeltaWeight;
        outSynapse.m_weight += newDeltaWeight;
    }
}

// NEURON
//
//









//
//
// NET

class Net
{
private: // attr
    std::vector<t_Layer> m_arr_layers; // m_layers[layerNum][neuronNum]

private: // attr -> error
    double m_error;
    double m_recentAvgError;
private: // static attr -> error
    static double k_recentAvgSmoothingFactor;

public: // ctor/dtor
    Net(const std::vector<uint32_t> &arr_topology);

public: // public method(s)
    void feedForward(const t_vals &inputVals);
    void backProp(const t_vals &targetVals);
    void getResults(t_vals &resultVals) const;

public: // public method(s) -> error
    inline double getError(void) const { return m_error; }
    inline double getRecentAverageError(void) const { return m_recentAvgError; }
};

double Net::k_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const std::vector<uint32_t>& arr_topology)
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

void Net::feedForward(const t_vals &inputVals)
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
            currLayer[jj].feedForward(prevLayer);
    }
}

void Net::backProp(const t_vals &arr_targetVals)
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
    m_error = sqrt(m_error); // RMS

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

    const uint32_t numHidden = uint32_t(m_arr_layers.size()) - 2;

    for (uint32_t ii = numHidden; ii > 0; --ii)
    {
        t_Layer &arr_currLayer = m_arr_layers[ii];
        t_Layer &arr_nextLayer = m_arr_layers[ii + 1];

        for (uint32_t jj = 0; jj < arr_currLayer.size(); ++jj)
        {
            arr_currLayer[jj].calcHiddenGradients(arr_nextLayer);
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
        t_Layer &arr_currLayer = m_arr_layers[ii];

        // exclude last neuron (bias neuron)
        const uint32_t sizeLayer = uint32_t(arr_currLayer.size()) - 1;

        for (uint32_t jj = 0; jj < sizeLayer; ++jj)
        {
            arr_currLayer[jj].updateInputWeights(arr_prevLayer);
        }
    }
}

void Net::getResults(t_vals &arr_resultVals) const
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

// NET
//
//









//
//
// MAIN

void showVectorVals(const std::string& prefix, const t_vals &arr_values)
{
    std::cout << prefix << " ";
    for (uint32_t i = 0; i < arr_values.size(); ++i)
        std::cout << std::fixed << std::setprecision(2) << arr_values[i] << " ";

    std::cout << std::endl;
}

void printUsageAndExit(const char* programName)
{
	std::cerr << "Usage 1: " << programName << " and" << std::endl;
	std::cerr << "Usage 2: " << programName << " or" << std::endl;
	std::cerr << "Usage 3: " << programName << " no" << std::endl;
	std::cerr << "Usage 4: " << programName << " xor" << std::endl;
	exit(EXIT_FAILURE);
}


int main(int argc, char** argv)
{
    if (argc != 2) {
		printUsageAndExit(argv[0]);
	}

    std::string trainingFilename = argv[1];

    // TrainingData trainData("trainsample/out_xor.txt");
    // TrainingData trainData("trainsample/out_and.txt");
    // TrainingData trainData("trainsample/out_or.txt");
    // TrainingData trainData("trainsample/out_no.txt");
    TrainingData trainData(trainingFilename);

    // e.g., { 3, 2, 1 }
    std::vector<uint32_t> arr_topology;
    trainData.getTopology(arr_topology);

    Net myNet(arr_topology);

    t_vals arr_inputVals, arr_targetVals, arr_resultVals;
    int trainingPass = 0;

    while (!trainData.isEof())
    {
        ++trainingPass;
        std::cout << std::endl << "Pass " << trainingPass << std::endl;

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
        assert(arr_targetVals.size() == arr_topology.back());

        myNet.backProp(arr_targetVals);

        // Report how well the training is working, average over recent samples:
        std::cout << "Net current error: " << myNet.getError() << std::endl;
        std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;

        if (trainingPass > 100 && myNet.getRecentAverageError() < 0.05)
        {
            std::cout << std::endl << "average error acceptable -> break" << std::endl;
            break;
        }
    }

    std::cout << std::endl << "Done" << std::endl;

    if (arr_topology.size() < 2 ||
        arr_topology.front() != 2)
    {
        std::cout << "Unexpected topology, no test" << std::endl;
    }
    else
    {
        std::cout << "TEST" << std::endl;
        std::cout << std::endl;

        uint32_t dblarr_test[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };

        for (uint32_t i = 0; i < 4; ++i)
        {
            arr_inputVals.clear();
            arr_inputVals.push_back(dblarr_test[i][0]);
            arr_inputVals.push_back(dblarr_test[i][1]);

            myNet.feedForward(arr_inputVals);
            myNet.getResults(arr_resultVals);

            showVectorVals("Inputs:", arr_inputVals);
            showVectorVals("Outputs:", arr_resultVals);

            std::cout << std::endl;
        }

        std::cout << "/TEST" << std::endl;
    }
}

// MAIN
//
//



