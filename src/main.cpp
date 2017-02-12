// neural-net-tutorial.cpp
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529


#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

// using namespace std;







//
//
// TrainingData

// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

typedef std::vector<double> t_vals;

class TrainingData
{
private: // attr
    std::ifstream   m_file_trainingData;

public: // ctor/dtor
    TrainingData(const std::string& filename);

public: // getter/setter
    inline bool isEof(void) const { return m_file_trainingData.eof(); }

public: // public method(s)
    void getTopology(std::vector<unsigned> &arr_topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(t_vals &arr_inputVals);
    unsigned getTargetOutputs(t_vals &arr_targetOutputVals);
};

TrainingData::TrainingData(const std::string& filename)
{
    m_file_trainingData.open(filename.c_str());
}

void TrainingData::getTopology(std::vector<unsigned> &arr_topology)
{
    std::string line;
    std::string label;

    std::getline(m_file_trainingData, line);
    std::stringstream ss(line);
    ss >> label;

    if (this->isEof() || label != "topology:")
        abort();

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        arr_topology.push_back(n);
    }
}

unsigned TrainingData::getNextInputs(t_vals &arr_inputVals)
{
    arr_inputVals.clear();

    std::string str_line;
    std::getline(m_file_trainingData, str_line);
    std::stringstream sstr(str_line);

    std::string str_label;
    sstr >> str_label;
    if (str_label == "in:")
    {
        double dbl_oneValue;

        while (sstr >> dbl_oneValue)
            arr_inputVals.push_back(dbl_oneValue);
    }

    return arr_inputVals.size();
}

unsigned TrainingData::getTargetOutputs(t_vals &arr_targetOutputVals)
{
    arr_targetOutputVals.clear();

    std::string str_line;
    std::getline(m_file_trainingData, str_line);
    std::stringstream sstr(str_line);

    std::string str_label;
    sstr >> str_label;
    if (str_label == "out:")
    {
        double dbl_oneValue;

        while (sstr >> dbl_oneValue)
            arr_targetOutputVals.push_back(dbl_oneValue);
    }

    return arr_targetOutputVals.size();
}

// TrainingData
//
//








struct t_Connection
{
    double m_weight;
    double m_deltaWeight;
};
typedef std::vector<t_Connection> t_Connections;


class Neuron;

typedef std::vector<Neuron> t_Layer;









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

    static inline double randomWeight(void)
    {
        return rand() / double(RAND_MAX);
    }

private: // attr
    double          m_outputVal;
    t_Connections   m_arr_outputWeights;
    unsigned        m_layer_index;
    double          m_gradient; // used by the backpropagation

public: // ctor/dtor
    Neuron(unsigned numOutputs, unsigned myIndex);

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

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
    : m_layer_index(myIndex)
{
    for (unsigned i = 0; i < numOutputs; ++i)
    {
        m_arr_outputWeights.push_back(t_Connection());
        m_arr_outputWeights.back().m_weight = randomWeight();
        m_arr_outputWeights.back().m_deltaWeight = 0;
    }
}

double Neuron::sumDOW(const t_Layer &arr_nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    unsigned num_neuron = (arr_nextLayer.size() - 1); // exclude bias neuron

    for (unsigned n = 0; n < num_neuron; ++n)
        sum += m_arr_outputWeights[n].m_weight * arr_nextLayer[n].m_gradient;

    return sum;
}

void Neuron::feedForward(const t_Layer &arr_prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < arr_prevLayer.size(); ++n)
        sum += arr_prevLayer[n].getOutputVal() * arr_prevLayer[n].m_arr_outputWeights[m_layer_index].m_weight;

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

    for (unsigned n = 0; n < arr_prevLayer.size(); ++n)
    {
        Neuron &neuron = arr_prevLayer[n];
        double oldDeltaWeight = neuron.m_arr_outputWeights[m_layer_index].m_deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                k_eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + k_alpha
                * oldDeltaWeight
                ;

        neuron.m_arr_outputWeights[m_layer_index].m_deltaWeight = newDeltaWeight;
        neuron.m_arr_outputWeights[m_layer_index].m_weight += newDeltaWeight;
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
    Net(const std::vector<unsigned> &arr_topology);

public: // public method(s)
    void feedForward(const t_vals &inputVals);
    void backProp(const t_vals &targetVals);
    void getResults(t_vals &resultVals) const;

public: // public method(s) -> error
    inline double getError(void) const { return m_error; }
    inline double getRecentAverageError(void) const { return m_recentAvgError; }
};

double Net::k_recentAvgSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const std::vector<unsigned>& arr_topology)
    :   m_error(0.0),
        m_recentAvgError(0.0)
{
    assert( !arr_topology.empty() ); // no empty topology

    for (unsigned i = 0; i < arr_topology.size(); ++i)
    {
        unsigned num_neuron = arr_topology[i];

        assert( num_neuron > 0 ); // no empty layer

        m_arr_layers.push_back(t_Layer());

        t_Layer& arr_new_layer = m_arr_layers.back();

        bool is_last_layer = (i == (arr_topology.size() - 1));

        // 0 output if on the last layer
        unsigned numOutputs = ((is_last_layer) ? (0) : (arr_topology[i + 1]));

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned j = 0; j < (num_neuron + 1); ++j) // add a bias neuron
            arr_new_layer.push_back( Neuron(numOutputs, j) );

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        Neuron& bias_neuron = arr_new_layer.back();
        bias_neuron.setOutputVal(1.0);
    }
}

void Net::feedForward(const t_vals &arr_inputVals)
{
    assert( arr_inputVals.size() == (m_arr_layers[0].size() - 1) ); // exclude bias neuron

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < arr_inputVals.size(); ++i)
        m_arr_layers[0][i].setOutputVal(arr_inputVals[i]);

    // forward propagate
    for (unsigned i = 1; i < m_arr_layers.size(); ++i) // exclude input layer
    {
        t_Layer& arr_prevLayer = m_arr_layers[i - 1];
        t_Layer& arr_currLayer = m_arr_layers[i];

        unsigned num_neuron = (arr_currLayer.size() - 1); // exclude bias neuron
        for (unsigned n = 0; n < num_neuron; ++n)
            arr_currLayer[n].feedForward(arr_prevLayer);
    }
}

void Net::backProp(const t_vals &arr_targetVals)
{
    //
    // error

    // Calculate overall net error (RMS of output neuron errors)

    t_Layer &arr_outputLayer = m_arr_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < arr_outputLayer.size() - 1; ++n)
    {
        double delta = arr_targetVals[n] - arr_outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= (arr_outputLayer.size() - 1); // get average error squared
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

    for (unsigned n = 0; n < (arr_outputLayer.size() - 1); ++n)
        arr_outputLayer[n].calcOutputGradients(arr_targetVals[n]);

    // Calculate hidden layer gradients

    for (unsigned i = (m_arr_layers.size() - 2); i > 0; --i)
    {
        t_Layer &arr_hiddenLayer = m_arr_layers[i];
        t_Layer &arr_nextLayer = m_arr_layers[i + 1];

        for (unsigned n = 0; n < arr_hiddenLayer.size(); ++n)
            arr_hiddenLayer[n].calcHiddenGradients(arr_nextLayer);
    }

    // Gradients
    //


    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned i = (m_arr_layers.size() - 1); i > 0; --i)
    {
        t_Layer &arr_currLayer = m_arr_layers[i];
        t_Layer &arr_prevLayer = m_arr_layers[i - 1];

        for (unsigned n = 0; n < (arr_currLayer.size() - 1); ++n) // exclude bias
            arr_currLayer[n].updateInputWeights(arr_prevLayer);
    }
}

void Net::getResults(t_vals &arr_resultVals) const
{
    arr_resultVals.clear();

    const t_Layer& arr_outputLayer = m_arr_layers.back();

    // exclude last neuron (bias neuron)
    unsigned total_neuron = (arr_outputLayer.size() - 1);

    for (unsigned n = 0; n < total_neuron; ++n)
        arr_resultVals.push_back(arr_outputLayer[n].getOutputVal());
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
    for (unsigned i = 0; i < arr_values.size(); ++i)
        std::cout << std::fixed << arr_values[i] << " ";

    std::cout << std::endl;
}


int main()
{
    TrainingData trainData("trainsample/out_xor.txt");
    // TrainingData trainData("trainsample/out_and.txt");
    // TrainingData trainData("trainsample/out_or.txt");
    // TrainingData trainData("trainsample/out_no.txt");

    // e.g., { 3, 2, 1 }
    std::vector<unsigned> arr_topology;
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

        unsigned dblarr_test[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };

        for (unsigned i = 0; i < 4; ++i)
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



