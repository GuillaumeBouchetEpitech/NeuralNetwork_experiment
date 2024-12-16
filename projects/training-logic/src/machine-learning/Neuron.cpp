
#include "Neuron.hpp"


#include <cassert>




namespace ActivationFunctions {
    namespace tanh {
        double activation(double x)
        {
            // tanh - output range [-1.0..1.0]
            return std::tanh(x);
        }

        double derivative(double x)
        {
            // tanh derivative

            // faster, less accurate
            // return 1.0 - x * x;

            return 1.0 - std::tanh(x * x);
        }
    }

    namespace relu {
        double activation(double x)
        {
            // relu - output range [0.0..1.0]
            return std::max(x, 0.0);
        }

        double derivative(double x)
        {
            // relu derivative
            // return x < 0.0 ? 0.0 : 1.0;
            return x < 0.0 ? 0.0 : x;
        }
    }

    namespace leakyRelu {
        double activation(double x)
        {
            // leaky relu
            if (x > 0) {
                return x;
            }
            return x * 0.1;
        }

        double derivative(double x)
        {
            // leaky relu derivative
            return x < 0.1 ? 0.0 : x;
        }
    }
}








namespace {
    double k_learningRate = 0.15;  // overall net learning rate, [0.0..1.0]
    double k_alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..1.0]
}

Neuron::Neuron(uint32_t numOutputs, uint32_t myIndex, RandomNumberGenerator& rng)
    : _layerIndex(myIndex)
{
    if (numOutputs == 0) {
        return;
    }

    _outputSynapses.reserve(numOutputs);
    for (uint32_t ii = 0; ii < numOutputs; ++ii) {
        _outputSynapses.emplace_back(rng.getRangedValue(0.0f, 1.0f));
    }
}

// #define D_USE_RELU

void Neuron::feedForward(const t_Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (uint32_t ii = 0; ii < prevLayer.size(); ++ii) {
        const auto& currNeuron = prevLayer[ii];

        sum += currNeuron._outputVal * currNeuron._outputSynapses[_layerIndex].weight;
    }

#ifndef D_USE_RELU
    _outputVal = ActivationFunctions::tanh::activation(sum);
#else
    _outputVal = ActivationFunctions::leakyRelu::activation(sum);
#endif
}

void Neuron::calcOutputGradients(double targetVal)
{
#ifndef D_USE_RELU
    const double delta = targetVal - _outputVal;
    _gradientOutputVal = delta * ActivationFunctions::tanh::derivative(_outputVal);
#else
    _gradientOutputVal = 2.0 * (_outputVal - targetVal);
    // const double delta = _outputVal - targetVal;
    // _gradientOutputVal = delta * ActivationFunctions::leakyRelu::derivative(_outputVal);
#endif
}

void Neuron::calcHiddenGradients(const t_Layer &nextLayer)
{
    const double dow = _sumDOW(nextLayer);
#ifndef D_USE_RELU
    _gradientOutputVal = dow * ActivationFunctions::tanh::derivative(_outputVal);
#else
    _gradientOutputVal = dow * ActivationFunctions::leakyRelu::derivative(_outputVal);
#endif
}

void Neuron::updateInputWeights(t_Layer &prevLayer) const
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (Neuron &prevNeuron : prevLayer) {

        auto& prevSynapse = prevNeuron._outputSynapses[_layerIndex];

        // const double oldDeltaWeight = prevSynapse.deltaWeight;

        const double newDeltaWeight =
            // Individual input, magnified by the gradient and train rate:
            k_learningRate * prevNeuron._outputVal * _gradientOutputVal
            // // Also add momentum = a fraction of the previous delta weight;
            // + k_alpha * oldDeltaWeight
            ;

        prevSynapse.deltaWeight = newDeltaWeight;
        prevSynapse.weight += newDeltaWeight;
    }
}

double Neuron::_sumDOW(const t_Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    const uint32_t totalNeurons = uint32_t(nextLayer.size()) - 1; // exclude bias neuron

    for (uint32_t ii = 0; ii < totalNeurons; ++ii) {
        sum += _outputSynapses[ii].weight * nextLayer[ii]._gradientOutputVal;
    }

    return sum;
}
