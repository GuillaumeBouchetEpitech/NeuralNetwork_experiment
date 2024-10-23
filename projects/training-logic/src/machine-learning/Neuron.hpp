
#pragma once



#include "../utilities/RandomNumberGenerator.hpp"



struct SynapseConnection
{
    double weight;
    double deltaWeight = 0;

    SynapseConnection(double inputWeight)
        : weight(inputWeight)
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
private: // attr
    double          _outputVal;
    SynapseConnections   _outputSynapses;
    uint32_t        _layerIndex;
    double          _gradientOutputVal; // used by the backpropagation

public: // ctor/dtor
    Neuron(uint32_t numOutputs, uint32_t myIndex, RandomNumberGenerator& rng);

public: // public method(s)
    void    feedForward(const t_Layer& arr_prevLayer);
    void    calcOutputGradients(double targetVal);
    void    calcHiddenGradients(const t_Layer& arr_nextLayer);
    void    updateInputWeights(t_Layer& arr_prevLayer) const;

public: // getter/setter
    inline void     setOutputVal(double val) { _outputVal = val; }
    inline double   getOutputVal(void) const { return _outputVal; }

private: // private method(s)
    double  _sumDOW(const t_Layer &arr_nextLayer) const;

};

// NEURON
//
//








