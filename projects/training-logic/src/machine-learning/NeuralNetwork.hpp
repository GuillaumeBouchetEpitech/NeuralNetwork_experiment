
#pragma once

#include "./Neuron.hpp"


#include "../utilities/RandomNumberGenerator.hpp"

//
//
// NET

using t_vals = std::vector<double>;

class NeuralNetwork
{
private: // attr
    std::vector<t_Layer> m_arr_layers; // m_layers[layerNum][neuronNum]

private: // attr -> error
    double m_error;
    double m_recentAvgError;
private: // static attr -> error
    static double k_recentAvgSmoothingFactor;

public: // ctor/dtor
    NeuralNetwork(const std::vector<uint32_t> &arr_topology);

public: // public method(s)
    void feedForward(const t_vals &inputVals);
    void backProp(const t_vals &targetVals);
    void getResults(t_vals &resultVals) const;

public: // public method(s) -> error
    inline double getError(void) const { return m_error; }
    inline double getRecentAverageError(void) const { return m_recentAvgError; }
};

// NET
//
//


