
#pragma once

#include <vector>
#include <fstream>


// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

using t_vals = std::vector<double>;

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
