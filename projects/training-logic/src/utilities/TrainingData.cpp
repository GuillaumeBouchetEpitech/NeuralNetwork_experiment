
#include "./TrainingData.hpp"

#include <sstream>

TrainingData::TrainingData(const std::string& filename)
{
    m_file_trainingData.open(filename.c_str());

    if (m_file_trainingData.fail()) {
        throw std::invalid_argument("file not found");
    }
}

void TrainingData::getTopology(std::vector<unsigned> &arr_topology)
{
    arr_topology.reserve(10); // pre-allocate

    std::string line;
    std::string label;

    std::getline(m_file_trainingData, line);
    std::stringstream ss(line);
    ss >> label;

    if (this->isEof() || label != "topology:")
    {
        abort();
    }

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
