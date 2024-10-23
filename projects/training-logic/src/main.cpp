// neural-net-tutorial.cpp
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529


#include "./machine-learning/NeuralNetwork.hpp"

#include "./utilities/TrainingData.hpp"
#include "./utilities/RandomNumberGenerator.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <array>












//
//
// MAIN

void showVectorVals(const std::string& prefix, const t_vals &arr_values)
{
    std::cout << prefix << " ";
    for (uint32_t ii = 0; ii < arr_values.size(); ++ii)
    {
        std::cout << std::fixed << std::setprecision(2) << arr_values[ii] << " ";
    }
    std::cout << "\n";
}

void printUsageAndExit(const char* programName)
{
	std::cerr << "Usage: " << programName << " TRAINING_DATA_FILENAME" << std::endl;
	exit(EXIT_FAILURE);
}


int main(int argc, char** argv)
{
    if (argc != 2) {
		printUsageAndExit(argv[0]);
	}

    const std::string trainingFilename = argv[1];

    TrainingData trainData(trainingFilename);

    // e.g., { 2, 3, 1 }
    std::vector<uint32_t> arr_topology;
    trainData.getTopology(arr_topology);

    NeuralNetwork myNet(arr_topology);

    t_vals arr_inputVals;
    t_vals arr_targetVals;
    t_vals arr_resultVals;
    int32_t trainingPass = 0;

    while (!trainData.isEof())
    {
        ++trainingPass;
        std::cout << "\nPass " << trainingPass << "\n";

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
        std::cout << "Net current error: " << myNet.getError() << "\n";
        std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;

        if (
            // we need enough sample data for the average, here 100 samples
            trainingPass > 100 &&
            // is the average error acceptable?
            myNet.getRecentAverageError() < 0.05
        ) {
            std::cout << "\naverage error acceptable -> break" << std::endl;
            break;
        }
    }

    std::cout << "\nDone\n";

    if (
        arr_topology.size() < 2 ||
        arr_topology.front() != 2 ||
        arr_topology.back() != 1
    ) {
        std::cout << "Unexpected topology, no test\n";
    }
    else
    {
        std::cout << "TEST (trainingPass: " << trainingPass << ")\n\n";

        const std::array<std::array<uint32_t, 2>, 4> toTest =
        {{
            {{0,0}},
            {{0,1}},
            {{1,0}},
            {{1,1}}
        }};

        for (uint32_t ii = 0; ii < 4; ++ii)
        {
            arr_inputVals.clear();
            arr_inputVals.push_back(toTest[ii][0]);
            arr_inputVals.push_back(toTest[ii][1]);

            myNet.feedForward(arr_inputVals);
            myNet.getResults(arr_resultVals);

            showVectorVals("Inputs:", arr_inputVals);
            showVectorVals("Outputs:", arr_resultVals);

            std::cout << "\n";
        }

        std::cout << "/TEST" << std::endl;
    }
}

// MAIN
//
//



