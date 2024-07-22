

#include "./utilities/RandomNumberGenerator.hpp"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <functional>
#include <unordered_map>

//
//
//
//
//

void printUsage(const char* programName)
{
	std::cerr << "Usage 1: " << programName << " and" << std::endl;
	std::cerr << "Usage 2: " << programName << " or" << std::endl;
	std::cerr << "Usage 3: " << programName << " no" << std::endl;
	std::cerr << "Usage 4: " << programName << " xor" << std::endl;
}

//
//
//
//
//

using outputCallback = std::function<int(int, int)>;

void generateSamples(const outputCallback& callback)
{

	RandomNumberGenerator rng;
	rng.ensureRandomSeed();

	std::cout << "topology: 2 4 1" << std::endl;
	for (int ii = 0; ii < 2000; ++ii)
	{
		const int input1 = std::round(rng.getRangedValue(0.0f, 1.0f));
		const int input2 = std::round(rng.getRangedValue(0.0f, 1.0f));

		const int output = callback(input1, input2); // should be 0 or 1

		std::cout << "in: " << input1 << ".0 " << input2 << ".0 " << std::endl;
		std::cout << "out: " << output << ".0 " << std::endl;
	}

}

//
//
//
//
//

int main(int argc, char** argv)
{
	if (argc != 2) {
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}

	std::unordered_map<std::string, outputCallback> funcsMap;
	funcsMap["and"] = [](int input1, int input2) -> int { return input1 & input2; };
	funcsMap["or"] = [](int input1, int input2) -> int { return input1 | input2; };
	funcsMap["no"] = [](int input1, int input2) -> int { return !input1 & !input2; };
	funcsMap["xor"] = [](int input1, int input2) -> int { return input1 ^ input2; };

	std::string funcAsked = argv[1];

	const auto it = funcsMap.find(funcAsked);

	if (it == funcsMap.end()) {
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}

	const outputCallback callback = it->second;

	generateSamples(callback);

	return EXIT_SUCCESS;
}

