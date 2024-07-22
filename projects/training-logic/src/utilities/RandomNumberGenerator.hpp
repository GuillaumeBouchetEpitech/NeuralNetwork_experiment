
#pragma once

#include <random>

class RandomNumberGenerator {
private:
  std::mt19937 _engine;

public:
  void setSeed(uint32_t seed);
  void ensureRandomSeed();

public:
  float getRangedValue(float min, float max);
  double getRangedValue(double min, double max);
};
