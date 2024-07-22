
#include "./RandomNumberGenerator.hpp"

#include <chrono>

void RandomNumberGenerator::setSeed(uint32_t seed) { _engine.seed(seed); }

void RandomNumberGenerator::ensureRandomSeed() {
  auto currTime = std::chrono::high_resolution_clock::now();
  auto seed = currTime.time_since_epoch().count();
  _engine.seed(uint32_t(seed));
}

float RandomNumberGenerator::getRangedValue(float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(_engine);
}

double RandomNumberGenerator::getRangedValue(double min, double max) {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(_engine);
}
