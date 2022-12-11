#include <iostream>

#include "kernels.cuh"

using namespace experiments;

int main() {
  auto kvset = MakeKeyValueSet(8);
  fmt::print("{}\n", kvset.num_pairs);

  auto hash = HashTable(kvset);
}