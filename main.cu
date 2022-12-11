#include <iostream>

#include "kernels.cuh"

using namespace experiments;

int main() {
  constexpr int num_keys = 1 << 24;
  auto          kvset    = MakeKeyValueSet(num_keys);
  fmt::print("num_pairs: {}\n", kvset.num_pairs);

  auto hash = HashTable<3>(kvset);

  auto query_kvset = MakeQueryKeyValueSet(num_keys);
  hash.query(query_kvset);

  // validate
  auto dpc = thrust::device_pointer_cast(query_kvset.kvpairs);
  thrust::host_vector<int> hv(dpc, dpc + num_keys * 2);
  for (int i = 0; i < hv.size(); ++i) {
    if (hv[i] != i + 1) {
      fmt::print("{} {}\n", i, hv[i]);
    }
  }
}