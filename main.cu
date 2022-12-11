#ifdef NDEBUG
#undef NDEBUG
#endif

#include <iostream>

#include "gen.cuh"
#include "kernels.cuh"

using namespace experiments;

int main() {
  constexpr int num_keys = 10;
  auto          kvset    = MakeRandomKeyValueSet(num_keys);
  auto          ref      = MakeRefCPUMap(kvset);
  fmt::print("num_pairs: {}\n", kvset.num_pairs);

  auto hash = HashTable<3>(kvset);

  auto query_kvset = MakeQueryKeyValueSet(num_keys);
  // auto query_kvset = MakeQueryKeyValueSetProportion(kvset, 0.2);
  print_device_vector(query_kvset.kvpairs, 2 * query_kvset.num_pairs,
                      "kvpairs");
  hash.query(query_kvset);

  // validate
  auto dpc = thrust::device_pointer_cast(query_kvset.kvpairs);
  thrust::host_vector<int> hv(dpc, dpc + num_keys * 2);
  for (int i = 0; i < hv.size(); i += 2) {
    bool exists_in_set = hv[i + 1] != 0;
    bool exists_in_map = ref.contains(hv[i]);
    assert(exists_in_set == exists_in_map);
    if (exists_in_set) {
      const int res_in_set = hv[i + 1];
      const int res_in_map = ref[hv[i]];
      assert(res_in_set == res_in_map);
    }
  }
}
