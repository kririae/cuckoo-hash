#ifndef __GEN_CUH__
#define __GEN_CUH__

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <boost/container/flat_map.hpp>
#include <cstddef>
#include <random>

namespace experiments {
struct KeyValueSet {
  int*        kvpairs; /* interleaved [[key, value] [key, value]] */
  std::size_t num_pairs;

  ~KeyValueSet() { thrust::free(thrust::device_system_tag{}, kvpairs); }
};

__global__ void make_query_kernel(int* kvpairs, std::size_t num_pairs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < num_pairs; i += blockDim.x * gridDim.x) kvpairs[i * 2 + 1] = 0;
}

void TransformKeyValueSetToQuery(KeyValueSet& kvset) {
  auto it  = thrust::counting_iterator<int>(0);
  int* ptr = kvset.kvpairs;
  thrust::for_each(
      thrust::device_system_tag{}, it, it + kvset.num_pairs,
      [ptr] __host__ __device__(int idx) -> void { ptr[idx * 2 + 1] = 0; });
}

KeyValueSet MakeRandomKeyValueSet(std::size_t num_keys) {
  auto kvpairs =
      thrust::malloc<int>(thrust::device_system_tag{}, 2 * num_keys).get();
  std::mt19937     rng(std::random_device{}());
  std::vector<int> host_vec(2 * num_keys);
  std::iota(host_vec.begin(), host_vec.end(), 1);
  std::shuffle(host_vec.begin(), host_vec.end(), rng);
  cudaMemcpy(kvpairs, host_vec.data(), sizeof(int) * 2 * num_keys,
             cudaMemcpyHostToDevice);
  return KeyValueSet{.kvpairs = kvpairs, .num_pairs = num_keys};
}

KeyValueSet MakeKeyValueSet(std::size_t num_keys) {
  auto kvpairs = thrust::malloc<int>(thrust::device_system_tag{}, 2 * num_keys);
  thrust::fill(kvpairs, kvpairs + 2 * num_keys, 1);
  thrust::inclusive_scan(kvpairs, kvpairs + 2 * num_keys, kvpairs);
  return KeyValueSet{.kvpairs   = thrust::raw_pointer_cast(kvpairs.get()),
                     .num_pairs = num_keys};
}

KeyValueSet MakeQueryKeyValueSet(std::size_t num_keys) {
  auto result = MakeKeyValueSet(num_keys);
  thrust::transform(
      thrust::device_system_tag{}, result.kvpairs,
      result.kvpairs + 2 * result.num_pairs, result.kvpairs,
      [] __host__ __device__(int val) { return val % 2 == 0 ? 0 : val; });
  return result;
}

KeyValueSet MakeQueryKeyValueSetProportion(const KeyValueSet& kvset,
                                           float              prop_existing) {
  // extract the first num_pairs * prop_existing * 2
  auto result = KeyValueSet{
      .kvpairs =
          thrust::malloc<int>(thrust::device_system_tag{}, 2 * kvset.num_pairs)
              .get(),
      .num_pairs = kvset.num_pairs,
  };

  const int num_exists = kvset.num_pairs * prop_existing;
  cudaMemcpy(result.kvpairs, kvset.kvpairs, num_exists * 2 * sizeof(int),
             cudaMemcpyDeviceToDevice);
  std::vector<int> host_vec(2 * (kvset.num_pairs - num_exists));
  std::iota(host_vec.begin(), host_vec.end(), 2 * kvset.num_pairs);
  cudaMemcpy(result.kvpairs + 2 * num_exists, host_vec.data(),
             host_vec.size() * sizeof(int), cudaMemcpyHostToDevice);

  /// I need stride iterator
  // TODO
  uint64_t* stride_pairs = reinterpret_cast<uint64_t*>(result.kvpairs);
  thrust::shuffle(thrust::device_system_tag{}, stride_pairs,
                  stride_pairs + kvset.num_pairs,
                  thrust::default_random_engine{});

  TransformKeyValueSetToQuery(result);
  return result;
}

boost::container::flat_map<int, int> MakeRefCPUMap(const KeyValueSet& kvset) {
  boost::container::flat_map<int, int> result;
  auto                     dpc = thrust::device_pointer_cast(kvset.kvpairs);
  thrust::host_vector<int> hv(dpc, dpc + kvset.num_pairs * 2);
  for (int i = 0; i < hv.size(); i += 2) result[hv[i]] = hv[i + 1];
  return result;
}
}  // namespace experiments

#endif
