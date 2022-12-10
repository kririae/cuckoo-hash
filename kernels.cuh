#ifndef __HASH_TABLE_CUH_
#define __HASH_TABLE_CUH_

#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

using namespace thrust::placeholders;

template <typename T>
static void print_device_vector(T* vec, int num_elements,
                                const std::string& name = "") {
  auto ptr = thrust::device_pointer_cast(vec);
  fmt::print("[{}] [size={}] [gpu]: [", name, num_elements);
  for (int i = 0; i < num_elements; ++i) fmt::print("{} ", ptr[i]);
  fmt::print("]\n");
}

namespace experiments {
struct KeyValueSet {
  int*        kvpairs; /* interleaved [[key, value] [key, value]] */
  std::size_t num_pairs;
};

KeyValueSet MakeKeyValueSet(std::size_t num_keys) {
  auto kvpairs = thrust::malloc<int>(thrust::device_system_tag{}, 2 * num_keys);
  thrust::fill(kvpairs, kvpairs + 2 * num_keys, 1);
  thrust::inclusive_scan(kvpairs, kvpairs + 2 * num_keys, kvpairs);
  return KeyValueSet{.kvpairs   = thrust::raw_pointer_cast(kvpairs.get()),
                     .num_pairs = num_keys};
}
}  // namespace experiments

/// HASH TABLE
struct HashTableParams {
  std::size_t num_buckets;
  std::size_t num_pairs;
  int*        kvpairs;

  int* count; /* for distributing into buckets */
  int* start;
  int* offset;
  int* bucket_buffer; /* num_pairs * 2 */
};

__device__ __constant__ HashTableParams params;

#define LOCALIZE_CONSTANT_PARAMS                                           \
  [[maybe_unused]] const std::size_t num_buckets   = params.num_buckets;   \
  [[maybe_unused]] const std::size_t num_pairs     = params.num_pairs;     \
  [[maybe_unused]] const int*        kvpairs       = params.kvpairs;       \
  [[maybe_unused]] int*              count         = params.count;         \
  [[maybe_unused]] int*              start         = params.start;         \
  [[maybe_unused]] int*              offset        = params.offset;        \
  [[maybe_unused]] int*              bucket_buffer = params.bucket_buffer; \
  (void)num_buckets;                                                       \
  (void)num_pairs;                                                         \
  (void)kvpairs;                                                           \
  (void)count;                                                             \
  (void)start;                                                             \
  (void)offset;                                                            \
  (void)bucket_buffer;

namespace detail {
__device__ static int hash_bucket(int k) {
  LOCALIZE_CONSTANT_PARAMS
  return abs((19260817 + 16383 * k) % 1900813) % num_buckets;
}

/**
 * @brief This kernel initializes `count` and `offset` from the key-value set
 */
__global__ void distribute_buckets_kernel01() {
  LOCALIZE_CONSTANT_PARAMS

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  for (; i < num_pairs; i += blockDim.x * gridDim.x) {
    const int bucket_index = hash_bucket(i);
    const int prev_count   = atomicAdd(&count[bucket_index], 1);
    offset[i]              = prev_count;
    assert(offset[i] <= 512);
  }
}

/**
 * @brief This kernel store the (k, v) buffer into the buckets
 */
__global__ void distribute_buckets_kernel02() {
  LOCALIZE_CONSTANT_PARAMS

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < num_pairs; i += blockDim.x * gridDim.x) {
    const int bucket_index = hash_bucket(i);
    const int key          = kvpairs[2 * i];
    const int value        = kvpairs[2 * i + 1];

    const int start_index    = start[bucket_index];
    const int offset_index   = offset[i];
    const int index          = (start_index + offset_index) * 2;
    bucket_buffer[index]     = key;
    bucket_buffer[index + 1] = value;
  }
}

__global__ void cuckoo_hash_iteration() {
  LOCALIZE_CONSTANT_PARAMS

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < num_buckets; i += blockDim.x * gridDim.x) {
    const int* bucket_start = bucket_buffer + start[i];
  }
}
}  // namespace detail

class HashTable {
 public:
  /**
   * @brief Construct a new Hash Table from KeyValuePair
   *
   * @param kvset
   */
  HashTable(const experiments::KeyValueSet& kvset) {
    num_buckets = static_cast<std::size_t>(
        std::ceil(static_cast<double>(kvset.num_pairs) / 409));
    num_pairs = kvset.num_pairs;
    kvpairs   = kvset.kvpairs;

    // Perform some allocation
    count = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{}, sizeof(int) * num_buckets)
            .get());
    start = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{}, sizeof(int) * num_buckets)
            .get());
    offset = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{}, sizeof(int) * num_pairs)
            .get());
    bucket_buffer = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{}, sizeof(int) * num_pairs * 2)
            .get());

    thrust::fill(thrust::device_system_tag{}, count, count + num_buckets, 0);

    // setup constant memory
    setup_params();

    phase1();
    phase2();
  }

  ~HashTable() {
    thrust::free(thrust::device_system_tag{}, count);
    thrust::free(thrust::device_system_tag{}, start);
    thrust::free(thrust::device_system_tag{}, offset);
    thrust::free(thrust::device_system_tag{}, bucket_buffer);
  }

 private:
  /// Basic info
  std::size_t num_buckets, num_pairs;
  int*        kvpairs /* num_pairs */;
  int*        count /* num_buckets */;
  int*        start; /* num_buckets */
  int*        offset /* num_pairs */;
  int*        bucket_buffer /* num_pairs * 2 */;

  /// Cuckoo hashing info

  void setup_params() {
    auto host_params = HashTableParams{.num_buckets   = num_buckets,
                                       .num_pairs     = num_pairs,
                                       .kvpairs       = kvpairs,
                                       .count         = count,
                                       .start         = start,
                                       .offset        = offset,
                                       .bucket_buffer = bucket_buffer};
    cudaMemcpyToSymbol(params, &host_params, sizeof(HashTableParams));
  }

  /**
   * @brief Bucket collection phase
   */
  void phase1() {
    // 3: for each k ∈ keys in parallel do
    // 4:   compute h(k) to determine bucket bk containing k
    // 5:   atomically increment count[bk ], learning internal offset[k]
    // 6: end for
    constexpr int THREADS_PER_BLOCK = 512;
    const int     NUM_BLOCKS =
        (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    detail::distribute_buckets_kernel01<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>();

    // Collect the starting point
    thrust::exclusive_scan(thrust::device_system_tag{}, count,
                           count + num_buckets, start);

    // 8: for each key-value pair (k, v) in parallel do
    // 9:   store (k, v) in buffer at start[bk] + offset[k]
    // 10:end for
    detail::distribute_buckets_kernel02<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>();

#if 0
print_device_vector(start, num_buckets, "start");
print_device_vector(count, num_buckets, "count");
print_device_vector(offset, num_pairs, "offset");
print_device_vector(bucket_buffer, 2 * num_pairs, "bucket_buffer");
#endif

    cudaDeviceSynchronize();
  }

  /**
   * @brief Cuckoo hashing phase
   */
  void phase2() {}
};

#endif