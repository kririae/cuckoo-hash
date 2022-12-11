#ifndef __HASH_TABLE_CUH__
#define __HASH_TABLE_CUH__

#define FMT_HEADER_ONLY
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <fmt/core.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <random>

using namespace thrust::placeholders;
namespace cg = cooperative_groups;

template <typename T>
inline void print_device_vector(T* vec, int num_elements,
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

  ~KeyValueSet() { thrust::free(thrust::device_system_tag{}, kvpairs); }
};

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
  int* hash_table;    /* the result */

  int *c0, *c1; /* num_buckets * 3 */
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
  [[maybe_unused]] int*              hash_table    = params.hash_table;    \
  [[maybe_unused]] int*              c0            = params.c0;            \
  [[maybe_unused]] int*              c1            = params.c1;            \
  (void)num_buckets;                                                       \
  (void)num_pairs;                                                         \
  (void)kvpairs;                                                           \
  (void)count;                                                             \
  (void)start;                                                             \
  (void)offset;                                                            \
  (void)bucket_buffer;                                                     \
  (void)hash_table;                                                        \
  (void)c0;                                                                \
  (void)c1;

namespace detail {
__device__ int hash_bucket(int k) {
  LOCALIZE_CONSTANT_PARAMS
  uint64_t _k = static_cast<uint64_t>(k);
  return (114514 + 19260817 * _k) % num_buckets;
}

template <int SUBTABLE_SIZE>
__device__ int hash_cuckoo(int k, int c0, int c1) {
  uint64_t _k = static_cast<uint64_t>(k);
  return ((c0 + c1 * _k) % 1900813) % SUBTABLE_SIZE;
}

__device__ unsigned long long pack_key_value(int key, int value) {
  return (static_cast<unsigned long long>(value) << 32) | key;
}

/**
 * @brief This kernel initializes `count` and `offset` from the key-value set
 */
__global__ void distribute_buckets_kernel01() {
  LOCALIZE_CONSTANT_PARAMS

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  for (; i < num_pairs; i += blockDim.x * gridDim.x) {
    const int key          = kvpairs[2 * i];
    const int bucket_index = hash_bucket(key);
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
    const int key          = kvpairs[2 * i];
    const int value        = kvpairs[2 * i + 1];
    const int bucket_index = hash_bucket(key);

    const int start_index    = start[bucket_index];
    const int offset_index   = offset[i];
    const int index          = (start_index + offset_index) * 2;
    bucket_buffer[index]     = key;
    bucket_buffer[index + 1] = value;
  }
}

template <int NUM_SUBTABLES, int SUBTABLE_SIZE>
__global__ void cuckoo_hash_iteration() {
  LOCALIZE_CONSTANT_PARAMS

  cg::thread_block g = cg::this_thread_block();
  assert(g.num_threads() == 512);
  const int thread_rank = g.thread_rank();
  const int block_index = g.group_index().x;

  const int      MAX_ITER = 16;
  __shared__ int bucket[NUM_SUBTABLES][SUBTABLE_SIZE * 2 /* interleave */];

  // block vote
  using BlockReduce = cub::BlockReduce<int, 512>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ int                               block_vote;

  const int  bucket_index = block_index;
  const int  bucket_size  = count[bucket_index];
  const bool valid_op     = (thread_rank < bucket_size);
  const int  i            = (thread_rank + start[bucket_index]) * 2;
  const int  key          = valid_op ? bucket_buffer[i] : -1;
  const int  value        = valid_op ? bucket_buffer[i + 1] : -1;
  assert(thread_rank < 512);

  int chain_length = 0;
  while (true) {
    // initialize shared memory
#pragma unroll
    for (int k = 0; k < NUM_SUBTABLES; ++k) {
      if (thread_rank < SUBTABLE_SIZE) {
        bucket[k][thread_rank << 1]       = 0;
        bucket[k][(thread_rank << 1) + 1] = 0;
      }
    }

    // barrier
    g.sync();

    /// cuckoo iteration
    int it;
    for (it = 0; it < MAX_ITER; ++it) {
      const int subtable_index =
          (it % NUM_SUBTABLES);  // the subtable we're currently working on

      // Check if this key is already in the bucket
      bool exists_in_hashtable = false;
      if (valid_op)  // disable lane
#pragma unroll
        for (int k = 0; k < NUM_SUBTABLES; ++k) {
          const int c_index = NUM_SUBTABLES * bucket_index + k;
          const int g =
              hash_cuckoo<SUBTABLE_SIZE>(key, c0[c_index], c1[c_index]);
          exists_in_hashtable |= (bucket[k][g << 1] == key);
        }

      // to makesure that all `read` operations are done
      g.sync();

      // Collect the number of unfinished operations
      const int num_nonfinished =
          BlockReduce(temp_storage)
              .Reduce((!exists_in_hashtable && valid_op) ? 1 : 0, cub::Sum());
      if (g.thread_rank() == 0) block_vote = num_nonfinished;
      g.sync();
      if (block_vote == 0) break;

      // branch
      if (!exists_in_hashtable && valid_op) {
        const int c_index = NUM_SUBTABLES * bucket_index + subtable_index;
        const int g = hash_cuckoo<SUBTABLE_SIZE>(key, c0[c_index], c1[c_index]);

        // write key-value pair
#if __CUDA_ARCH__ >= 600
        atomicExch_block(reinterpret_cast<unsigned long long*>(
                             &bucket[subtable_index][g << 1]),
                         pack_key_value(key, value));
#else
        atomicExch(reinterpret_cast<unsigned long long*>(
                       &bucket[subtable_index][g << 1]),
                   pack_key_value(key, value));
#endif
      }

      // write i into the bucket
      g.sync();
    }

    ++chain_length;
    if (block_vote == 0) break;
    if (g.thread_rank() == 0) {
#if 0
      printf("[%d] [%d] error block_vote %d\n", block_index, thread_rank,
             block_vote);
#endif
// modify the random numbers
#pragma unroll
      for (int k = 0; k < NUM_SUBTABLES; ++k) {
        const int c_index = NUM_SUBTABLES * bucket_index + k;
        c0[c_index]++;
        c1[c_index]++;
      }
    }

    g.sync();
  }

  // const int max_chain_length =
  //     BlockReduce(temp_storage).Reduce(chain_length, cub::Max());
  // if (g.thread_rank() == 0) printf("max_chain_length: %d\n",
  // max_chain_length);

  // write local buckets to global hash_table
  int* hash_table_start =
      hash_table + (NUM_SUBTABLES * SUBTABLE_SIZE * 2) * bucket_index;
  for (int k = 0; k < NUM_SUBTABLES; ++k) {
    int* local_hash_table_start = hash_table_start + (SUBTABLE_SIZE * 2) * k;

    if (thread_rank < SUBTABLE_SIZE) {
      local_hash_table_start[thread_rank << 1] = bucket[k][thread_rank << 1];
      local_hash_table_start[(thread_rank << 1) + 1] =
          bucket[k][(thread_rank << 1) + 1];
    }
  }
}

template <int NUM_SUBTABLES, int SUBTABLE_SIZE>
__global__ void query_hashtable(int*        query_kvpairs,
                                std::size_t query_num_pairs) {
  LOCALIZE_CONSTANT_PARAMS

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < query_num_pairs; i += blockDim.x * gridDim.x) {
    const int key          = query_kvpairs[i << 1];
    const int bucket_index = hash_bucket(key);
    int*      hash_table_start =
        hash_table + (NUM_SUBTABLES * SUBTABLE_SIZE * 2) * bucket_index;

    for (int k = 0; k < NUM_SUBTABLES; ++k) {
      const int c_index = NUM_SUBTABLES * bucket_index + k;
      const int g = hash_cuckoo<SUBTABLE_SIZE>(key, c0[c_index], c1[c_index]);

      int* local_hash_table_start = hash_table_start + (SUBTABLE_SIZE * 2) * k;
      if (local_hash_table_start[g << 1] == key) {
        query_kvpairs[(i << 1) + 1] = local_hash_table_start[(g << 1) + 1];
        break;
      }
    }
  }
}
}  // namespace detail

template <int NUM_SUBTABLES = 2, int SUBTABLE_SIZE = 576 / NUM_SUBTABLES>
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
    hash_table_num_elements = num_buckets * NUM_SUBTABLES * SUBTABLE_SIZE * 2;
    hash_table_size         = hash_table_num_elements * sizeof(int);
    hash_table              = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{}, hash_table_size).get());

    c0 = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{},
                       sizeof(int) * num_buckets * NUM_SUBTABLES)
            .get());
    c1 = static_cast<int*>(
        thrust::malloc(thrust::device_system_tag{},
                       sizeof(int) * num_buckets * NUM_SUBTABLES)
            .get());

    // random generator on CPU
    std::random_device rd;
    std::mt19937       rng(rd());
    // rng.seed();
    std::uniform_int_distribution<int> uniform(1e4, 1e8);
    std::vector<int>                   host_vec(num_buckets * NUM_SUBTABLES);
    std::generate(host_vec.begin(), host_vec.end(), std::bind(uniform, rng));
    cudaMemcpy(c0, host_vec.data(), sizeof(int) * num_buckets * NUM_SUBTABLES,
               cudaMemcpyHostToDevice);
    std::generate(host_vec.begin(), host_vec.end(), std::bind(uniform, rng));
    cudaMemcpy(c1, host_vec.data(), sizeof(int) * num_buckets * NUM_SUBTABLES,
               cudaMemcpyHostToDevice);

    build();
  }

  ~HashTable() {
    thrust::free(thrust::device_system_tag{}, count);
    thrust::free(thrust::device_system_tag{}, start);
    thrust::free(thrust::device_system_tag{}, offset);
    thrust::free(thrust::device_system_tag{}, bucket_buffer);
    thrust::free(thrust::device_system_tag{}, hash_table);
    thrust::free(thrust::device_system_tag{}, c0);
    thrust::free(thrust::device_system_tag{}, c1);
  }

  void build() {
    // initialize all to zero
    reset();

    // setup constant memory
    setup_params();

    // main execution
    float       time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    phase1();
    phase2();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fmt::print("build_time: {:.3f}\n", time);
    fmt::print("build_MOPs: {:.3f}\n", num_pairs / (time / 1000.0) / 1e6);
  }

  void reset() {
    thrust::fill(thrust::device_system_tag{}, count, count + num_buckets, 0);
    thrust::fill(thrust::device_system_tag{}, hash_table,
                 hash_table + hash_table_num_elements, 0);
  }

  void query(const experiments::KeyValueSet& kvset) {
    constexpr int THREADS_PER_BLOCK = 512;
    const int     NUM_BLOCKS =
        (kvset.num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float       time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    detail::query_hashtable<NUM_SUBTABLES, SUBTABLE_SIZE>
        <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(kvset.kvpairs, kvset.num_pairs);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    fmt::print("query_time", time);
    fmt::print("query_MOPs: {:.3f}\n", kvset.num_pairs / (time / 1000.0) / 1e6);
  }

 private:
  /// Basic info
  std::size_t num_buckets, num_pairs;
  std::size_t hash_table_num_elements, hash_table_size;
  int*        kvpairs /* num_pairs */;
  int*        count /* num_buckets */;
  int*        start; /* num_buckets */
  int*        offset /* num_pairs */;
  int*        bucket_buffer /* num_pairs * 2 */;
  int*        hash_table /* num_buckets * num_subtables * 512 * 2 */;
  int *       c0, *c1; /* num_buckets * 3 */

  /// Cuckoo hashing info

  void setup_params() {
    auto host_params = HashTableParams{.num_buckets   = num_buckets,
                                       .num_pairs     = num_pairs,
                                       .kvpairs       = kvpairs,
                                       .count         = count,
                                       .start         = start,
                                       .offset        = offset,
                                       .bucket_buffer = bucket_buffer,
                                       .hash_table    = hash_table,
                                       .c0            = c0,
                                       .c1            = c1};
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

    cudaDeviceSynchronize();

#if 0
    print_device_vector(start, num_buckets, "start");
    print_device_vector(count, num_buckets, "count");
    print_device_vector(offset, num_pairs, "offset");
    print_device_vector(bucket_buffer, 2 * num_pairs, "bucket_buffer");
#endif
  }

  /**
   * @brief Cuckoo hashing phase
   */
  void phase2() {
    detail::cuckoo_hash_iteration<NUM_SUBTABLES, SUBTABLE_SIZE>
        <<<num_buckets, 512>>>();
    cudaDeviceSynchronize();

#if 0
    print_device_vector(start, num_buckets, "start");
#endif
  }
};

#endif