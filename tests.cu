#include "gen.cuh"
#include "kernels.cuh"

using namespace experiments;

template <int NUM_HASH_FUNCTIONS>
class TestCases {
 public:
  TestCases(int round) : m_round(round) {}

  void test_insertion(int s) {
    const int num_keys = 1 << s;
    auto      kvset    = MakeRandomKeyValueSet(num_keys);
    fmt::print(">>> test_insertion\n");
    print_basic_information();
    fmt::print("num_keys: {}\n", num_keys);
    auto hash = HashTable<NUM_HASH_FUNCTIONS>(kvset);
    fmt::print("<<< test_insertion\n");
  }

  void test_lookup(int i) {
    // (100 - 10 * i): percentage of existing
    const int num_keys = 1 << 24;
    auto      kvset    = MakeRandomKeyValueSet(num_keys);
    fmt::print(">>> test_lookup\n");
    float exist_proportion = (100.0 - 10 * i) / 100;
    print_basic_information();
    fmt::print("exist_proportion: {:.3f}\n", exist_proportion);
    fmt::print("num_keys: {}\n", num_keys);
    auto hash = HashTable<NUM_HASH_FUNCTIONS>(kvset);

    auto query_kvset = MakeQueryKeyValueSetProportion(kvset, exist_proportion);
    hash.query(query_kvset);
    fmt::print("<<< test_lookup\n");
  }

  template <int PARAMS>
  void test_insertion_table_size(bool execute = true) {
    const int num_keys = 1 << 24;
    fmt::print(">>> test_insertion_table_size\n");
    fmt::print("bucket_size: {}\n", PARAMS);
    print_basic_information();

    // We don't want to waste time re-executing these parts
    if (execute) {
      auto kvset = MakeRandomKeyValueSet(num_keys);
      auto hash =
          HashTable<NUM_HASH_FUNCTIONS, PARAMS / NUM_HASH_FUNCTIONS, 409>(
              kvset);
    }

    fmt::print("succeed: {}\n", execute);
    fmt::print(">>> test_insertion_table_size\n");
  }

  template <int MAX_ITER>
  void test_max_iter() {
    const int num_keys = 1 << 24;
    fmt::print(">>> test_max_iter\n");
    fmt::print("max_iter: {}\n", MAX_ITER);
    print_basic_information();

    // We don't want to waste time re-executing these parts
    auto kvset = MakeRandomKeyValueSet(num_keys);
    auto hash =
        HashTable<NUM_HASH_FUNCTIONS, 576 / NUM_HASH_FUNCTIONS, 409, MAX_ITER>(
            kvset);

    fmt::print(">>> test_max_iter\n");
  }

  void execute() {
    // Test 1
    for (int s = 10; s <= 24; ++s) test_insertion(s);

    // Test 2
    for (int i = 0; i < 10; ++i) test_lookup(i);

    // Test 3
    test_insertion_table_size<static_cast<int>(1.1 * 409)>(false);
    test_insertion_table_size<static_cast<int>(1.2 * 409)>(false);
    test_insertion_table_size<static_cast<int>(1.3 * 409)>(false);
    test_insertion_table_size<static_cast<int>(1.4 * 409)>();
    test_insertion_table_size<static_cast<int>(1.5 * 409)>();

    // Test 4
    recursive_function_call<15, 40>();
  }

 private:
  int m_round = -1;

  void print_basic_information() {
    fmt::print("num_hash_functions: {}\n", NUM_HASH_FUNCTIONS);
    fmt::print("round: {}\n", m_round);
  }

  template <int L, int U>
  void recursive_function_call() {
    if constexpr (L < U) {
      test_max_iter<L>();
      recursive_function_call<L + 1, U>();
    }
  }
};

int main() {
  for (int r = 0; r < 5; ++r) {
    TestCases<2> test_cases(r);
    test_cases.execute();

    TestCases<3> test_cases_alter(r);
    test_cases_alter.execute();
  }
}
