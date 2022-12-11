#include "gen.cuh"
#include "kernels.cuh"

using namespace experiments;

template <int NUM_HASH_FUNCTIONS>
class TestCases {
 public:
  void test_insertion(int s) {
    int  num_keys = 1 << s;
    auto kvset    = MakeRandomKeyValueSet(num_keys);
    fmt::print(">>> test_insertion\n");
    fmt::print("num_hash_functions: {}\n", NUM_HASH_FUNCTIONS);
    fmt::print("num_keys: {}\n", num_keys);
    auto hash = HashTable<NUM_HASH_FUNCTIONS>(kvset);
    fmt::print("<<< end test_insertion\n");
  }

  void test_lookup(int i) {
    // (100 - 10 * i): percentage of existing
    int  num_keys = 1 << 24;
    auto kvset    = MakeRandomKeyValueSet(num_keys);
    fmt::print(">>> test_lookup\n");
    float exist_proportion = (100.0 - 10 * i) / 100;
    fmt::print("exist_proportion: {:.3f}\n", exist_proportion);
    fmt::print("num_keys: {}\n", num_keys);
    auto hash = HashTable<NUM_HASH_FUNCTIONS>(kvset);

    auto query_kvset = MakeQueryKeyValueSetProportion(kvset, exist_proportion);
    hash.query(query_kvset);
    fmt::print("<<< end test_lookup\n");
  }

  void execute() {
    // Test 1
    for (int s = 10; s <= 24; ++s) test_insertion(s);

    // Test 2
    for (int i = 0; i < 10; ++i) test_lookup(i);
  }
};

int main() {
  TestCases<2> test_cases;
  test_cases.execute();

  TestCases<3> test_cases_alter;
  test_cases_alter.execute();
}
