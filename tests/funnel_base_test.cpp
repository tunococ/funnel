#include <funnel/funnel_base.hpp>

#include <catch2/catch_test_macros.hpp>
#include <funnel_test_lib.hpp>

using namespace std;

namespace funnel {

class FunnelTest {
public:

};

} // namespace funnel

using namespace funnel;

TEST_CASE("FunnelLayer") {
  FunnelLayer<size_t> layer;

  size_t n = 23;
  for (size_t i{0}; i < n; ++i) {
    layer.push_back(i);
  }

  std::set<size_t> live_indices{};
  for (size_t i = 0; i < n; ++i) {
    live_indices.insert(i);
  }

  auto removeIndex = [&layer, &live_indices](size_t i) {
    if (layer.deque[i].isLive()) {
      layer.eraseAtIndex(i);
      live_indices.erase(i);
      return true;
    }
    return false;
  };

  auto checkLiveIndices = [&layer, &live_indices]() {
    CHECK(listsEqual(forwardLiveIndices(layer), live_indices));
    CHECK(listsEqual(backwardLiveIndices(layer), live_indices));
    CHECK(layer.live() == live_indices.size());
  };
  
  SECTION("Dead indices") {
    SECTION("Remove front") {
      for (size_t i = 0; i < 10; ++i) {
        removeIndex(i);
        for (size_t j = 0; j <= n; ++j) {
          CHECK(layer.rightDeadEndIndex(j) == (j <= i ? i + 1 : j));
          CHECK(layer.leftDeadBeginIndex(j) == (j <= i + 1 ? 0 : j));
        }
        checkLiveIndices();
      }
    }
    SECTION("Remove back") {
      for (size_t i = n; i > n - 10; ) {
        --i;
        removeIndex(i);
        for (size_t j = 0; j <= n; ++j) {
          CHECK(layer.rightDeadEndIndex(j) == (j >= i ? n : j));
          CHECK(layer.leftDeadBeginIndex(j) == (j >= i ? i : j));
        }
        checkLiveIndices();
      }
    }
  }

  SECTION("Live indices") {
    for (size_t m : {11, 7, 2, 5, 3, 19, 13, 17}) {
      cout << "  layer:\n    " << printLayer(layer) << "\n";
      cout << "  Removing every " << m << " elements...\n";
      for (size_t i = 0; i < n; i += m) {
        if (removeIndex(i)) {
          cout << "  After removing " << i << "\n    "
              << printLayer(layer) << "\n";
          checkLiveIndices();
        }
      }
    }
    removeIndex(1);
    checkLiveIndices();
    CHECK(layer.live() == 0);
  }
}

TEST_CASE("FunnelBase") {
  CHECK(false);
}

