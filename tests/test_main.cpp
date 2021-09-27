#include <cassert>
#include <iostream>
#include <optional>
#include <string>

#include "test_lib.hpp"

#include "funnel_base.hpp"

using namespace std;
using namespace funnel;

namespace funnel {

class FunnelTest {
public:

  template<typename FunnelCellType>
  static void writeCellToStream(
      ostream& os,
      FunnelCellType const& cell) {
    if (cell.state.index() == 0) {
      os << cell.entry;
    } else if (cell.state.index() == 1) {
      auto& root = get<1>(cell.state);
      os << '[' << root.left
         << '|' << root.rank
         << '|' << root.right
         << ']'
         << '(' << cell.entry << ')';
    } else if (cell.state.index() == 2) {
      os << '[' << get<2>(cell.state) << ']'
         << '(' << cell.entry << ')';
    }
  }

  template<typename FunnelLayerType>
  static void writeLayerToStream(
      ostream& os,
      FunnelLayerType const& layer) {
    writeListToStream(
        os,
        layer.deque,
        writeCellToStream<typename FunnelLayerType::cell_type>);
  }

  template<typename FunnelBaseType>
  static void writeLayersToStream(
      ostream& os,
      FunnelBaseType const& funnel,
      char const* indent = "") {
    writeListToStream(
        os,
        funnel.layers_,
        writeLayerToStream<typename FunnelBaseType::layer_type>,
        "",
        indent,
        "\n");
  }

  template<typename FunnelBaseType>
  auto printLayers(
      FunnelBaseType const& funnel,
      char const* indent = "") {
    return FunctionPrinter{
        [&funnel, indent](ostream& os) {
          writeLayersToStream(os, funnel, indent);
        }};
  }

  template<typename FunnelBaseType>
  static ostream& printFunnelLayers(
      ostream& os,
      FunnelBaseType const& funnel,
      char const* layer_prefix = "",
      char const* layer_separator = "\n") {
    bool first = true;
    for (auto& layer : funnel.layers_) {
      if (!first) {
        os << layer_separator;
      }
      first = false;
      for (auto& cell : layer.deque) {
        os << layer_prefix;
        if (cell.state.index() == 0) {
          os << cell.entry << " ";
        } else {
          os << "(" << cell.entry << ") ";
        }
      }
    }
    return os;
  }

  void test_funnel_set() {
    FunnelSet<int> funnel;

    funnel.forceInsert(100, false);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(50, false);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(20, false);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(80, false);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(30, false);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(70, false);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(10, false);
    cout << printLayers(funnel) << endl;

    cout << "----- Optimize" << endl;
    funnel.optimize();
    cout << printLayers(funnel) << endl;

    cout << "----- Binary search" << endl;
    funnel.clear();

    funnel.forceInsert(100);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(50);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(20);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(80);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(30);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(70);
    cout << printLayers(funnel) << endl;
    funnel.forceInsert(10);
    cout << printLayers(funnel) << endl;
  }

  void run() {
    test_funnel_set();
  }
};

} // namespace funnel

void runTest() {
  funnel::FunnelTest funnel_test;
  funnel_test.run();
}

int main(int argc, char** argv) {
  try {
    runTest();
  } catch (AssertionError const& e) {
    cout
        << "FAILED assertion in file " << e.file_name
        << ", line number " << e.line_number
        << ": " << e.message
        << endl;
    if (!e.debug_message.empty()) {
      cout << "***** Debug message" << endl;
      cout << e.debug_message << endl;
      cout << "***** End debug message" << endl;
    }
    return 1;
  } catch (exception const& e) {
    cout << "ERROR: Exception thrown: " << e.what() << endl;
  } catch (...) {
    cout << "ERROR: Uncaught exception." << endl;
    return 1;
  }
  cout << "All tests PASSED." << endl;
  return 0;
}

