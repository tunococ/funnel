#pragma once

#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

/**
 *  @brief
 *  Compares if two lists are equal.
 */
template<class A, class B>
constexpr bool listsEqual(A const& a, B const& b) {
  return std::equal(std::begin(a), std::end(a), std::begin(b), std::end(b));
}

/**
 *  @brief
 *  Compares if a list is equal to an initializer list.
 */
template<class A, class B>
constexpr bool listsEqual(A const& a, std::initializer_list<B> b) {
  return std::equal(std::begin(a), std::end(a), std::begin(b), std::end(b));
}

/**
 *  @brief
 *  Compares if a list is equal to an initializer list.
 */
template<class A, class B>
constexpr bool listsEqual(std::initializer_list<A> a, B const& b) {
  return std::equal(std::begin(a), std::end(a), std::begin(b), std::end(b));
}

/**
 *  @brief
 *  Struct containing a reference to a function that takes a `std::ostream&`
 *  and writes to it.
 *
 *  The member `print_fn` must be callable with one argument of type
 *  `std::ostream&`.
 *
 *  This type allows overloading `operator<<`.
 */
template <typename PrintFn>
struct FunctionPrinter {
  FunctionPrinter(PrintFn print_fn)
    : print_fn{print_fn} {}
  PrintFn print_fn;
};

/**
 *  @brief
 *  Calls `function_printer.print_fn(os)`, then returns `os`.
 */
template<typename PrintFn>
std::ostream& operator<<(
    std::ostream& os,
    FunctionPrinter<PrintFn> function_printer) {
  function_printer.print_fn(os);
  return os;
}

/**
 *  @brief
 *  Writes a list to an output stream with given element writer, `separator`,
 *  `prefix`, and `suffix`.
 *
 *  `write_element` is a customizable write function that takes two arguments.
 *  `write_element(os, element)` should write `element` to `os`, where `os` is
 *  `std::ostream&`.
 */
template<typename List, typename WriteElement>
void writeListToStream(
    std::ostream& os,
    List const& list,
    WriteElement write_element,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  bool first = true;
  for (auto& element : list) {
    if (!first) {
      os << separator;
    }
    os << prefix;
    write_element(os, element);
    os << suffix;
    first = false;
  }
}

/**
 *  @brief
 *  An overload of `writeListToStream()` that uses `operator<<` to write each
 *  element of the list.
 */
template<typename List>
void writeListToStream(
    std::ostream& os,
    List const& list,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  writeListToStream(
      os,
      list,
      [](std::ostream& os, typename List::value_type const& element) {
        os << element;
      },
      separator,
      prefix,
      suffix);
}

/**
 *  @brief
 *  Creates a `FunctionPrinter` object for writing a list to an output stream.
 *
 *  `write_element` is a customizable write function that takes two arguments.
 *  `write_element(os, element)` should write `element` to `os`, where `os` is
 *  `std::ostream&`.
 *
 *  The return value of `printList()` can be fed as an argument to
 *  `operator<<`.
 */
template<typename List, typename WriteElement>
auto printList(
    List const& list,
    WriteElement&& write_element,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return FunctionPrinter{
      [&list, &write_element, separator, prefix, suffix](std::ostream& os) {
        writeListToStream(
            os,
            list,
            write_element,
            separator, prefix, suffix);
      }};
}

/**
 *  @brief
 *  An overload of `printList()` that uses `operator<<` to write each element
 *  of the list.
 */
template<typename List>
auto printList(
    List const& list,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return FunctionPrinter{
      [&list, separator, prefix, suffix](std::ostream& os) {
        writeListToStream(
            os,
            list,
            separator, prefix, suffix);
      }};
}

/**
 *  @brief
 *  An overload of `printList()` that accepts an initializer list.
 */
template<typename E, typename WriteElement>
auto printList(
    std::initializer_list<E> list,
    WriteElement&& write_element,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return FunctionPrinter{
      [&list, &write_element, separator, prefix, suffix](std::ostream& os) {
        writeListToStream(
            os,
            list,
            write_element,
            separator, prefix, suffix);
      }};
}

/**
 *  @brief
 *  An overload of `printList()` that accepts an initializer list and uses
 *  `operator<<` to write each element of the list.
 */
template<typename E>
auto printList(
    std::initializer_list<E> const& list,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return FunctionPrinter{
      [&list, separator, prefix, suffix](std::ostream& os) {
        writeListToStream(
            os,
            list,
            separator, prefix, suffix);
      }};
}

// Funnel-specific helper functions.

/**
 *  @brief
 *  Writes a `FunnelCell` to a stream.
 *
 *  The output depends on the `state` of the cell.
 */
template<typename FunnelCellType>
static void writeCellToStream(
    std::ostream& os,
    FunnelCellType const& cell) {
  if (cell.state.index() == 0) {
    os << cell.entry;
  } else if (cell.state.index() == 1) {
    auto& root = std::get<1>(cell.state);
    os << '<' << root.left
       << '|' << root.rank
       << '|' << root.right
       << '>'
       << '(' << cell.entry << ')';
  } else if (cell.state.index() == 2) {
    os << '[' << std::get<2>(cell.state) << ']'
       << '(' << cell.entry << ')';
  }
}

/**
 *  @brief
 *  Writes a `FunnelLayer` to a stream.
 *
 *  Each cell is written to the stream by calling `writeCellToStream()`.
 */
template<typename FunnelLayerType>
static void writeLayerToStream(
    std::ostream& os,
    FunnelLayerType const& layer) {
  writeListToStream(
      os,
      layer.deque,
      writeCellToStream<typename FunnelLayerType::cell_type>);
}

/**
 *  @brief
 *  Creates a `FunctionPrinter` that wraps around `writeLayerToStream()`.
 */
template<typename FunnelLayerType>
static auto printLayer(FunnelLayerType const& layer) {
  return FunctionPrinter{
      [&layer](std::ostream& os) {
        writeLayerToStream(os, layer);
      }};
}

/**
 *  @brief
 *  Writes all layers of a `Funnel` to a stream.
 */
template<typename FunnelBaseType>
static void writeLayersToStream(
    std::ostream& os,
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

/**
 *  @brief
 *  Creates a `FunctionPrinter` that wraps around `writeLayersToStream()`.
 */
template<typename FunnelBaseType>
static auto printLayers(
    FunnelBaseType const& funnel,
    char const* indent = "") {
  return FunctionPrinter{
      [&funnel, indent](std::ostream& os) {
        writeLayersToStream(os, funnel, indent);
      }};
}

template<typename FunnelLayerType>
std::vector<typename FunnelLayerType::size_type> forwardLiveIndices(
    FunnelLayerType const& layer) {
  if (layer.empty()) {
    return {};
  }
  using size_type = typename FunnelLayerType::size_type;
  std::vector<size_type> output;
  size_type i{layer.rightDeadEndIndex(0)};
  for (; i < layer.size(); i = layer.nextLiveIndex(i)) {
    output.push_back(i);
  }
  return output;
}

template<typename FunnelLayerType>
std::vector<typename FunnelLayerType::size_type> backwardLiveIndices(
    FunnelLayerType const& layer) {
  if (layer.empty()) {
    return {};
  }
  using size_type = typename FunnelLayerType::size_type;
  std::vector<size_type> output;
  std::optional<size_type> i{layer.prevLiveIndex(layer.size())};
  for (; i; i = layer.prevLiveIndex(*i)) {
    output.push_back(*i);
  }
  std::reverse(output.begin(), output.end());
  return output;
}

