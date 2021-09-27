#pragma once

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

class AssertionError: public std::exception {
 public:
  AssertionError(
      std::string&& file_name,
      std::size_t line_number,
      std::string&& message,
      std::string&& debug_message) noexcept
    : file_name{file_name},
      line_number{line_number},
      message{message},
      debug_message{debug_message} {}
  AssertionError(AssertionError const&) noexcept = default;
  AssertionError(AssertionError&&) noexcept = default;
  AssertionError& operator=(AssertionError const&) noexcept = default;
  AssertionError& operator=(AssertionError&&) noexcept = default;
  virtual ~AssertionError() = default;
  virtual char const* what() const noexcept { return "AssertionError"; }

  std::string file_name;
  std::size_t line_number;
  std::string message;
  std::string debug_message;
};

class AssertionThrower {
 public:
  AssertionThrower(
      char const* file_name,
      std::size_t line_number,
      char const* message)
    : file_name{file_name}, line_number{line_number}, message{message} {
  }

  template<typename T>
  AssertionThrower& operator<<(T&& x) {
    debug_message << std::forward<T>(x);
    return *this;
  }

  operator bool() {
    throw AssertionError{
        std::move(file_name),
        line_number,
        std::move(message),
        debug_message.str()};
  }

  std::string file_name;
  std::size_t line_number;
  std::string message;
  std::ostringstream debug_message;
};

#define ASSERT(condition) \
  (condition) || AssertionThrower(__FILE__, __LINE__, #condition)

/**
 *  @brief
 *  Struct containing a reference to a function that takes a `std::ostream&`
 *  and writes to it.
 *
 *  The member `print_fn` must be callable with one argument of type
 *  `std::ostream&`.
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

template<typename List, typename WriteElement>
void writeListToStream(
    std::ostream& os,
    List&& list,
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

template<typename List>
void writeListToStream(
    std::ostream& os,
    List&& list,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return writeListToStream(
      os,
      std::forward<List>(list),
      [&os](typename List::value_type const& element) {
        os << element;
      },
      separator,
      prefix,
      suffix);
}

template<typename List, typename WriteElement>
auto printList(
    List&& list,
    WriteElement&& write_element,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return FunctionPrinter{
      [&list, &write_element, separator, prefix, suffix](std::ostream& os) {
        writeListToStream(
            os,
            std::forward<List>(list),
            write_element,
            separator, prefix, suffix);
      }};
}

template<typename List>
auto printList(
    List&& list,
    char const* separator = " ",
    char const* prefix = "",
    char const* suffix = "") {
  return FunctionPrinter{
      [&list, separator, prefix, suffix](std::ostream& os) {
        writeListToStream(
            os,
            std::forward<List>(list),
            separator, prefix, suffix);
      }};
}

