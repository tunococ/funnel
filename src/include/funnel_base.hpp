#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <tuple>
#include <type_traits>
#include <variant>

namespace funnel {

/**
 *  @brief
 *  Pair of an entry (value) and a state.
 *  The state is needed to support lazy deletion.
 *
 *  Lazy deletion is the only way deletion can be implemented for a funnel, so
 *  a *funnel cell* contains a desired entry and a state for managing deleted
 *  entries.
 *
 *  A cell is *live* if its entry has not been deleted; it is *dead* otherwise.
 */
template<class Entry, class SizeType = std::size_t>
struct FunnelCell {
  /// This type.
  using this_type = FunnelCell<Entry, SizeType>;
  /// `Entry`.
  using entry_type = Entry;
  /// `SizeType`.
  using size_type = SizeType;
  /// `std::make_signed_t<size_type>`.
  using offset_type = std::make_signed_t<size_type>;

  template<class... Args>
  FunnelCell(Args&&... args): entry(std::forward<Args>(args)...) {}

  FunnelCell(this_type const&) = default;
  FunnelCell(this_type&&) = default;
  this_type& operator=(this_type const&) = default;
  this_type& operator=(this_type&&) = default;
  
  /// Data for a root node in the disjoint set data structure.
  struct Root {
    /**
     *  @brief
     *  Distance to the nearest live cell on the left.
     */
    size_type left;
    /**
     *  @brief
     *  Distance to the nearest live cell on the right.
     */
    size_type right;
    /// Rank of the root--for the union-by-rank operation.
    size_type rank;

    Root(size_type left, size_type right, size_type rank)
      : left{left}, right{right}, rank{rank} {}
    Root() = default;
    Root(Root const&) = default;
    Root(Root&&) = default;
    Root& operator=(Root const&) = default;
    Root& operator=(Root&&) = default;
  };

  /// Value that contains a comparable key.
  entry_type entry;

  /**
   *  @brief
   *  State of the cell.
   *
   *  A funnel can support deletion via lazy deletion, i.e., marking cells
   *  whose entries have been *deleted* as *dead* instead of removing them from
   *  memory.
   *
   *  To support quick searching in the presence of dead cells, offsets to
   *  nearest live cells must be quickly accessible from each dead cell.
   *
   *  To support efficient bookkeeping of those offsets, a disjoint-set data
   *  structure is employed. Adjacent dead cells (those in the same contiguous
   *  segment of dead cells) belong to the same *set* because they have the
   *  same nearest live cells. A *root* stores the offsets to the nearest live
   *  cells, while a *child* only stores the offset to its parent.
   *
   *  Possible variant `index` values are:
   *  - 0: The cell is live.
   *  - 1: The cell is dead, and it is a root.
   *  - 2: The cell is dead, and it is not a root.
   *
   *  @remark
   *  `state` is delcared `mutable` because it may change during a union-find
   *  operation with path compression.
   */
  mutable std::variant<std::monostate, Root, offset_type> state{};

  constexpr bool isLive() const {
    return state.index() == 0;
  }

  constexpr bool isDead() const {
    return state.index() != 0;
  }
};

/**
 *  @brief
 *  Layer in a funnel.
 *
 *  A funnel contains a list of layers, each of which contains a list of cells
 *  sorted by their entries. The list of cells is called `deque`.
 *  The type `Deque` can be any list data structure that supports efficient
 *  random access with an index and insertion at both ends.
 *
 *  The leftmost cell and the rightmost cell in `deque`, if exist, are always
 *  live. That means the deletion of the leftmost cell is not lazy, and it will
 *  trigger removals of all consecutive dead cells from `deque` until `deque`
 *  is left with a live leftmost cell, or until `deque` is empty.
 *  The same is true for the deletion of the rightmost cell.
 *
 *  `FunnelLayer` keeps track of the number of live cells.
 *
 *  @tparam Entry
 *    Main storage type of *entries* in the funnel.
 *  @tparam Allocator
 *    Allocator type for `Entry`.
 *    This allocator type should support rebinding for type
 *    `Cell<entry_type, size_type>`, i.e.,
 *    `allocator_traits<Allocator>::rebind_alloc<Cell<entry_type, size_type>>`
 *    should be a compatible allocator type.
 *  @tparam Deque
 *    Functor of kind `* -> * -> *`.
 *    `Deque<T, A>` should be similar to `std::deque<T, A>`.
 *  @tparam Cell
 *    Functor of kind `* -> * -> *`.
 *    `Cell<T, U>` should be similar to `FunnelCell<T, U>`.
 */
template<
    class Entry,
    class Allocator = std::allocator<Entry>,
    template<class, class> class Deque = std::deque,
    template<class, class> class Cell = FunnelCell>
struct FunnelLayer {
  /// This type.
  using this_type = FunnelLayer<Entry, Allocator, Deque, Cell>;
  /// `Entry`.
  using entry_type = Entry;
  /// `Allocator`.
  using entry_allocator_type = Allocator;
  /// `entry_allocator_type::size_type`.
  using size_type = typename entry_allocator_type::size_type;
  /// `Cell<entry_type, size_type>`.
  using cell_type = Cell<entry_type, size_type>;
  /// `allocator_traits<entry_allocator_type>::rebind_alloc<cell_type>`.
  using allocator_type =
      typename std::allocator_traits<entry_allocator_type>::
      template rebind_alloc<cell_type>;
  /// `Deque<cell_type, allocator_type>`.
  using deque_type = Deque<cell_type, allocator_type>;

  static_assert(std::is_same_v<
      cell_type,
      typename deque_type::value_type>);
  static_assert(std::is_same_v<
      size_type,
      typename cell_type::size_type>);
  static_assert(std::is_same_v<
      size_type,
      typename deque_type::size_type>);

  /// `cell_type`.
  using value_type = cell_type; 
  /// `deque_type::difference_type`.
  using difference_type = typename deque_type::difference_type;
  /// `deque_type::reference`.
  using reference = typename deque_type::reference;
  /// `deque_type::const_reference`.
  using const_reference = typename deque_type::const_reference;
  /// `deque_type::pointer`.
  using pointer = typename deque_type::pointer;
  /// `deque_type::const_pointer`.
  using const_pointer = typename deque_type::const_pointer;
  /// `deque_type::iterator`.
  using iterator = typename deque_type::iterator;
  /// `deque_type::const_iterator`.
  using const_iterator = typename deque_type::const_iterator;
  /// `deque_type::reverse_iterator`.
  using reverse_iterator = typename deque_type::reverse_iterator;
  /// `deque_type::const_reverse_iterator`.
  using const_reverse_iterator = typename deque_type::const_reverse_iterator;

  /// `entry_type&`.
  using entry_reference = entry_type&;
  /// `entry_type const&`.
  using entry_const_reference = entry_type const&;
  /// `allocator_traits<entry_allocator_type>::pointer`.
  using entry_pointer =
      typename std::allocator_traits<entry_allocator_type>::pointer;
  /// `allocator_traits<entry_allocator_type>::const_pointer`.
  using entry_const_pointer =
      typename std::allocator_traits<entry_allocator_type>::const_pointer;

  // Static assertions
  static_assert(std::is_same<size_type, typename cell_type::size_type>::value);

  // Constructor
  FunnelLayer(allocator_type const& alloc) : deque(alloc) {}

  FunnelLayer(this_type const&) = default;
  FunnelLayer(this_type&&) = default;
  this_type& operator=(this_type const&) = default;
  this_type& operator=(this_type&&) = default;

  /**
   *  @brief
   *  Returns `index` if it points to a live cell in `deque`, or returns the
   *  index of a live cell closest to the right of the given `index`.
   *
   *  The return value is always greater than or equal to `index`.
   */
  static constexpr size_type rightLiveIndex(
      deque_type const& deque,
      size_type index) {
    return deque[index].isLive()
        ? index
        : index + std::get<1>(deque[findRootIndex(deque, index)].state).right;
  }

  /**
   *  @brief
   *  Calls `rightLiveIndex(deque, index)` where `deque` is the member variable
   *  of `FunnelLayer`.
   */
  constexpr size_type rightLiveIndex(size_type index) const {
    return rightLiveIndex(deque, index);
  }

  /**
   *  @brief
   *  Returns `index` if it points to a live cell in `deque`, or returns the
   *  index of a live cell closest to the left of the given `index`.
   *
   *  The return value is always smaller than or equal to `index`.
   */
  static constexpr size_type leftLiveIndex(
      deque_type const& deque,
      size_type index) {
    return deque[index].isLive()
        ? index
        : index - std::get<1>(deque[findRootIndex(deque, index)].state).left;
  }

  /**
   *  @brief
   *  Calls `leftLiveIndex(deque, index)` where `deque` is the member variable
   *  of `FunnelLayer`.
   */
  constexpr size_type leftLiveIndex(size_type index) const {
    return leftLiveIndex(deque, index);
  }

  /**
   *  @brief
   *  Returns the index of the live cell that is closest to the right of the
   *  given `index`. If `index` points to the last cell, `index + 1` will be
   *  returned.
   *
   *  This function is used for enumerating all live cells.
   */
  static constexpr size_type nextLiveIndex(
      deque_type const& deque,
      size_type index) {
    size_t j{index + 1};
    if (j == deque.size() || deque[j].isLive()) {
      return j;
    }
    return j + std::get<1>(deque[findRootIndex(deque, j)].state).right;
  }

  /**
   *  @brief
   *  Calls `nextLiveIndex(deque, index)` where `deque` is the member variable
   *  of `FunnelLayer`.
   */
  constexpr size_type nextLiveIndex(size_type index) const {
    return nextLiveIndex(deque, index);
  }

  template<class... Args>
  void emplace_front(Args&&... args) {
    deque.emplace_front(std::forward<Args>(args)...);
    ++num_live;
  }

  template<class... Args>
  void emplace_back(Args&&... args) {
    deque.emplace_back(std::forward<Args>(args)...);
    ++num_live;
  }

  template<class CellType>
  void push_back(CellType&& cell) {
    deque.push_back(std::forward<CellType>(cell));
    ++num_live;
  }

  /// Removes dead cells.
  void optimize() {
    // If there are no dead cells, do nothing.
    if (deque.size() == 0 || num_live == deque.size()) {
      return;
    }
    size_type i{1};
    // `i` must be smaller than `num_live` because there is a dead cell.
    for (; deque[i].isLive(); ++i) {}
    size_type j{i};
    while (i < num_live) {
      j = rightLiveIndex(j);
      deque[i] = std::move(deque[j]);
      ++i;
      ++j;
    }
    deque.resize(num_live);
  }

  constexpr size_type size() const {
    return deque.size();
  }

  constexpr size_type live() const {
    return num_live;
  }

  /**
   *  @brief
   *  Deletes an entry at the given `index`.
   *
   *  This function marks the cell dead, and may or may not remove the cell
   *  from `deque`.
   *
   *  The caller of this function has to make sure that the cell at the given
   *  `index` is live before calling it.
   *
   *  This function guarantees that there are no dead cells at the left or the
   *  right end of `deque`--those dead cells will be removed from `deque`.
   */
  void deleteEntry(size_type index) {
    --num_live;
    auto& state = deque[index].state;
    assert(state.index() == 0);

    // Mark this cell as dead.
    state.template emplace<2>(0);

    if (index == 0) {
      // If the marked cell is the leftmost cell, remove all dead cells from
      // the left end of `deque`.
      size_type j = 0;
      for (; j < deque.size() && deque[j].index != 0; ++j) {}
      deque.erase(deque.begin(), std::next(deque.begin(), j));
      return;
    }
    if (index + 1 == deque.size()) {
      // If the marked cell is the rightmost cell, remove all dead cells from
      // the right end of `deque`.
      size_type j = 0;
      size_type const last_index = deque.size() - 1;
      for (; j < deque.size() && deque[last_index - j].index != 0; ++j) {}
      deque.erase(std::next(deque.begin(), deque.size() - j), deque.end());
      return;
    }

    // The marked cell is somewhere in the middle.

    using offset_type = typename cell_type::offset_type;
    auto& left_state = deque[index - 1];
    auto& right_state = deque[index + 1];
    if (left_state.index() == 0) {
      // The cell on the left of the marked cell is live.
      if (right_state.index() == 0) {
        // The cell on the right of the marked cell is live.
        // The marked cell is an isolated dead cell, so a new root is created.
        state.template emplace<1>(1, 1, 0);
        return;
      }
      // The cell on the right of the marked cell is dead.
      // Find its root.
      size_type right_root_index = findRootIndex(index + 1);
      // Update the distance to the nearest live cell to the left.
      std::get<1>(deque[right_root_index].state).left += 1;
      // Link the marked cell to the root.
      state.template emplace<2>(
          static_cast<offset_type>(right_root_index) -
          static_cast<offset_type>(index));
      return;
    }

    // The cell on the left of the marked cell is dead.
    // Find its root.
    size_type left_root_index = findRootIndex(deque, index - 1);
    // Update the distance to the nearest live cell to the right.
    auto& left_root_state = deque[left_root_index].state;
    auto& left_root = std::get<1>(left_root_state);
    std::get<1>(left_root_state).right += 1;
    // Link the marked cell to the root.
    state.template emplace<2>(
        static_cast<offset_type>(left_root_index) -
        static_cast<offset_type>(index));

    if (right_state.index() == 0) {
      // The cell on the right of the marked cell is live.
      // There's nothing left to do.
      return;
    }

    // The cell on the right of the marked cell is also dead.
    // This means two contiguous segments of dead cells must be merged.

    // Find the root on the right.
    size_type right_root_index = findRootIndex(deque, index + 1);
    auto& right_root_state = deque[right_root_index].state;
    auto& right_root = std::get<1>(deque[right_root_index].state);

    if (left_root.rank < right_root.rank) {
      // Increase the span for the right root.
      size_type const left_span = left_root.left + left_root.right - 1;
      right_root.left += left_span;
      // Make the left root a child of the right root.
      left_root_state.template emplace<2>(
          static_cast<offset_type>(right_root_index) -
          static_cast<offset_type>(left_root_index));
      return;
    }
    // If the ranks are equal, increase the rank of the remaining root by one.
    if (left_root.rank == right_root.rank) {
      ++left_root.rank;
    }
    // Increase the span for the left root.
    size_type const right_span = right_root.left + right_root.right - 1;
    left_root.right += right_span;
    // Make the right root a child of the left root.
    right_root_state.template emplace<2>(
        static_cast<offset_type>(left_root_index) -
        static_cast<offset_type>(right_root_index));
  }

  constexpr void swap(this_type& other) {
    deque.swap(other.deque);
    std::swap(num_live, other.num_live);
  }

  // Member variables

  /// Sorted list of cells.
  deque_type deque;

  /**
   *  @brief
   *  The number of live cells in this layer.
   *
   *  This number is used to decide when to merge layers.
   */
  size_type num_live{0};

 protected:

  /**
   *  @brief
   *  Given a `deque` and an `index` of a dead cell, find the index of its
   *  root.
   *
   *  This function also performs path splitting.
   */
  static constexpr size_type findRootIndex(
      deque_type const& deque,
      size_type index) {
    assert(deque[index].state.index() != 0);
    while (true) {
      auto& state = deque[index].state;
      if (state.index() == 1) {
        return index;
      }
      size_type parent_index{index + std::get<2>(state)};
      auto& parent_state = deque[parent_index].state;

      if (parent_state.index() == 2) {
        // If parent is not a root, perform 1-step path compression.
        state.template emplace<2>(
            parent_index + std::get<2>(parent_state) - std::get<2>(state));
      }
      index = parent_index;
    }
  }

};

/**
 *  @brief
 *  Merge policy based on a rational ratio `SMALLER / LARGER`.
 *
 *  A *merge policy* consists of two callback functions: `mergeOnInsert()` and
 *  `mergeOnErase()`. These functions will be called by the funnel after an
 *  insert or an erase operation finishes. The layer iterator that points to
 *  the layer whose element was inserted to erased will be supplied in the
 *  callback.
 *
 *  The merge criterion after insertion is meant to guarantee that two adjacent
 *  layers have sizes that are different enough. This is necessary for bounding
 *  the number of layers based on the number of all cells (dead and alive) in
 *  the funnel.
 *
 *  The merge criterion after deletion is meant to guarantee that not too many
 *  dead cells remain in the funnel. This is necessary for bounding the number
 *  of all cells in the funnel based on the number of live cells.
 *
 *  The *never-merge* policy can be created by choosing `LARGER = 0`.
 *
 *  When `SMALLER` and `LARGER` are positive, we get the following guarantees:
 *  - The number of live cells is bounded below by
 *    `c * SMALLER / (SMALLER + LARGER)`; and
 *  - The number of layers is bounded above by
 *    `log(c) / log(LARGER / SMALLER)`;
 *  where `c` is the number of all cells (dead and live) in the funnel.
 *  Consequently, if we let `n` denote the number of live cells in the funnel
 *  (which is the *size* of the funnel), then the number of layers is bounded
 *  above by `log(n * (SMALLER + LARGER) / SMALLER) / log(LARGER / SMALLER)`.
 */
template<std::size_t SMALLER, std::size_t LARGER, class FunnelType>
struct SimpleRatioMerger {
  using this_type = SimpleRatioMerger<SMALLER, LARGER, FunnelType>;
  using funnel_type = FunnelType;
  using size_type = typename FunnelType::size_type;
  using layer_type = typename FunnelType::layer_type;
  using layer_list_type = typename FunnelType::layer_list_type;
  using layer_iterator = typename FunnelType::layer_iterator;
  using const_layer_iterator = typename FunnelType::const_layer_iterator;

  static constexpr size_type smaller{static_cast<size_type>(SMALLER)};
  static constexpr size_type larger{static_cast<size_type>(LARGER)};

  /**
   *  @brief
   *  Merges `layer` with its previous layer if the *merge criterion* based on
   *  the ratio `smaller / larger` is met, sets `layer` to the previous layer,
   *  and repeats this process until the merge criterion is not met or until
   *  `layer` reaches `funnel.layers_.begin()`. At the end, `layer` is
   *  returned.
   *
   *  The *merge criterion* is:
   *  - `larger` is not `0`, and the ratio `layer->live() / prev_size` exceeds
   *    `smaller / larger`, where `prev_size` is the size (counting both dead
   *    and live cells) of the previous layer.
   *
   *  The returned `layer` will contain all the elements in the original input
   *  `layer`. This can be used to identify the layer that contains the newly
   *  inserted element after merging.
   *
   *  This merge function guarantees that the number of layers will be
   *  `O(log n)` where `n` is the number of cells. More specifically, the
   *  number of layers is bounded above by `log(n) / log(larger / smaller)`.
   */
  layer_iterator mergeOnInsert(FunnelType& funnel, layer_iterator layer) {
    while (layer != funnel.layers_.begin()) {
      layer_iterator prev_layer = std::prev(layer);
      if (larger * layer->live() <= smaller * prev_layer->size()) {
        break;
      }
      funnel.mergeLayers(prev_layer, layer);
      layer = prev_layer;
    }
    return layer;
  }

  /**
   *  @brief
   *  Merges `layer` with its next layer if the *merge criterion* based on the
   *  ratio `smaller / larger` is met, then calls `mergeOnInsert()`.
   *
   *  The *merge criterion* is:
   *  - `larger` is not `0`, and the ratio
   *    `layer->live() / layer->size()` is smaller than or equal to
   *    `smaller / (smaller + larger)`.
   *
   *  The return value will be `layer` if no merging occurs; otherwise, it will
   *  be the return value of the call to `mergeOnInsert()`.
   *
   *  This merge function guarantees that the total number of live cells will
   *  be `O(n)` where `n` is the number of cells. More specifically, the ratio
   *  of dead cells over all cells in each layer is bounded above by
   *  `larger / (smaller + larger)`.
   */
  layer_iterator mergeOnErase(FunnelType& funnel, layer_iterator layer) {
    if ((smaller + larger) * layer->live() > smaller * layer->size()) {
      return layer;
    }
    funnel.mergeLayers(layer, std::next(layer));
    return mergeOnInsert(funnel, layer);
  }
};

/**
 *  @brief
 *  Helper class for currying the first two parameters of `SimpleRatioMerger`.
 */
template<std::size_t SMALLER, std::size_t LARGER>
struct SimpleRatioMergerCurried {
  template<class FunnelType>
  using type = SimpleRatioMerger<SMALLER, LARGER, FunnelType>;
};

/**
 *  @brief
 *  Base class for `FunnelSet`, `FunnelMap`, `FunnelMultiset`, and
 *  `FunnelMultimap`.
 *
 *  A funnel consists of a list of *layers*, each of which is a sorted deque of
 *  *cells*. A cell consists of an *entry* and a *state*. Cells are ordered by
 *  entries that they contain.
 *
 *  To cover both *set* and *map* data structures, `FunnelBase` takes `Entry`
 *  and `Key` as potentially different types.
 *
 *  - For `FunnelSet` and `FunnelMultiset`, `Key` and `Entry` will be the same,
 *    and `ExtractKey` is the type of an identity function.
 *  - For `FunnelMap` and `FunnelMultimap`, `Entry` will be a pair of `Key` and
 *    `Value`, and `ExtractKey` will take `Entry` and return its `Key`.
 *
 *  @tparam Entry
 *    Main storage type of *entries* in the funnel.
 *  @tparam DeleteEntry
 *    Since a funnel uses lazy deletion, a deleted entry may still hold its place
 *    in a funnel--the containing cell may simply be marked *dead*. This means
 *    the destructor of the entry might not be called right after it is deleted.
 *    To allow the user to perform an immediate optimize when an entry is marked
 *    deleted, the user can supply a *deletion callback* during the construction
 *    of a funnel. This callback must behave like a function with prototype
 *    `void(Entry*)`. `DeleteEntry` is the type of the callback.
 *  @tparam Key
 *    Type of *keys* in the funnel.
 *    For `FunnelSet` and `FunnelMultiset`, `Key` will be the same as `Entry`.
 *    For `FunnelMap` and `FunnelMultimap`, `Entry` will be
 *    `std::pair<Key, Value>`, where `Value` is the *value* type.
 *  @tparam ExtractKey
 *    For `FunnelSet` and `FunnelMultiset`, `ExtractKey` will be the type of
 *    an identity function.
 *    For `FunnelMap` and `FunnelMultimap`, `ExtractKey` will be the type of
 *    a function that projects `std::pair<Key, Value>` onto its `first`
 *    element.
 *  @tparam Compare
 *    Type of the comparison operator.
 *    This should be compatible with a function with prototype
 *    `bool(Key const&, key const&)`.
 *  @tparam Allocator
 *    Allocator type for `Entry`.
 *    This allocator type should support rebinding for the following types:
 *    `layer_type`, `cell_type` and `HorizontalIterator`.
 *  @tparam Deque
 *    Functor of kind `* -> * -> *`.
 *    `Deque` will be supplied as the third argument of `Layer`.
 *  @tparam LayerList
 *    Functor of kind `* -> * -> *`.
 *    `FunnelBase` will instantiate
 *    `LayerList<layer_type, layer_allocator_type>` and expect it to behave
 *    like `std::vector<layer_type, layer_allocator_type>`.
 *  @tparam Cell
 *    Functor of kind `* -> * -> *`.
 *    `Cell` will be supplied as the fourth argument of `Layer`.
 *  @tparam Layer
 *    Functor of kind `* -> * -> (* -> * -> *) -> (* -> * -> *) -> *`.
 *    `FunnelBase` will instantiate
 *    `Layer<Entry, Allocator, Deque, Cell>` and expect it to behave like
 *    `FunnelLayer<Entry, Allocator, Deque, Cell>`.
 *  @tparam Merger
 *    Functor of kind `* -> *`.
 *    `FunnelBase` will instantiate `Merger<this_type>` and expect it to
 *    contain the following two callback functions:
 *    \code{.cpp}
 *        layer_iterator mergeOnInsert(this_type&, layer_iterator);
 *        layer_iterator mergeOnErase(this_type&, layer_iterator);
 *    \endcode
 *    As a simple example, `Merger<this_type>` should behave like
 *    `SimpleRatioMerger<1, 2, this_type>`.
 */
template<
    class Entry,
    class DeleteEntry,
    class Key,
    class ExtractKey,
    class Compare = std::less<Key>,
    class Allocator = std::allocator<Entry>,
    template<class, class> class Deque = std::deque,
    template<class, class> class LayerList = std::vector,
    template<class> class Merger = SimpleRatioMergerCurried<1, 2>::type,
    template<class, class> class Cell = FunnelCell,
    template<
      class,
      class,
      template<class, class> class,
      template<class, class> class> class Layer = FunnelLayer>
class FunnelBase {
 public:
  /// This class.
  using this_type = FunnelBase<
      Entry,
      DeleteEntry,
      Key,
      ExtractKey,
      Compare,
      Allocator,
      Deque,
      LayerList,
      Merger,
      Cell,
      Layer>;
  /// `Entry`.
  using entry_type = Entry;
  /// `DeleteEntry`.
  using delete_entry = DeleteEntry;
  /// `Key`.
  using key_type = Key;
  /// `ExtractKey`.
  using extract_key = ExtractKey;
  /// `Compare`.
  using key_compare = Compare;
  /// `Allocator`.
  using allocator_type = Allocator;
  /// `allocator_type::size_type`.
  using size_type = typename allocator_type::size_type;
  /// `Layer<Entry, Allocator, Deque, Cell>`.
  using layer_type = Layer<Entry, Allocator, Deque, Cell>;

  static_assert(std::is_same_v<
      size_type,
      typename layer_type::size_type>);
  static_assert(std::is_same_v<
      entry_type,
      typename layer_type::entry_type>);

  /// `allocator_traits<allocator_type>::rebind_alloc<layer_type>`.
  using layer_allocator_type =
      typename std::allocator_traits<allocator_type>::
      template rebind_alloc<layer_type>;
  /// `LayerList<layer_type, layer_allocator_type>`.
  using layer_list_type = LayerList<layer_type, layer_allocator_type>;

  static_assert(std::is_same_v<
      size_type,
      typename layer_list_type::size_type>);
  static_assert(std::is_same_v<
      layer_type,
      typename layer_list_type::value_type>);

  /// `layer_type::cell_type`.
  using cell_type = typename layer_type::cell_type;
  /// `layer_type::deque_type`.
  using deque_type = typename layer_type::deque_type;
  /// `Merger<this_type>`.
  using merger_type = Merger<this_type>;

  /// `entry_type`.
  using value_type = entry_type;
  /// `layer_type::difference_type`.
  using difference_type = typename layer_type::difference_type;
  /// `layer_type::allocator_type`.
  using cell_allocator_type = typename layer_type::allocator_type;
  /// `value_type&`.
  using reference = value_type&;
  /// `value_type const&`.
  using const_reference = value_type const&;
  /// `layer_type::entry_pointer`.
  using pointer = typename layer_type::entry_pointer;
  /// `layer_type::entry_const_pointer`.
  using const_pointer = typename layer_type::entry_const_pointer;

  /**
   *  @brief
   *  Operator for comparing the member `entry` in `cell_type` objects.
   */
  struct cell_compare {
    key_compare key_comp;
    extract_key ext_key;
    constexpr cell_compare(
        extract_key const& ext_key, key_compare const& key_comp)
      : ext_key{ext_key}, key_comp{key_comp} {}
    cell_compare(cell_compare const&) = default;
    cell_compare(cell_compare&&) = default;
    template<class K1, class K2>
    constexpr bool operator()(
        K1 const& a,
        K2 const& b) const {
      return key_comp(a, b);
    }
    template<class K1, class S1, class K2>
    constexpr bool operator()(
        FunnelCell<K1, S1> const& a,
        K2 const& b) const {
      return key_comp(ext_key(a.entry), b);
    }
    template<class K1, class K2, class S2>
    constexpr bool operator()(
        K1 const& a,
        FunnelCell<K2, S2> const& b) const {
      return key_comp(a, ext_key(b.entry));
    }
    template<class K1, class S1, class K2, class S2>
    constexpr bool operator()(
        FunnelCell<K1, S1> const& a,
        FunnelCell<K2, S2> const& b) const {
      return key_comp(ext_key(a.entry), ext_key(b.entry));
    }
    template<class A, class B>
    constexpr int compare(A const& a, B const& b) const {
      return (*this)(a, b) ? -1 : ((*this)(b, a) ? 1 : 0);
    }
  };

 protected:
  /// `this_type`.
  using funnel_type = this_type;

  /**
   *  @brief
   *  Iterator types of `FunnelBase`.
   *
   *  The template argument specifies whether it is a const or a non-const
   *  version.
   */
  template<bool constant>
  class Iterator;

  /**
   *  @brief
   *  Reverse iterator types of `FunnelBase`.
   *
   *  The template argument specifies whether it is a const or a non-const
   *  version.
   */
  template<bool constant>
  class ReverseIterator;

 public:
  using iterator = Iterator<false>;
  using const_iterator = Iterator<true>;
  using reverse_iterator = ReverseIterator<false>;
  using const_reverse_iterator = ReverseIterator<true>;
  using layer_iterator = typename layer_list_type::iterator;
  using const_layer_iterator = typename layer_list_type::const_iterator;

  /**
   *  @brief
   *  Constructs an empty `FunnelBase` object.
   *
   *  @param del_ent
   *    Unary operator that will be called when an entry is *erased* from the
   *    funnel. `del_ent(v)` should be valid when `v` has type
   *    `value_type const&`.
   *  @param ext_key
   *    Unary operator for extracting `key_type const&` from
   *    `value_type const&`. `ext_key(v)` should return `key_type const&` when
   *    `v` has type `value_type const&`.
   *  @param key_comp
   *    Binary operator for comparing two `key_type const&` values.
   *    `key_comp(a, b)` should return `true` if and only if its `a` is strictly
   *    less than `b`, where `a` and `b` have type `key_type const&`.
   *  @param alloc
   *    Allocator for constructing objects of type `value_type`.
   *    `alloc` must be rebindable to allocators for types `cell_type` and
   *    `layer_type`.
   *  @param merger
   *    Merge callback object.
   *    `merger.mergeOnInsert(*this, layer)` will be called after an element is
   *    added to `layer` of the funnel, while
   *    `merger.mergeOnErase(*this, layer)` will be called after an element is
   *    removed from `layer` of the funnel.
   *    Both functions should return a `layer_iterator` pointing to the layer
   *    that eventually contains all the elements in the original input
   *    `layer`.
   *    `FunnelBase` will always supply a dereferenceable `layer` to these
   *    functions. (`layer` cannot be the past-the-end iterator.)
   */
  constexpr FunnelBase(
      delete_entry const& del_ent = delete_entry(),
      extract_key const& ext_key = extract_key(),
      key_compare const& key_comp = key_compare(),
      allocator_type const& alloc = allocator_type(),
      merger_type const& merger = merger_type())
    : del_ent_{del_ent},
      cell_comp_{ext_key, key_comp},
      value_alloc_{alloc},
      layer_alloc_{alloc},
      cell_alloc_{alloc},
      layers_{layer_alloc_},
      merger_{merger} {
  }

  /// Returns the `delete_entry` callback.
  delete_entry del_ent() const noexcept {
    return del_ent_;
  }

  /// Returns the `extract_key` callback.
  extract_key ext_key() const noexcept {
    return cell_comp_.ext_key_;
  }

  /// Returns the `key_compare` callback.
  key_compare key_comp() const noexcept {
    return cell_comp_.key_comp;
  }

  /// Returns the allocator.
  allocator_type get_allocator() const noexcept {
    return value_alloc_;
  }

  /// Returns the `merger_type` callback object.
  merger_type get_merger() const noexcept {
    return merger_;
  }

  /// Default copy constructor.
  FunnelBase(this_type const&) = default;
  /// Default move constructor.
  FunnelBase(this_type&&) = default;
  /// Default copy assignment.
  this_type& operator=(this_type const&) = default;
  /// Default move assignment.
  this_type& operator=(this_type&&) = default;

  constexpr void clear() {
    layers_.clear();
    num_cells_ = 0;
    size_ = 0;
  }

  constexpr size_type size() const {
    return size_;
  }

  constexpr bool empty() const {
    return size_ == 0;
  }

  constexpr iterator begin() {
    return iterator::begin(*this);
  }

  constexpr const_iterator begin() const {
    return const_iterator::begin(*this);
  }

  constexpr const_iterator cbegin() const {
    return const_iterator::begin(*this);
  }

  constexpr iterator begin_async(std::launch policy) {
    return iterator::begin_async(*this, policy);
  }

  constexpr const_iterator begin_async(std::launch policy) const {
    return const_iterator::begin_async(*this, policy);
  }

  constexpr const_iterator cbegin_async(std::launch policy) const {
    return const_iterator::begin_async(*this, policy);
  }

  constexpr iterator end() {
    return iterator::end(*this);
  }

  constexpr const_iterator end() const {
    return const_iterator::end(*this);
  }

  constexpr const_iterator cend() const {
    return const_iterator::end(*this);
  }

  constexpr iterator end_async(std::launch policy) {
    return iterator::end_async(*this, policy);
  }

  constexpr const_iterator end_async(std::launch policy) const {
    return const_iterator::end_async(*this, policy);
  }

  constexpr const_iterator cend_async(std::launch policy) const {
    return const_iterator::end_async(*this, policy);
  }

  template<class K>
  constexpr iterator lower_bound(K const& key) {
    return iterator::lower_bound(*this, key);
  }

  template<class K>
  constexpr const_iterator lower_bound(K const& key) const {
    return const_iterator::lower_bound(*this, key);
  }

  template<class K>
  constexpr iterator lower_bound_async(
      std::launch policy,
      K const& key) {
    return iterator::lower_bound_async(*this, policy, key);
  }

  template<class K>
  constexpr const_iterator lower_bound_async(
      std::launch policy,
      K const& key) const {
    return const_iterator::lower_bound_async(*this, policy, key);
  }

  template<class K>
  constexpr iterator upper_bound(K const& key) {
    return iterator::upper_bound(*this, key);
  }

  template<class K>
  constexpr const_iterator upper_bound(K const& key) const {
    return const_iterator::upper_bound(*this, key);
  }

  template<class K>
  constexpr iterator upper_bound_async(
      std::launch policy,
      K const& key) {
    return iterator::upper_bound_async(*this, policy, key);
  }

  template<class K>
  constexpr const_iterator upper_bound_async(
      std::launch policy,
      K const& key) const {
    return const_iterator::upper_bound_async(*this, policy, key);
  }

  template<class K>
  constexpr std::pair<iterator, iterator> equal_range(K const& key) {
    return iterator::equal_range(*this, key);
  }

  template<class K>
  constexpr std::pair<const_iterator, const_iterator> equal_range(
      K const& key) const {
    return const_iterator::equal_range(*this, key);
  }

  template<class K>
  constexpr std::pair<iterator, iterator> equal_range_async(
      std::launch policy,
      K const& key) {
    return iterator::equal_range_async(*this, policy, key);
  }

  template<class K>
  constexpr std::pair<const_iterator, const_iterator> equal_range_async(
      std::launch policy,
      K const& key) const {
    return const_iterator::equal_range_async(*this, policy, key);
  }

  template<class K>
  constexpr iterator find(K const& key) {
    return iterator::find(*this, key);
  }

  template<class K>
  constexpr const_iterator find(K const& key) const {
    return const_iterator::find(*this, key);
  }

  template<class K>
  constexpr iterator find_async(
      std::launch policy,
      K const& key) {
    return iterator::find_async(*this, policy, key);
  }

  template<class K>
  constexpr const_iterator find_async(
      std::launch policy,
      K const& key) const {
    return const_iterator::find_async(*this, policy, key);
  }

  void optimize() {
    if (layers_.empty()) {
      return;
    }
    layer_iterator curr_layer = layers_.end();
    while (curr_layer != layers_.begin()) {
      layer_iterator prev_layer = std::prev(curr_layer);
      mergeLayers(prev_layer, curr_layer);
      curr_layer = prev_layer;
    }
  }

 protected:
  friend class FunnelTest;

  /**
   *  @brief
   *  Iterator types for entries in `FunnelBase`.
   *
   *  The template argument `constant` specifies whether it is a const or a
   *  non-const variant.
   *
   *  An iterator consists of a list of `HorizontalIterator`s, one for each
   *  layer. This list is stored as a priority queue so that its top element
   *  can be quickly accessed and replaced.
   *  When the iterator is dereferenced, the entry associated with the top
   *  element in the priority queue will be returned.
   *  This entry will be the smallest entry among all entries that are
   *  associated to `HorizontalIterator` objects in the priority queue.
   *  (See `CompareHorizontalIterator` for more detail.)
   */
  template<bool constant>
  class Iterator {
   protected:
    friend funnel_type;
    friend class FunnelTest;

    using this_type = Iterator<constant>;

    using container_type = std::conditional_t<constant,
        funnel_type const,
        funnel_type>;
    using layer_iterator = std::conditional_t<constant,
        typename layer_list_type::const_iterator,
        typename layer_list_type::iterator>;

    /**
     *  @brief
     *  Iterator types within a layer.
     *
     *  This struct contains an iterator to a layer and an index of a cell in
     *  the layer.
     */
    struct HorizontalIterator {
      layer_iterator layer;
      size_type cell_index;
      HorizontalIterator(
          layer_iterator const& layer,
          size_type cell_index)
        : layer{layer}, cell_index{cell_index} {}
      HorizontalIterator() = default;
      HorizontalIterator(HorizontalIterator const&) = default;
      HorizontalIterator(HorizontalIterator&&) = default;
      HorizontalIterator& operator=(HorizontalIterator const&) = default;
      HorizontalIterator& operator=(HorizontalIterator&&) = default;
    };

    /**
     *  @brief
     *  Allocator for `HorizontalIterator` objects.
     *
     *  An iterator contains a `std::vector` whose elements are of type
     *  `HorizontalIterator`. The allocator for this `std::vector` will be a
     *  result of rebinding the layer allocator from `FunnelBase`.
     */
    using HorizontalIteratorAllocator =
        typename std::allocator_traits<layer_allocator_type>::
        template rebind_alloc<HorizontalIterator>;
    /// `std::vector<HorizontalIterator, HorizontalIteratorAllocator>`.
    using HorizontalIteratorList =
        std::vector<HorizontalIterator, HorizontalIteratorAllocator>;

    /**
     *  @brief
     *  Binary operator that compares values in `HorizontalIterator`s.
     *
     *  This operator compares keys in two `FunnelCell`s. It does not check if
     *  the cell is live or dead.
     *
     *  Note that the comparison order is reversed because we want to extract
     *  the minimum element with `std::priority_queue::top()` and
     *  `std::priority_queue::pop()`, while `key_comp` is a *less than*
     *  operator.
     */
    struct CompareHorizontalIterator {
      cell_compare cell_comp;
      constexpr CompareHorizontalIterator(cell_compare const& cell_comp)
        : cell_comp{cell_comp} {}
      constexpr bool operator()(
          HorizontalIterator const& a,
          HorizontalIterator const& b) const {
        if (a.cell_index >= a.layer->deque.size()) {
          return b.cell_index < b.layer->deque.size();
        }
        if (b.cell_index >= b.layer->deque.size()) {
          return false;
        }
        return cell_comp(
            b.layer->deque[b.cell_index],
            a.layer->deque[a.cell_index]);
      }
    };

    Iterator(
        HorizontalIteratorList&& horz_it_list,
        size_type index,
        cell_compare const& cell_comp)
      : queue_{
          CompareHorizontalIterator{cell_comp},
          std::move(horz_it_list)},
        index_{index} {}

    // --- Private member variables --- //
    std::priority_queue<
        HorizontalIterator,
        HorizontalIteratorList,
        CompareHorizontalIterator> queue_;
    size_type index_;

   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = typename funnel_type::difference_type;
    using value_type = std::conditional_t<constant,
        typename funnel_type::value_type const,
        typename funnel_type::value_type>;
    using pointer = std::conditional_t<constant,
        typename funnel_type::const_pointer,
        typename funnel_type::pointer>;
    using reference = std::conditional_t<constant,
        typename funnel_type::const_reference,
        typename funnel_type::reference>;

    Iterator(this_type const&) = default;
    Iterator(this_type&&) = default;
    this_type& operator=(this_type const&) = default;
    this_type& operator=(this_type&&) = default;

    reference operator*() const {
      auto& horz_it = queue_.top();
      return horz_it.layer->deque[horz_it.cell_index].entry;
    }

    pointer operator->() const {
      return &*this;
    }

    constexpr bool operator==(this_type const& other) const {
      return index_ == other.index_;
    }

    constexpr bool operator!=(this_type const& other) const {
      return index_ != other.index_;
    }

    this_type& operator++() {
      HorizontalIterator horz_it{queue_.top()};
      queue_.pop();
      auto& layer = horz_it.layer;
      auto& deque = layer->deque;
      auto& cell_index = horz_it.cell_index;
      assert((deque[cell_index].state.index() == 0));
      ++index_;
      ++cell_index;
      if (cell_index < deque.size() && deque[cell_index].state.index() != 0) {
        size_type right_live_index = layer->rightLiveIndex(cell_index);
        assert((right_live_index >= cell_index));
        index_ += right_live_index - cell_index;
        cell_index = right_live_index;
      }
      queue_.push(horz_it);
      return *this;
    }

    this_type& operator--() {
      HorizontalIterator horz_it{queue_.top()};
      queue_.pop();
      auto& layer = horz_it.layer;
      auto& deque = layer->deque;
      auto& cell_index = horz_it.cell_index;
      assert((index_ > 0));
      assert((cell_index > 0));
      assert((deque[cell_index].state.index() == 0));
      --index_;
      --cell_index;
      if (deque[cell_index].state.index() != 0) {
        size_type left_live_index = layer->leftLiveIndex(cell_index);
        assert((index_ + left_live_index >= cell_index));
        index_ -= cell_index - left_live_index;
        cell_index = left_live_index;
      }
      queue_.push(horz_it);
      return *this;
    }

    this_type operator++(int) {
      this_type old{*this};
      ++(*this);
      return old;
    }

    this_type operator--(int) {
      this_type old{*this};
      --(*this);
      return old;
    }

   protected:
    /**
     *  @brief
     *  Generic function for constructing an iterator.
     *
     *  This function is a template for search functions.
     *
     *  The input `layer_func` must be callable with the following prototype:
     *
     *      size_type layer_func(deque_type const&);
     *
     *  `layer_func` will be called for each layer. Its return value will be
     *  stored as `cell_index` in the corresponding `HorizontalIterator`.
     */
    template<class LayerFunc>
    static constexpr this_type build(
        container_type& container,
        LayerFunc layer_func) {
      size_type const num_layers = container.layers_.size();
      HorizontalIteratorList horz_it_list{
          num_layers,
          HorizontalIteratorAllocator{container.value_alloc_}};
      size_type index{0};
      size_type layer_index{0};
      for (layer_iterator layer = container.layers_.begin();
           layer != container.layers_.end();
           ++layer) {
        HorizontalIterator& horz_it = horz_it_list[layer_index++];
        horz_it.layer = layer;
        deque_type const& deque = layer->deque;
        size_type const cell_index{layer_func(deque)};
        horz_it.cell_index = cell_index;
        index += cell_index;
      }
      return this_type{
          std::move(horz_it_list),
          index,
          container.cell_comp_};
    }

    /**
     *  @brief
     *  Generic function for constructing an iterator.
     *  This overload allows calls to `layer_func` to execute according to the
     *  specified policy.
     *
     *  @remark
     *  If `policy` is `std::launch::async`, `layer_func` must support
     *  concurrent calls.
     */
    template<class LayerFunc>
    static constexpr this_type build_async(
        container_type& container,
        std::launch policy,
        LayerFunc layer_func) {
      size_type const num_layers = container.layers_.size();
      HorizontalIteratorList horz_it_list{
          num_layers,
          HorizontalIteratorAllocator{container.value_alloc_}};
      std::vector<std::future<size_type>> tasks{num_layers};
      size_type index{0};
      size_type layer_index{0};
      for (layer_iterator layer = container.layers_.begin();
           layer != container.layers_.end();
           ++layer) {
        tasks[layer_index] = std::async(
            policy,
            [&layer_func,
             layer,
             &horz_it = horz_it_list[layer_index++]]() {
              horz_it.layer = layer;
              deque_type const& deque = layer->deque;
              size_type const cell_index{layer_func(deque)};
              horz_it.cell_index = cell_index;
              return cell_index;
            });
      }
      for (auto& task : tasks) {
        index += task.get();
      }
      return this_type{
          std::move(horz_it_list),
          index,
          container.cell_comp_};
    }

    static constexpr this_type begin(container_type& container) {
      return build(
          container,
          [](deque_type const&) { return size_type{0}; });
    }

    static constexpr this_type begin_async(
        container_type& container,
        std::launch policy) {
      return build_async(
          container, policy,
          [](deque_type const&) { return size_type{0}; });
    }

    static constexpr this_type end(container_type& container) {
      return build(
          container,
          [](deque_type const& deque) { return deque.size(); });
    }

    static constexpr this_type end_async(
        container_type& container,
        std::launch policy) {
      return build_async(
          container, policy,
          [](deque_type const& deque) { return deque.size(); });
    }

    template<class K>
    static constexpr size_type layer_lower_bound(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_comp) {
      size_type const cell_index{
          std::distance(
            deque.begin(),
            std::lower_bound(deque.begin(), deque.end(), key, cell_comp))};
      return cell_index < deque.size()
          ? layer_type::rightLiveIndex(deque, cell_index)
          : cell_index;
    }

    template<class K>
    static constexpr this_type lower_bound(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_comp = container.cell_comp](deque_type const& deque) {
            return layer_lower_bound(deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr this_type lower_bound_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_comp = container.cell_comp](deque_type const& deque) {
            return layer_lower_bound(deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr size_type layer_upper_bound(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_comp) {
      size_type const cell_index{
          std::distance(
            deque.begin(),
            std::upper_bound(deque.begin(), deque.end(), key, cell_comp))};
      return cell_index < deque.size()
          ? layer_type::rightLiveIndex(deque, cell_index)
          : cell_index;
    }

    template<class K>
    static constexpr this_type upper_bound(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_comp = container.cell_comp](deque_type const& deque) {
            return layer_upper_bound(deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr this_type upper_bound_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_comp = container.cell_comp](deque_type const& deque) {
            return layer_upper_bound(deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr std::pair<this_type, this_type> equal_range(
        container_type& container,
        K const& key) {
      return {lower_bound(container, key), upper_bound(container, key)};
    }

    template<class K>
    static constexpr std::pair<this_type, this_type> equal_range_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      std::future<this_type> lower{std::async(
          policy,
          [policy, &container, &key]() {
            return lower_bound_async(container, policy, key);
          })};
      std::future<this_type> upper{std::async(
          policy,
          [policy, &container, &key]() {
            return upper_bound_async(container, policy, key);
          })};
      return {lower.get(), upper.get()};
    }

    template<class K>
    static constexpr this_type layer_find(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_comp) {
      size_type const cell_index{
          std::distance(
            deque.begin(),
            std::lower_bound(deque.begin(), deque.end(), key, cell_comp))};
      if (cell_index >= deque.size()) {
        return deque.size();
      }
      size_type const live_cell_index{
          layer_type::rightLiveIndex(deque, cell_index)};
      if (cell_comp.compare(cell_comp.ext_key(deque[live_cell_index].entry), key)
          == 0) {
        return live_cell_index;
      }
      return deque.size();
    }

    template<class K>
    static constexpr this_type find(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_comp = container.cell_comp](deque_type const& deque) {
            return layer_find(deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr size_type find_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_comp = container.cell_comp](deque_type const& deque) {
            return layer_find(deque, key, cell_comp);
          });
    }

    HorizontalIterator const& top() const {
      return queue_.top();
    }

    HorizontalIterator& top() {
      return queue_.top();
    }

  }; // class Iterator

  template<bool constant>
  static constexpr Iterator<constant> begin(
      typename Iterator<constant>::container_type& container) {
    return Iterator<constant>::begin(container);
  }

  template<bool constant>
  static constexpr Iterator<constant> begin_async(
      typename Iterator<constant>::container_type& container,
      std::launch policy) {
    return Iterator<constant>::begin_async(container, policy);
  }

  template<bool constant>
  static constexpr Iterator<constant> end(
      typename Iterator<constant>::container_type& container) {
    return Iterator<constant>::end(container);
  }

  template<bool constant>
  static constexpr Iterator<constant> end_async(
      typename Iterator<constant>::container_type& container,
      std::launch policy) {
    return Iterator<constant>::end_async(container, policy);
  }

  template<bool constant>
  static constexpr Iterator<constant> lower_bound(
      typename Iterator<constant>::container_type& container,
      key_type const& key) {
    return Iterator<constant>::lower_bound(container, key);
  }

  template<bool constant>
  static constexpr Iterator<constant> lower_bound_async(
      typename Iterator<constant>::container_type& container,
      std::launch policy,
      key_type const& key) {
    return Iterator<constant>::lower_bound_async(container, policy, key);
  }

  template<bool constant>
  static constexpr Iterator<constant> upper_bound(
      typename Iterator<constant>::container_type& container,
      key_type const& key) {
    return Iterator<constant>::upper_bound(container, key);
  }

  template<bool constant>
  static constexpr Iterator<constant> upper_bound_async(
      typename Iterator<constant>::container_type& container,
      std::launch policy,
      key_type const& key) {
    return Iterator<constant>::upper_bound_async(container, policy, key);
  }

  template<bool constant>
  static constexpr Iterator<constant> find(
      typename Iterator<constant>::container_type& container,
      key_type const& key) {
    return Iterator<constant>::find(container, key);
  }

  template<bool constant>
  static constexpr Iterator<constant> find_async(
      typename Iterator<constant>::container_type& container,
      std::launch policy,
      key_type const& key) {
    return Iterator<constant>::find_async(container, policy, key);
  }

  /**
   *  @brief
   *  Pointer to a layer and one of its end.
   */
  struct LayerEnd {
    layer_iterator layer;
    bool front;
    LayerEnd(layer_iterator layer, bool front) : layer{layer}, front{front} {}
  };

  /**
   *  @brief
   *  Where to insert a new entry.
   *
   *  This is the return type of `insertPositionLinearSearch()` and
   *  `insertPositionBinarySearch()`.
   *
   *  The empty state means that a new layer needs to be created.
   *  Otherwise, the value contains the layer iterator and a flag whether an
   *  insertion should take place in the front or in the back of the layer.
   */
  using InsertPosition = std::optional<LayerEnd>;

  /**
   *  @brief
   *  Finds a place to insert an element with a given `key`.
   *
   *  This function assumes that duplicate keys are allowed.
   *
   *  @remark
   *  If `layer_list_type` supports fast random access, a binary search can be
   *  done instead. We do not do that here.
   */
  template<class K>
  constexpr InsertPosition insertPositionLinearSearch(K const& key) {
    layer_iterator layer = layers_.begin();
    for (; layer != layers_.end(); ++layer) {
      if (!cell_comp_(key, layer->deque.back())) {
        return InsertPosition{std::in_place, layer, false};
      }
      if (!cell_comp_(layer->deque.front(), key)) {
        return InsertPosition{std::in_place, layer, true};
      }
    }
    return {};
  }

  /**
   *  @brief
   *  Same as `insertPositionLinearSearch()`, but use binary search.
   *
   *  @remark
   *  This should not be used when `layer_list_type` does not support fast
   *  random access.
   */
  template<class K>
  constexpr InsertPosition insertPositionBinarySearch(K const& key) {
    size_type const num_layers = layers_.size();
    if (num_layers == 0) {
      return {};
    }
    assert((!layers_.back().deque.empty()));
    if (!cell_comp_(key, layers_.back().deque.back())) {
      layer_iterator layer{
          std::lower_bound(layers_.begin(), layers_.end(), key,
            [&cell_comp = cell_comp_](layer_type const& layer, K const& k) {
              return cell_comp(k, layer.deque.back());
            }
          )};
      if (layer == layers_.end()) {
        return {};
      }
      return InsertPosition{std::in_place, layer, false};
    }
    if (!cell_comp_(layers_.back().deque.front(), key)) {
      layer_iterator layer{
          std::lower_bound(layers_.begin(), layers_.end(), key,
            [&cell_comp = cell_comp_](layer_type const& layer, K const& k) {
              return cell_comp(layer.deque.front(), k);
            })};
      if (layer == layers_.end()) {
        return {};
      }
      return InsertPosition{std::in_place, layer, true};
    }
    return {};
  }

  template<class K>
  constexpr InsertPosition insertPosition(
      K const& key,
      bool binary_search = true) {
    return binary_search ?
        insertPositionBinarySearch(key) :
        insertPositionLinearSearch(key);
  }

  template<class... Args>
  layer_iterator forceEmplaceAt(
      InsertPosition const& insert_position,
      Args&&... args) {
    ++size_;
    if (insert_position.has_value()) {
      auto& layer_insert_position = insert_position.value();
      if (layer_insert_position.front) {
        layer_insert_position.layer->emplace_front(
            std::forward<Args>(args)...);
      } else {
        layer_insert_position.layer->emplace_back(
            std::forward<Args>(args)...);
      }
      return insert_position.value().layer;
    }
    layers_.emplace_back(cell_alloc_).emplace_back(
        std::forward<Args>(args)...);
    return std::prev(layers_.end());
  }

  template<class E>
  layer_iterator forceInsert(
      E&& entry, bool binary_search = true) {
    InsertPosition insert_position{
        insertPosition(cell_comp_.ext_key(entry), binary_search)};
    return forceEmplaceAt(insert_position, std::forward<E>(entry));
  }

  template<class... Args>
  layer_iterator forceEmplace(
      Args&&... args, bool binary_search = true) {
    Entry entry{std::forward<Args>(args)...};
    return forceInsert(std::move(entry), binary_search);
  }

  template<class K, class... VArgs>
  layer_iterator forceEmplacePair(
      K&& k,
      VArgs&&... v_args,
      bool binary_search = true) {
    Key key{std::forward<K>(k)};
    InsertPosition insert_position{insertPosition(key, binary_search)};
    return forceEmplaceAt(
        insert_position,
        std::move(key),
        std::forward<VArgs>(v_args)...);
  }

  template<class... KArgs, class... VArgs>
  layer_iterator forceEmplacePair(
      std::piecewise_construct_t,
      std::tuple<KArgs...> k_args,
      std::tuple<VArgs...> v_args,
      bool binary_search = true) {
    Key key{std::make_from_tuple<Key>(k_args)};
    InsertPosition insert_position{insertPosition(key, binary_search)};
    return forceEmplaceAt(
        insert_position,
        std::piecewise_construct,
        std::forward_as_tuple(std::move(key)),
        v_args);
  }

  template<class... Args>
  void multiEmplace(Args&&... args, bool binary_search = true) {
    layer_iterator layer{
        forceEmplace(std::forward<Args>(args)..., binary_search)};
    tryMergeOnInsert(layer);
  }

  /**
   *  @brief
   *  Merges `layer0` with `layer1`, then erases `layer1` from `layers_`.
   *  This function is supposed to be called by `merge_policy`.
   *
   *  `layer0` should point to a layer that is positioned before `layer1` in
   *  `layers_`.
   *
   *  If `layer1` is `layers_.end()`, this function simply removes all the dead
   *  cells in `layer0`.
   *
   *  Note: When `layer1` is not `layers_.end()`, calling
   *  `layers_.erase(layer1)` should not invalidate `layer0` as the merger may
   *  rely on this. This is the case when `LayerList` is `std::list`, or when
   *  `LayerList` is `std::vector` and `layer0` is positioned before `layer1`
   *  in `layers_`.
   */
  void mergeLayers(layer_iterator layer0, layer_iterator layer1) {
    if (layer1 == layers_.end()) {
      layer0->optimize();
      return;
    }
    deque_type& deque0{layer0->deque};
    deque_type& deque1{layer1->deque};
    size_t index0{0};
    size_t index1{0};
    layer_type merged_layer{cell_alloc_};
    while (true) {
      if (index0 == deque0.size()) {
        for (
            ; index1 < deque1.size()
            ; index1 = layer_type::nextLiveIndex(deque1, index1)) {
          merged_layer.push_back(std::move(deque1[index1]));
        }
        break;
      }
      if (index1 == deque1.size()) {
        for (
            ; index0 < deque0.size()
            ; index0 = layer_type::nextLiveIndex(deque0, index0)) {
          merged_layer.push_back(std::move(deque0[index0]));
        }
        break;
      }
      if (cell_comp_(deque0[index0], deque1[index1])) {
        merged_layer.push_back(std::move(deque0[index0]));
        index0 = layer_type::nextLiveIndex(deque0, index0);
      } else {
        merged_layer.push_back(std::move(deque1[index1]));
        index1 = layer_type::nextLiveIndex(deque1, index1);
      }
    }
    layer0->swap(merged_layer);
    layers_.erase(layer1);
  }

  // --- Private member variables --- //

  // del_ent_(value_type&)
  delete_entry del_ent_;

  // cell_comp_ contains ext_key and key_comp
  cell_compare cell_comp_;

  // Allocators.
  allocator_type value_alloc_;
  layer_allocator_type layer_alloc_;
  cell_allocator_type cell_alloc_;

  // Merge policy.
  merger_type merger_;

  // List of layers.
  layer_list_type layers_;

  // Number of cells. `iterator::index_` will not exceed `num_cells_`.
  size_type num_cells_{0};

  // Number of active (not deleted) cells. `size()` will return this value.
  size_type size_{0};
};

template<class Entry>
struct DefaultDeleteEntry {
  using entry_type = Entry;
  constexpr void operator()(entry_type& entry) const {}
};

// `std::identity` in C++20 could be used here. (The member typedefs are not
// used.)
template<class Key>
struct ExtractKeyPassthrough {
  using key_type = Key;
  using value_type = Key const;
  constexpr key_type const& operator()(value_type const& v) const {
    return v;
  }
};

template<class Key, class Value>
struct ExtractKeyFromPair {
  using key_type = Key;
  using mapped_type = Value;
  using value_type = std::pair<key_type const, mapped_type>;
  constexpr key_type const& operator()(value_type const& v) const {
    return v.first;
  }
};

template<
    class Key,
    class Compare = std::less<Key>,
    class Allocator = std::allocator<Key>,
    class DeleteEntry = DefaultDeleteEntry<Key>,
    template<class, class> class Deque = std::deque,
    template<class, class> class LayerList = std::vector,
    template<class> class Merger = SimpleRatioMergerCurried<1, 2>::type>
class FunnelSet : public FunnelBase<
    Key,
    DefaultDeleteEntry<Key>,
    Key,
    ExtractKeyPassthrough<Key>,
    Compare,
    Allocator,
    Deque,
    LayerList,
    Merger> {
 public:
  using this_type = FunnelSet<
      Key,
      Compare,
      Allocator,
      DeleteEntry,
      Deque,
      LayerList,
      Merger>;
  using super_type = FunnelBase<
      Key,
      DefaultDeleteEntry<Key>,
      Key,
      ExtractKeyPassthrough<Key>,
      Compare,
      Allocator,
      Deque,
      LayerList,
      Merger>;

  using key_type = typename super_type::key_type;
  using delete_entry = typename super_type::delete_entry;
  using extract_key = typename super_type::extract_key;
  using key_compare = typename super_type::key_compare;
  using allocator_type = typename super_type::allocator_type;
  using merger_type = typename super_type::merger_type;

  // Inherit the constructor from `FunnelBase`.
  using super_type::super_type;

/*
  constexpr FunnelSet(
      delete_entry const& del_ent = delete_entry(),
      extract_key const& ext_key = extract_key(),
      key_compare const& key_comp = key_compare(),
      allocator_type const& alloc = allocator_type(),
      merger_type const& merger = merger_type())
    : super_type{del_ent, ext_key, key_comp, alloc, merger} {}*/

  friend class FunnelTest;
};

} // namespace funnel

