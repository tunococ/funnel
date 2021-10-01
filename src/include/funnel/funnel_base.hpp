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
#include <set>
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
 *
 *  @tparam Entry
 *    Type of the entry.
 *    `Entry` will be assigned to the member type `entry_type`.
 *  @tparam SizeType
 *    Type of *sizes*.
 *    `SizeType` will be assigned to the member type `size_type`.
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
     *  Distance to the beginning index of the interval.
     */
    size_type left;
    /**
     *  @brief
     *  Distance to the pass-the-end index of the interval.
     *
     *  Note that this value is always greater than `0`.
     *  The length of interval is `left + right`.
     */
    size_type right;
    /// Rank of the root--for the union-by-rank operation.
    size_type rank;

    Root(size_type left = 0, size_type right = 1, size_type rank = 0)
      : left{left}, right{right}, rank{rank} {}
    Root() = default;
    Root(Root const&) = default;
    Root(Root&&) = default;
    Root& operator=(Root const&) = default;
    Root& operator=(Root&&) = default;
  };

  /// Value of the entry.
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

  /// Returns true if and only if the cell is live.
  constexpr bool isLive() const {
    return state.index() == 0;
  }

  /// Returns true if and only if the cell is dead.
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
 *  `FunnelLayer` keeps track of the number of live cells.
 *
 *  @tparam Entry
 *    Main storage type of *entries* in the funnel.
 *    `Entry` will be assigned to the member type `entry_type`.
 *  @tparam Allocator
 *    Allocator type for `Entry`.
 *    `Allocator` will be assigned to the member type `entry_allocator_type`.
 *    This allocator type must support rebinding for type
 *    `Cell<entry_type, size_type>`, which will be assigned to the member type
 *    `allocator_type`.
 *  @tparam Deque
 *    Functor of kind `* -> * -> *`.
 *    `Deque<cell_type, allocator_type>` will be assigned to the member type
 *    `deque_type`, which is expected to be similar to
 *    `std::deque<cell_type, allocator_type>`.
 *    Note, however, that we do not need iterator guarantees provided by
 *    `std::deque` because we will only access an element with an index,
 *    `front()`, or `back()`.
 *    For example, a growable circular array can be used here to improve
 *    access locality.
 *  @tparam Cell
 *    Functor of kind `* -> * -> *`.
 *    `Cell<entry_type, size_type>` will be assigned to the member type
 *    `cell_type`, which is expected to be similar to
 *    `FunnelCell<entry_type, size_type>`.
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
  /** 
   *  @brief
   *  Type of `index`. The funnel will generate layer indices for layers in an
   *  increasing order. `layer_index_type` is expected to never overflow
   *  during the lifetime of a funnel.
   */
  using layer_index_type = uint64_t;

  // Static assertions
  static_assert(std::is_same<size_type, typename cell_type::size_type>::value);

  // Constructor
  constexpr FunnelLayer(
      layer_index_type index = 0,
      allocator_type const& alloc = allocator_type())
    : index{index}, deque(alloc) {}

  constexpr FunnelLayer(this_type const&) = default;
  constexpr FunnelLayer(this_type&&) = default;
  constexpr this_type& operator=(this_type const&) = default;
  constexpr this_type& operator=(this_type&&) = default;

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
    assert((deque[index].state.index() != 0));
    while (true) {
      auto& state = deque[index].state;
      if (state.index() == 1) {
        return index;
      }
      size_type parent_index{index + std::get<2>(state)};
      auto& parent_state = deque[parent_index].state;

      if (parent_state.index() == 2) {
        using offset_type = typename cell_type::offset_type;
        // If parent is not a root, perform 1-step path compression.
        state.template emplace<2>(
            static_cast<offset_type>(parent_index + std::get<2>(parent_state))
            - static_cast<offset_type>(index));
      }
      index = parent_index;
    }
  }

  /**
   *  @brief
   *  Calls `findRootIndex(deque, index)` where `deque` is the member variable
   *  of `FunnelLayer`.
   */
  constexpr size_type findRootIndex(size_type index) const {
    return findRootIndex(deque, index);
  }

  /**
   *  @brief
   *  Returns `index` if it points to a live cell in `deque`, or returns the
   *  past-the-end index (to the right) of the interval of dead cells that
   *  contains the cell pointed to by `index`.
   */
  static constexpr size_type rightDeadEndIndex(
      deque_type const& deque,
      size_type index) {
    if (index >= deque.size() || deque[index].isLive()) {
      return index;
    }
    size_type root_index{findRootIndex(deque, index)};
    return root_index + std::get<1>(deque[root_index].state).right;
  }

  /**
   *  @brief
   *  Calls `rightDeadEndIndex(deque, index)` where `deque` is the member
   *  variable of `FunnelLayer`.
   */
  constexpr size_type rightDeadEndIndex(size_type index) const {
    return rightDeadEndIndex(deque, index);
  }

  /**
   *  @brief
   *  Returns the index of the live cell that is closest to the right of the
   *  given `index`. If `index` points to the last live cell, `deque.size()`
   *  will be returned.
   *
   *  This function can be used for enumerating all live cells.
   */
  static constexpr size_type nextLiveIndex(
      deque_type const& deque,
      size_type index) {
    size_t next_index{index + 1};
    if (next_index >= deque.size()) {
      return deque.size();
    }
    return rightDeadEndIndex(deque, next_index);
  }

  /**
   *  @brief
   *  Calls `nextLiveIndex(deque, index)` where `deque` is the member variable
   *  of `FunnelLayer`.
   */
  constexpr size_type nextLiveIndex(size_type index) const {
    return nextLiveIndex(deque, index);
  }

  /**
   *  @brief
   *  Returns the highest index of a cell on the left of the cell pointed to by
   *  `index` whose left neighbor is a live cell.
   *
   *  If `index` is `0`, `0` is returned.
   *  If a non-zero value is returned, the return value minus one will point to
   *  a live cell.
   *  In other words, negative indices correspond to imaginary live cells.
   */
  static constexpr size_type leftDeadBeginIndex(
      deque_type const& deque,
      size_type index) {
    if (index == 0 || deque[index - 1].isLive()) {
      return index;
    }
    size_type root_index{findRootIndex(deque, index - 1)};
    return root_index - std::get<1>(deque[root_index].state).left;
  }

  /**
   *  @brief
   *  Calls `leftDeadBeginIndex(deque, index)` where `deque` is the member
   *  variable of `FunnelLayer`.
   */
  constexpr size_type leftDeadBeginIndex(size_type index) const {
    return leftDeadBeginIndex(deque, index);
  }

  /**
   *  @brief
   *  Returns the index of the live cell that is closest to the left of the
   *  given `index`. If `index` points to the first live cell, no value will be
   *  returned.
   *
   *  This function can be used for enumerating all live cells.
   */
  static constexpr std::optional<size_type> prevLiveIndex(
      deque_type const& deque,
      size_type index) {
    size_type left_dead_index{leftDeadBeginIndex(deque, index)};
    if (left_dead_index == 0) {
      return {};
    } else {
      return {left_dead_index - 1};
    }
  }

  /**
   *  @brief
   *  Calls `prevLiveIndex(deque, index)` where `deque` is the member variable
   *  of `FunnelLayer`.
   */
  constexpr std::optional<size_type> prevLiveIndex(size_type index) const {
    return prevLiveIndex(deque, index);
  }

  /**
   *  @brief
   *  Constructs a cell at the front of the deque. 
   */
  template<class... Args>
  constexpr void emplace_front(Args&&... args) {
    ++num_live;
    deque.emplace_front(std::forward<Args>(args)...);
  }

  /**
   *  @brief
   *  Constructs a cell at the back of the deque. 
   */
  template<class... Args>
  constexpr void emplace_back(Args&&... args) {
    ++num_live;
    deque.emplace_back(std::forward<Args>(args)...);
  }

  /**
   *  @brief
   *  Adds a cell to the front of the deque. 
   */
  template<class CellType>
  constexpr void push_front(CellType&& cell) {
    ++num_live;
    deque.push_front(std::forward<CellType>(cell));
  }

  /**
   *  @brief
   *  Adds a cell to the back of the deque. 
   */
  template<class CellType>
  constexpr void push_back(CellType&& cell) {
    ++num_live;
    deque.push_back(std::forward<CellType>(cell));
  }

  /**
   *  @brief
   *  Removes all dead cells, and optionally tracks a location of one entry.
   *  This function returns `false` if and only if `deque` is empty.
   *
   *  If `track_entry` is not null, its value is an index of the entry to
   *  track, and after `optimize()` finishes, its value may be changed to the
   *  new index of the same entry.
   *
   *  Note that if the leftmost or the rightmost cell is dead, `optimize()`
   *  will remove it. As a result, the range of keys represented by this layer
   *  will change.
   */
  constexpr bool optimize(size_type* tracked_index = nullptr) {
    // If there are no dead cells, do nothing.
    if (deque.size() == 0) {
      return false;
    }
    if (num_live == 0) {
      deque.clear();
      return false;
    }
    size_type i{0};
    // `i` must be smaller than `num_live` because there is a dead cell.
    for (; deque[i].isLive(); ++i) {}
    size_type j{i};
    while (i < num_live) {
      j = rightDeadEndIndex(j);
      assert((j <= deque.size()));
      if (tracked_index && *tracked_index == j) {
        *tracked_index = i;
      }
      deque[i].entry = std::move(deque[j].entry);
      deque[i].state.template emplace<0>();
      ++i;
      ++j;
    }
    // If `tracked_index` is beyond the last live cell, we update it to the
    // past-the-end index.
    if (tracked_index && *tracked_index >= j) {
      *tracked_index = num_live;
    }
    deque.resize(num_live);
    return true;
  }

  /**
   *  @brief
   *  Returns the number of all cells (dead and live).
   */
  constexpr size_type size() const {
    return deque.size();
  }

  constexpr bool empty() const {
    return deque.empty();
  }

  /**
   *  @brief
   *  Returns the number of live cells.
   */
  constexpr size_type live() const {
    return num_live;
  }

  /**
   *  @brief
   *  Deletes an entry at the given `index`.
   *
   *  This function marks the cell dead and updates states of adjacent dead
   *  cells accordingly.
   *
   *  The caller of this function must make sure that the given `index` points
   *  to a live cell prior to calling it.
   */
  void eraseAtIndex(size_type index) {
    --num_live;
    auto& state = deque[index].state;
    assert((state.index() == 0));

    using offset_type = typename cell_type::offset_type;

    // The cell to delete does not have a cell on the left.
    if (index == 0) {
      if (deque.size() == 1) {
        state.template emplace<1>(0, 1, 0);
        return;
      }
      if (deque[index + 1].isLive()) {
        state.template emplace<1>(0, 1, 0);
        return;
      }
      size_type right_root_index{findRootIndex(index + 1)};
      state.template emplace<2>(
          static_cast<offset_type>(right_root_index) -
          static_cast<offset_type>(index));
      // Update the 'left' member of the root's state.
      auto& right_root_state = deque[right_root_index].state;
      auto& right_root = std::get<1>(right_root_state);
      ++right_root.left;
      return;
    }

    // The cell to delete does not have a cell on the right.
    if (index == deque.size() - 1) {
      if (deque[index - 1].isLive()) {
        state.template emplace<1>(0, 1, 0);
        return;
      }
      size_type left_root_index{findRootIndex(index - 1)};
      state.template emplace<2>(
          static_cast<offset_type>(left_root_index) -
          static_cast<offset_type>(index));
      // Update the 'right' member of the root's state.
      auto& left_root_state = deque[left_root_index].state;
      auto& left_root = std::get<1>(left_root_state);
      ++left_root.right;
      return;
    }

    // The cell to delete is somewhere in the middle.
    if (deque[index + 1].isLive()) {
      if (deque[index - 1].isLive()) {
        // Isolated deleted cell becomes a new root.
        state.template emplace<1>(0, 1, 0);
        return;
      }

      // Add the deleted cell to the interval on the left.
      size_type left_root_index{findRootIndex(index - 1)};
      state.template emplace<2>(
          static_cast<offset_type>(left_root_index) -
          static_cast<offset_type>(index));
      // Update the `right` member of the root's state.
      auto& left_root_state = deque[left_root_index].state;
      auto& left_root = std::get<1>(left_root_state);
      ++left_root.right;
      return;
    }

    // Add the deleted cell to the interval on the right.
    size_type right_root_index{findRootIndex(index + 1)};
    state.template emplace<2>(
        static_cast<offset_type>(right_root_index) -
        static_cast<offset_type>(index));
    // Update the `left` member of the root's state.
    auto& right_root_state = deque[right_root_index].state;
    auto& right_root = std::get<1>(right_root_state);
    ++right_root.left;

    // If there are no dead cells on the left, we are done.
    if (deque[index - 1].isLive()) {
      return;
    }

    // Merge intervals on the left and on the right.
    size_type left_root_index{findRootIndex(index - 1)};
    auto& left_root_state = deque[left_root_index].state;
    auto& left_root = std::get<1>(left_root_state);

    if (left_root.rank < right_root.rank) {
      // Keep the right root. Extend its `left` to cover the left interval.
      right_root.left += left_root.left + left_root.right;
      // Make the left root a child of the right root.
      left_root_state.template emplace<2>(
          static_cast<offset_type>(right_root_index) -
          static_cast<offset_type>(left_root_index));
      return;
    }

    // Keep the left root.
    if (left_root.rank == right_root.rank) {
      // If the ranks match, increase the rank of the left root by 1.
      ++left_root.rank;
    }
    // Extend the `right` part of the left root to cover the right interval.
    left_root.right += right_root.left + right_root.right;
    // Make the right root a child of the left root.
    right_root_state.template emplace<2>(
        static_cast<offset_type>(left_root_index) -
        static_cast<offset_type>(right_root_index));
  }

  /**
   *  @brief
   *  Swaps with another layer.
   */
  constexpr void swap(this_type& other) {
    deque.swap(other.deque);
    std::swap(index, other.index);
    std::swap(num_live, other.num_live);
  }

  // Member variables

  /**
   *  @brief
   *  Index of this layer.
   *
   *  Each layer in a funnel should have a unique index.
   *  When entries with the same key exist in multiple layers, they are ordered
   *  by the indices of layers they belong to.
   */
  size_type index{0};

  /**
   *  @brief
   *  Sorted list of cells.
   */
  deque_type deque;

  /**
   *  @brief
   *  The number of live cells in this layer.
   *
   *  This is used for deciding when to merge layers.
   */
  size_type num_live{0};
};

/**
 *  @brief
 *  Merge policy based on a rational ratio `SMALLER / LARGER`.
 *
 *  A *merge policy* consists of two callback functions:
 *  - `mergeOnInsert()`: called after an element is added to the funnel.
 *  - `mergeOnErase()`: called after an element is removed from the funnel.
 *
 *  `SimpleRatioMerger` attempts to merge adjacent layers based on their sizes
 *  and number of live cells. It provides the following guarantees:
 *  - The number of live cells in the funnel is bounded below by
 *    `c * SMALLER / (SMALLER + LARGER)`; and
 *  - The number of layers in the funnel is bounded above by
 *    `log(c) / log(LARGER / SMALLER)`;
 *  where `c` is the number of all cells (including dead cells) in the funnel.
 *  Consequently, if we let `n` denote the number of live cells in the funnel
 *  (which is the `size()` of the funnel), then the number of layers is
 *  bounded above by
 *  `log(n * (SMALLER + LARGER) / SMALLER) / log(LARGER / SMALLER)`.
 *
 *  \remark
 *  If `LARGER = 0`, there will be no merging, and the guarantees mentioned
 *  above will not hold.
 */
template<std::size_t SMALLER, std::size_t LARGER, class FunnelType>
class SimpleRatioMerger {
 public:
  using this_type = SimpleRatioMerger<SMALLER, LARGER, FunnelType>;
  using funnel_type = FunnelType;
  using size_type = typename FunnelType::size_type;
  using layer_type = typename FunnelType::layer_type;
  using layer_index_type = typename FunnelType::layer_index_type;
  using layer_list_type = typename FunnelType::layer_list_type;
  using layer_iterator = typename FunnelType::layer_iterator;
  using const_layer_iterator = typename FunnelType::const_layer_iterator;

  using CellLocation = typename FunnelType::CellLocation;
  using LayerSet = typename FunnelType::LayerSet;

  static constexpr size_type smaller{static_cast<size_type>(SMALLER)};
  static constexpr size_type larger{static_cast<size_type>(LARGER)};

  void mergeOnInsert(
      FunnelType& funnel,
      LayerSet&& touched_layers,
      CellLocation* tracked_cell = nullptr) {
    if (touched_layers.size() == 0) {
      return;
    }

    using TouchedLayer = typename LayerSet::const_iterator;
    TouchedLayer curr_touched_layer{touched_layers.end()};
    std::optional<TouchedLayer> prev_touched_layer{};
    if (curr_touched_layer != touched_layers.begin()) {
      prev_touched_layer.emplace(std::prev(curr_touched_layer));
    }

    while (prev_touched_layer) {
      curr_touched_layer == *prev_touched_layer;
      if (curr_touched_layer != touched_layers.begin()) {
        prev_touched_layer.emplace(std::prev(curr_touched_layer));
      } else {
        prev_touched_layer.reset();
      }

      auto curr_layer = *curr_touched_layer;

      while (curr_layer != funnel.layers_begin()) {
        auto prev_layer = std::prev(curr_layer);
        if (!tryMergeOnInsert(funnel, prev_layer, curr_layer, tracked_cell)) {
          break;
        }
        curr_layer = prev_layer;
        if (prev_touched_layer && *prev_touched_layer == curr_layer) {
          break;
        }
      }
    }
  }

  void mergeOnErase(
      FunnelType& funnel,
      LayerSet&& touched_layers,
      CellLocation* tracked_cell = nullptr) {
    if (touched_layers.size() == 0) {
      return;
    }
    
    using TouchedLayer = typename LayerSet::const_iterator;
    TouchedLayer curr_touched_layer{touched_layers.end()};
    std::optional<TouchedLayer> prev_touched_layer{};
    if (curr_touched_layer != touched_layers.begin()) {
      prev_touched_layer.emplace(std::prev(curr_touched_layer));
    }

    bool previously_merged{false};
    while (prev_touched_layer) {
      curr_touched_layer == *prev_touched_layer;
      if (curr_touched_layer != touched_layers.begin()) {
        prev_touched_layer.emplace(std::prev(curr_touched_layer));
      } else {
        prev_touched_layer.reset();
      }

      auto curr_layer = *curr_touched_layer;
      auto next_layer = std::next(curr_layer);
      if (previously_merged ||
          tryMergeOnErase(funnel, curr_layer, next_layer, tracked_cell)) {
        previously_merged = false;
        while (curr_layer != funnel.layers_begin()) {
          auto prev_layer = std::prev(curr_layer);
          if (!tryMergeOnInsert(
              funnel, prev_layer, curr_layer, tracked_cell)) {
            break;
          }
          curr_layer = prev_layer;
          if (prev_touched_layer && *prev_touched_layer == curr_layer) {
            previously_merged = true;
            break;
          }
        }
      }
    }
  }

 private:

  bool tryMergeOnInsert(
      FunnelType& funnel,
      layer_iterator& prev_layer,
      layer_iterator& curr_layer,
      CellLocation* tracked_cell) {
    if (larger * curr_layer->live() <= smaller * prev_layer->size()) {
      return false;
    }
    funnel.mergeLayers(prev_layer, curr_layer, tracked_cell);
    return true;
  }

  bool tryMergeOnErase(
      FunnelType& funnel,
      layer_iterator& curr_layer, 
      layer_iterator& next_layer,
      CellLocation* tracked_cell) {
    if ((smaller + larger) * curr_layer->live() >
        smaller * curr_layer->size()) {
      return false;
    }
    funnel.mergeLayers(curr_layer, next_layer, tracked_cell);
    return true;
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
 *    `Entry` will be assigned to the member type `entry_type`.
 *    `entry_type` is the main storage type of *entries* in the funnel.
 *  @tparam DeleteEntry
 *    `DeleteEntry` will be assigned to the member type `delete_entry`.
 *    Since a funnel uses lazy deletion, a deleted entry may still hold its place
 *    in a funnel--the containing cell may simply be marked *dead*. This means
 *    the destructor of the entry might not be called right after it is deleted.
 *    To allow the user to perform an immediate optimize when an entry is marked
 *    deleted, the user can supply a *deletion callback* during the construction
 *    of a funnel. This callback must behave like a function with prototype
 *    `void(entry_type*)`. `delete_entry` is the type of this callback.
 *  @tparam Key
 *    `Key` will be assigned to the member type `key_type`.
 *    `key_type` is the type of *keys* in the funnel.
 *    For `FunnelSet` and `FunnelMultiset`, `key_type` will be the same as
 *    `entry_type`.
 *    For `FunnelMap` and `FunnelMultimap`, `entry_type` will be
 *    `std::pair<key_type, MappedType>`, where `MappedType` is the *mapped*
 *    type.
 *  @tparam ExtractKey
 *    `ExtractKey` will be assigned to the member type `extract_key`.
 *    For `FunnelSet` and `FunnelMultiset`, `extract_key` will be the type of
 *    an identity function for type `key_type const&`.
 *    For `FunnelMap` and `FunnelMultimap`, `extract_key` will be the type of
 *    a function that projects `std::pair<key_type, MappedType> const&` onto
 *    its `first` element, where `MappedType` is the *mapped* type.
 *  @tparam Compare
 *    `Compare` will be assigned to the member type `key_compare`.
 *    `key_compare` is the type of the *less* operator.
 *    It should be compatible with a function with prototype
 *    `bool(key_type const&, key_type const&)`.
 *    The function is expected to return `true` if its first operand is
 *    strictly smaller than the second operand when both are viewed as
 *    `key_type`.
 *  @tparam Allocator
 *    `Allocator` will be assigned to the member type `allocator_type`.
 *    This is the type of an allocator for `entry_type`.
 *    It should support rebinding for the following types: `layer_type`,
 *    `cell_type`, and `CellLocation`.
 *  @tparam Merger
 *    `Merger` should be a functor of kind `* -> *`.
 *    `Merger<this_type>` will be assigned to the member type `merger_type`.
 *    `merger_type` is expected to be a class or struct with the following
 *    public member functions:
 *    \code{.cpp}
 *        layer_iterator mergeOnInsert(
 *            this_type&, layer_iterator, TrackedCell*);
 *
 *        layer_iterator mergeOnErase(
 *            this_type&, layer_iterator, TrackedCell*);
 *    \endcode
 *    As a simple example, `Merger<this_type>` should behave like
 *    `SimpleRatioMerger<1, 2, this_type>`.
 *    \remark
 *    `Merger<this_type>` will have access to private members of `FunnelBase`
 *    such as `mergeLayers()` and `TrackedCell` because `FunnelBase` will
 *    declare `Merger<this_type>` as a friend.
 *  @tparam Deque
 *    Functor of kind `* -> * -> *`.
 *    `Deque` will be supplied as the third argument of `Layer`, then
 *    `layer_type::deque_type` will be assigned to the member type
 *    `deque_type`.
 *  @tparam LayerList
 *    Functor of kind `* -> * -> *`.
 *    `LayerList<layer_type, layer_allocator_type>` will be assigned to the
 *    member type `layer_list_type`, which is expected to be similar to
 *    `std::list<layer_type, layer_allocator_type>`.
 *    Apart from common container functions, `layer_list_type` must support
 *    - Bidirectional iterators.
 *    - `erase()` at a given iterator.
 *    - `emplace_back()`.
 *    Note that the theoretical time complexities of `erase()` and
 *    `emplace_back()` will affect theoretical time complexities of
 *    `FunnelBase`'s operations, but the effect may be small if we have a good
 *    bound on the number of layers.
 *    (`std::list` is theoretically optimal, but there may be data structures
 *    that are better in practice.)
 *  @tparam Cell
 *    Functor of kind `* -> * -> *`.
 *    `Cell` will be supplied as the fourth argument of `Layer`, which is used
 *    to defined the member type `layer_type`.
 *    `layer_type::cell_type` will be assigned to the member type `cell_type`.
 *  @tparam Layer
 *    Functor of kind `* -> * -> (* -> * -> *) -> (* -> * -> *) -> *`.
 *    `Layer<Entry, Allocator, Deque, Cell>` will be assigned to the member
 *    type `layer_type`, which is expected to behave like
 *    `FunnelLayer<Entry, Allocator, Deque, Cell>`.
 */
template<
    class Entry,
    class DeleteEntry,
    class Key,
    class ExtractKey,
    class Compare = std::less<Key>,
    class Allocator = std::allocator<Entry>,
    template<class> class Merger = SimpleRatioMergerCurried<1, 2>::type,
    template<class, class> class Deque = std::deque,
    template<class, class> class LayerList = std::list,
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
      Merger,
      Deque,
      LayerList,
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
  /// `layer_type::layer_index_type`.
  using layer_index_type = typename layer_type::layer_index_type;
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
   *    `merger.mergeOnInsert(*this, layer, tracked_entry)` will be called
   *    after an entry is added to `layer` of the funnel at `tracked_entry`,
   *    while `merger.mergeOnErase(*this, layer, tracked_entry)` will be
   *    called after an entry is removed from `layer` of the funnel.
   *    (`tracked_entry` will be the `TrackedCell` of an entry after the
   *    erased entry.)
   *    Both functions should return a `layer_iterator` pointing to the layer
   *    that eventually contains all the elements in the original input
   *    `layer`.
   *    `FunnelBase` will always supply a dereferenceable `layer` to these
   *    functions. (`layer` cannot be the past-the-end iterator.)
   *    However, `FunnelBase` may supply `nullptr` in `tracked_entry` when it
   *    does not want to track an entry location.
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
    num_live_ = 0;
  }

  constexpr size_type size() const {
    return num_live_;
  }

  constexpr bool empty() const {
    return num_live_ == 0;
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

  constexpr iterator end() {
    return iterator::end(*this);
  }

  constexpr const_iterator end() const {
    return const_iterator::end(*this);
  }

  constexpr const_iterator cend() const {
    return const_iterator::end(*this);
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
  constexpr std::optional<iterator> search(K const& key) {
    return iterator::search(*this, key);
  }

  template<class K>
  constexpr std::optional<const_iterator> search(K const& key) const {
    return const_iterator::search(*this, key);
  }

  template<class K>
  constexpr std::optional<iterator> search_async(
      std::launch policy, K const& key) {
    return iterator::search_async(*this, policy, key);
  }

  template<class K>
  constexpr std::optional<const_iterator> search_async(
      std::launch policy, K const& key) const {
    return const_iterator::search_async(*this, policy, key);
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
  constexpr iterator find_async(std::launch policy, K const& key) {
    return iterator::find_async(*this, policy, key);
  }

  template<class K>
  constexpr const_iterator find_async(std::launch policy, K const& key) const {
    return const_iterator::find_async(*this, policy, key);
  }

  template<class K>
  constexpr bool contains(K const& key) const {
    return const_iterator::search(*this, key).has_value();
  }

  template<class K>
  constexpr bool contains_async(std::launch policy, K const& key) const {
    return const_iterator::search_async(*this, policy, key).has_value();
  }

  /**
   *  @brief
   *  Merges all layers and removes all dead cells.
   */
  void optimize() {
    if (layers_.empty()) {
      return;
    }
    if (layers_.size() == 1) {
      auto& layer = layers_.front();
      layer.optimize();
      layer.index = 0;
      layer_index_counter_ = 1;
      return;
    }
    layer_iterator curr_layer{std::prev(layers_.end())};
    while (curr_layer != layers_.begin()) {
      layer_iterator prev_layer{std::prev(curr_layer)};
      mergeLayers(prev_layer, curr_layer, nullptr);
      curr_layer = prev_layer;
    }
    curr_layer->index = 0;
    layer_index_counter_ = 1;
  }

 protected:
  // Friend declarations
  friend merger_type;
  friend deque_type;
  friend layer_list_type;
  friend layer_type;
  friend cell_type;

  /**
   *  @brief
   *  Struct containing a reference to a cell.
   *
   *  `CellLocation` contains a layer iterator and a cell index.
   *
   *  The layer iterator may become invalidated if it is the second argument in
   *  a `mergeLayers()` call.
   */
  struct CellLocation {
    constexpr CellLocation(layer_iterator layer, size_type cell_index)
      : layer{layer}, cell_index{cell_index} {}

    constexpr layer_index_type const& layerIndex() const {
      return layer->index;
    }

    layer_iterator layer;
    size_type cell_index;
  };

  using CellLocationList = std::vector<CellLocation>;

  struct CompareLayerIndex {
    constexpr bool operator()(
        layer_iterator const& a,
        layer_iterator const& b) const {
      return a->index < b->index;
    }
  };

  /// List of unique `layer_iterator`s sorted by layer indices.
  using LayerSet = std::set<layer_iterator, CompareLayerIndex>;

  /**
   *  @brief
   *  Iterator types for entries in `FunnelBase`.
   *
   *  The template argument `constant` specifies whether it is a const or a
   *  non-const variant.
   *
   *  An iterator consists of a list of `CellLocation`s, one for each
   *  layer. This list is stored as a priority queue so that its top element
   *  can be quickly accessed and replaced.
   *  When the iterator is dereferenced, the entry associated with the top
   *  element in the priority queue will be returned.
   *  This entry will be the smallest entry among all entries that are
   *  associated to `CellLocation` objects in the priority queue.
   *  (See `CompareCellLocation` for more detail.)
   */
  template<bool constant>
  class Iterator {
   protected:
    // Friend declarations
    friend merger_type;
    friend funnel_type;

    using this_type = Iterator<constant>;

    using container_type = std::conditional_t<constant,
        funnel_type const,
        funnel_type>;

    template<bool forward>
    struct CompareIndices {
      CompareIndices(
          CellLocationList const& cell_locations,
          cell_compare const& cell_comp)
        : cell_locations{cell_locations}, cell_comp{cell_comp} {}

      constexpr bool operator()(size_type i, size_type j) const {
        CellLocation& cell_location1 = cell_locations[i];
        CellLocation& cell_location2 = cell_locations[j];
        layer_iterator& layer1 = cell_location1.layer;
        layer_iterator& layer2 = cell_location2.layer;
        size_type index1 = cell_location1.cell_index;
        size_type index2 = cell_location2.cell_index;
        if constexpr (forward) {
          if (index1 >= layer1->size()) {
            if (index2 >= layer2->size()) {
              return layer1->index < layer2->index;
            }
            return false;
          }
          if (index2 >= layer2->size()) {
            return true;
          }
          auto& cell1 = layer1->deque[index1];
          auto& cell2 = layer2->deque[index2];
          if (cell_comp(cell1, cell2)) {
            return true;
          }
          if (cell_comp(cell2, cell1)) {
            return false;
          }
          return layer1->index < layer2->index;
        } else {
          std::optional<size_type> prev_index1{layer1->prevLiveIndex(index1)};
          std::optional<size_type> prev_index2{layer2->prevLiveIndex(index2)};
          if (!prev_index1) {
            if (!prev_index2) {
              return layer1->index < layer2->index;
            }
            return true;
          }
          if (!prev_index2) {
            return false;
          }

          auto& cell1 = layer1->deque[prev_index1];
          auto& cell2 = layer2->deque[prev_index2];
          if (cell_comp(cell1, cell2)) {
            return true;
          }
          if (cell_comp(cell2, cell1)) {
            return false;
          }
          return layer1->index < layer2->index;
        }
      }

      CellLocationList const& cell_locations;
      cell_compare const& cell_comp;
    };

    using Index = typename CellLocationList::size_type;
    using ForwardIndices = std::set<Index, CompareIndices<true>>;
    using BackwardIndices = std::set<Index, CompareIndices<false>>;

    Iterator(
        CellLocationList&& cell_locations,
        cell_compare const& cell_comp)
      : cell_locations_{std::move(cell_locations)},
        cell_comp_{cell_comp},
        forward_indices_{
          CompareIndices<true>{cell_locations_, cell_comp_}},
        backward_indices_{
          CompareIndices<false>{cell_locations_, cell_comp_}},
        forward_iterators_{},
        backward_iterators_{},
        index_{0} {
      for (Index i = 0; i < cell_locations_.size(); ++i) {
        auto& cell_location = cell_locations_[i];
        index_ += cell_location.cell_index;
        forward_iterators_.push_back(forward_indices_.insert(i).first);

        std::optional<size_type> prev_live_index{
            cell_location.layer->prevLiveIndex(cell_location.cell_index)};
        backward_iterators_.push_back(
            prev_live_index ?
              backward_indices_.insert(i) : backward_indices_.end());
      }
    }

    template<bool other_const>
    Iterator(Iterator<other_const> const& other)
      : cell_locations_{other.cell_locations},
        cell_comp_{other.cell_comp_},
        forward_indices_(other.forward_indices_),
        backward_indices_(other.backward_indices_),
        forward_iterators_(other.forward_iterators_),
        backward_iterators_(other.backward_iterators_),
        index_{other.index_} {}
    template<bool other_const>
    Iterator(Iterator<other_const>&& other)
      : cell_locations_{std::move(other.cell_locations)},
        cell_comp_{other.cell_comp_},
        forward_indices_(std::move(other.forward_indices_)),
        backward_indices_(std::move(other.backward_indices_)),
        forward_iterators_(std::move(other.forward_iterators_)),
        backward_iterators_(std::move(other.backward_iterators_)),
        index_{other.index_} {}

    Iterator<false> removeConst() const& {
      return Iterator<false>(*this);
    }
    
    Iterator<false> removeConst() && {
      return Iterator<false>(std::move(*this));
    }

    // --- Private member variables --- //

    CellLocationList cell_locations_;
    cell_compare cell_comp_;
    ForwardIndices forward_indices_;
    BackwardIndices backward_indices_;
    std::vector<typename ForwardIndices::iterator> forward_iterators_;
    std::vector<typename BackwardIndices::iterator> backward_iterators_;
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
      auto& cell_location = cell_locations_[*forward_indices_.begin()];
      return cell_location.layer->deque[cell_location.cell_index].entry;
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

    constexpr bool operator<(this_type const& other) const {
      return index_ < other.index_;
    }

    constexpr bool operator<=(this_type const& other) const {
      return index_ <= other.index_;
    }

    constexpr bool operator>(this_type const& other) const {
      return index_ > other.index_;
    }

    constexpr bool operator>=(this_type const& other) const {
      return index_ >= other.index_;
    }

    this_type& operator++() {
      // Extract the current cell.
      Index index{*forward_indices_.begin()};
      auto& cell_location = cell_locations_[index];
      auto& layer = cell_location.layer;
      size_type& cell_index = cell_location.cell_index;

      // Find the next live index in the same layer.
      size_type next_live_index{layer->nextLiveIndex(cell_index)};

      // Update index_.
      index_ += next_live_index - cell_index;

      // Remove the current forward index.
      auto& forward_it = forward_iterators_[index];
      forward_indices_.erase(forward_it);

      // If backward_indices contains `index`, remove it.
      auto& backward_it = backward_iterators_[index];
      if (backward_it != backward_indices_.end()) {
        backward_indices_.erase(backward_it);
      }

      // Update the index in cell_location.
      cell_index = next_live_index;

      // Add back the forward index.
      forward_it = forward_indices_.insert(index).first;

      // Add the backward index.
      backward_it = backward_indices_.insert(index).first;

      return *this;
    }

    this_type& operator--() {
      assert((!backward_indices_.empty()));

      // Extract the previous cell.
      Index index{*backward_indices_.rbegin()};
      auto& cell_location = cell_locations_[index];
      auto& layer = cell_location.layer;
      size_type& cell_index = cell_location.cell_index;

      // Find the previous live index in the same layer.
      std::optional<size_type> prev_live_index{
          layer->prevLiveIndex(cell_index)};

      // Update index_.
      index_ -= cell_index - prev_live_index.value_or(0);

      // Remove the current forward index.
      auto& forward_it = forward_iterators_[index];
      forward_indices_.erase(forward_it);

      // Remove the current backward index.
      auto& backward_it = backward_iterators_[index];
      backward_indices_.erase(backward_it);

      // Update the index in cell_location.
      cell_index = prev_live_index.value_or(0);

      // Add back the forward index.
      forward_it = forward_indices_.insert(index).first;

      // If `prev_live_index` has a value, add back the backward index.
      backward_it = prev_live_index ?
          backward_indices_.insert(index).first : backward_indices_.end();

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
     *  Convenience function for constructing an iterator.
     *
     *  An iterator is constructed from a list of `CellLocation`s, one
     *  for each layer. This function provides a common framework for
     *  constructing an iterator as follows.
     *
     *  For each `layer`, a `CellLocation` is created from the return
     *  value of `layer_func(layer)`, which should be an index of a cell in the
     *  layer. These `CellLocation`s are then fed into the constructor of
     *  `Iterator`.
     *
     *  The input `layer_func` must be callable with the following prototype:
     *
     *      size_type layer_func(layer_iterator const&);
     *
     *  It is possible for `layer_func` to return an index that is not
     *  dereferenceable, i.e., the past-the-end index may be returned.
     */
    template<class LayerFunc>
    static constexpr this_type build(
        container_type& container,
        LayerFunc layer_func) {
      size_type const num_layers = container.layers_.size();
      CellLocationList cell_location_list{num_layers};
      size_type layer_index{0};
      for (layer_iterator layer = container.layers_.begin();
           layer != container.layers_.end();
           ++layer) {
        CellLocation& cell_location = cell_location_list[layer_index++];
        cell_location.layer = layer;
        cell_location.cell_index = layer_func(layer);
      }
      return this_type{
          std::move(cell_location_list),
          container.cell_comp_};
    }

    /**
     *  @brief
     *  Asynchronous version of `build()`.
     *
     *  This function serves the same functionality as `build()`, but it
     *  attempts to construct `CellLocation`s asynchronously.
     *  This is possible when `layer_func` is local to one layer.
     */
    template<class LayerFunc>
    static constexpr this_type build_async(
        container_type& container,
        std::launch policy,
        LayerFunc layer_func) {
      size_type const num_layers = container.layers_.size();
      CellLocationList cell_location_list{num_layers};
      std::vector<std::future<size_type>> tasks{num_layers};
      size_type layer_index{0};
      for (layer_iterator layer = container.layers_.begin();
           layer != container.layers_.end();
           ++layer) {
        tasks[layer_index] = std::async(
            policy,
            [&layer_func,
             &layer,
             &cell_location = cell_location_list[layer_index]]() {
              cell_location.layer = layer;
              return cell_location.cell_index = layer_func(layer);
            });
        ++layer_index;
      }
      for (auto& task : tasks) {
        task.get();
      }
      return this_type{
          std::move(cell_location_list),
          container.cell_comp_};
    }

    static constexpr this_type begin(container_type& container) {
      return build(
          container,
          [](layer_iterator const&) { return size_type{0}; });
    }

    static constexpr this_type end(container_type& container) {
      return build(
          container,
          [](layer_iterator const& layer) { return layer->size(); });
    }

    /**
     *  @brief
     *  Finds a cell index of a lower bound of a given key in a layer.
     *
     *  This function searches through the whole layer (dead and live cells)
     *  for a lower bound index of the given key; then, if the index points to
     *  a dead cell, pushes the index upward to the nearest live cell.
     */
    template<class K>
    static constexpr size_type layer_lower_bound(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_comp) {
      size_type const cell_index{static_cast<size_type>(
          std::distance(
            deque.begin(),
            std::lower_bound(deque.begin(), deque.end(), key, cell_comp)))};
      return cell_index < deque.size()
          ? layer_type::rightDeadEndIndex(deque, cell_index)
          : cell_index;
    }

    template<class K>
    static constexpr this_type lower_bound(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_comp = container.cell_comp_](
              layer_iterator const& layer) {
            return layer_lower_bound(layer->deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr this_type lower_bound_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_comp = container.cell_comp_](
              layer_iterator const& layer) {
            return layer_lower_bound(layer->deque, key, cell_comp);
          });
    }

    /**
     *  @brief
     *  Finds a cell index of an upper bound of a given key in a layer.
     *
     *  This function searches through the whole layer (dead and live cells)
     *  for an upper bound index of the given key; then, if the index points to
     *  a dead cell, pushes the index upward to the nearest live cell.
     */
    template<class K>
    static constexpr size_type layer_upper_bound(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_comp) {
      size_type const cell_index{static_cast<size_type>(
          std::distance(
            deque.begin(),
            std::upper_bound(deque.begin(), deque.end(), key, cell_comp)))};
      return cell_index < deque.size()
          ? layer_type::rightDeadEndIndex(deque, cell_index)
          : cell_index;
    }

    template<class K>
    static constexpr this_type upper_bound(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_comp = container.cell_comp_](
              layer_iterator const& layer) {
            return layer_upper_bound(layer->deque, key, cell_comp);
          });
    }

    template<class K>
    static constexpr this_type upper_bound_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_comp = container.cell_comp_](
              layer_iterator const& layer) {
            return layer_upper_bound(layer->deque, key, cell_comp);
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

    /**
     *  @brief
     *  Returns an iterator to an entry whose key matches the given key if such
     *  an entry exists.
     *
     *  This function is slightly different from `find()` because it does not
     *  return `end()` if the funnel doesn't contain the given `key`.
     *  This is slightly more efficient than `find()` because `end()` requires
     *  constructing an iterator.
     */
    template<class K>
    static constexpr std::optional<this_type> search(
        container_type& container, K const& key) {
      this_type i{lower_bound(container, key)};
      if (i.queue_.empty()) {
        return {};
      }
      auto& cell_location = i.top();
      auto& cell_comp = container.cell_comp_;
      auto& found_key = cell_comp.ext_key(
          cell_location.layer->deque[cell_location.cell_index].entry);
      if (cell_comp(found_key, key) || cell_comp(key, found_key)) {
        return {};
      }
      return i;
    }

    /**
     *  @brief
     *  Asynchronous version of `search()`.
     */
    template<class K>
    static constexpr std::optional<this_type> search_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      this_type i{lower_bound_async(container, policy, key)};
      if (i.queue_.empty()) {
        return {};
      }
      auto& cell_location = i.top();
      auto& cell_comp = container.cell_comp_;
      auto& found_key = cell_comp.ext_key(
          cell_location.layer->deque[cell_location.cell_index].entry);
      if (cell_comp(found_key, key) || cell_comp(key, found_key)) {
        return {};
      }
      return i;
    }

    template<class K>
    static constexpr this_type find(container_type& container, K const& key) {
      return search(container, key).value_or(end(container));
    }

    template<class K>
    static constexpr std::optional<this_type> find_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return search_async(container, policy, key).value_or(end(container));
    }

    /**
     *  @brief
     *  Returns the cell index within a given `layer` such that the result from
     *  `build()` is a valid iterator whose `top()` is a `CellLocation`
     *  matching the given `target_layer_index` and `target_cell_index`.
     *
     *  This is used in `buildAt()` and `buildAt_async()`.
     */
    template<class K>
    static constexpr size_type layer_at(
        layer_index_type target_layer_index,
        size_type target_cell_index,
        K const& target_key,
        layer_iterator const& layer,
        cell_compare const& cell_comp) {
      if (layer->index < target_layer_index) {
        return layer_upper_bound(layer->deque, target_key, cell_comp);
      }
      if (layer->index > target_layer_index) {
        return layer_lower_bound(layer->deque, target_key, cell_comp);
      }
      return target_cell_index;
    }

    /**
     *  @brief
     *  Builds an iterator whose `top()` is the given `CellLocation`.
     */
    static constexpr this_type buildAt(
        container_type& container,
        CellLocation const& cell_location) {
      auto& cell_comp = container.cell_comp_;
      return build(
          container,
          [&cell_comp,
           target_layer_index = cell_location.layer->index,
           target_cell_index = cell_location.cell_index,
           &target_key = cell_comp.ext_key(
             cell_location.layer->deque[cell_location.cell_index].entry)
          ](layer_iterator const& layer) {
            return layer_at(
                target_layer_index,
                target_cell_index,
                target_key,
                layer,
                cell_comp);
          });
    }

    /**
     *  @brief
     *  Asynchronous version of `buildAt()`.
     */
    static constexpr this_type buildAt_async(
        container_type& container,
        std::launch policy,
        CellLocation const& cell_location) {
      auto& cell_comp = container.cell_comp_;
      return build_async(
          container, policy,
          [&cell_comp,
           target_layer_index = cell_location.layer->index,
           target_cell_index = cell_location.cell_index,
           &target_key = cell_comp.ext_key(
             cell_location.layer->deque[cell_location.cell_index].entry)
          ](layer_iterator const& layer) {
            return layer_at(
                target_layer_index,
                target_cell_index,
                target_key,
                layer,
                cell_comp);
          });
    }

    /**
     *  @brief
     *  `const` version of `top()`.
     */
    constexpr CellLocation const& top() const {
      return cell_locations_[*forward_indices_.begin()];
    }

    // For unit tests.
    friend class FunnelTest;

  }; // class Iterator

  // Internal operations.
  //
  // Some of these functions will have a `_unique` or `_multi` suffix to
  // indicate that they are specific for *unique* (set and map) or *multi*
  // (multiset and multimap) data structure.

  /**
   *  @brief
   *  Implementation of `count()` for set and map types.
   */
  template<class K>
  constexpr size_type count_unique(K const& key) const {
    return contains(key) ? 1 : 0;
  }

  /**
   *  @brief
   *  Asynchronous version of `count_unique()`.
   */
  template<class K>
  constexpr size_type count_async_unique(
      std::launch policy,
      K const& key) const {
    return contains_async(policy, key) ? 1 : 0;
  }

  /**
   *  @brief
   *  Implementation of `count()` for multiset and multimap types.
   */
  template<class K>
  constexpr size_type count_multi(K const& key) const {
    std::pair<const_iterator, const_iterator> range{equal_range(key)};
    return static_cast<size_type>(std::distance(range.first, range.second));
  }

  /**
   *  @brief
   *  Asynchronous version of `count_multi()`.
   */
  template<class K>
  constexpr size_type count_async_multi(
      std::launch policy,
      K const& key) const {
    std::pair<const_iterator, const_iterator> range{
        equal_range_async(policy, key)};
    return static_cast<size_type>(std::distance(range.first, range.second));
  }

  template<bool constant>
  constexpr iterator erase(Iterator<constant> pos) {
    CellLocation cell{pos.top()};
    ++pos;
    CellLocation next_cell{pos.top()};
    cell.layer->eraseAtIndex(cell.cell_index);
    --num_live_;
    merger_.mergeOnErase(*this, LayerSet{cell.layer}, &next_cell);
    return iterator::buildAt(*this, next_cell);
  }

  template<bool constant>
  constexpr iterator erase(
      Iterator<constant> first, Iterator<constant> last,
      size_type* returned_count = nullptr) {
    if (first == last) {
      return iterator(first);
    }
    CellLocation last_cell{last.top()};
    LayerSet touched_layers;
    size_type count{0};
    for (; first != last; ++first) {
      CellLocation& cell = first.top();
      cell.layer->eraseAtIndex(cell.cell_index);
      touched_layers.insert(cell.layer);
      ++count;
    }
    num_live_ -= count;
    if (returned_count) {
      *returned_count = count;
    }
    merger_.mergeOnErase(*this, touched_layers, &last_cell);
    return iterator::buildAt(*this, last_cell);
  }

  template<class K>
  constexpr size_type erase_unique(K const& key) {
    std::optional<iterator> i{search(key)};
    if (!i) {
      return 0;
    }
    erase(*i);
    return 1;
  }

  template<class K>
  constexpr size_type erase_multi(K const& key) {
    std::pair<iterator, iterator> range{equal_range(key)};
    size_type count{0};
    erase(range.first, range.second, &count);
    return count;
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
   */
  template<class K>
  constexpr InsertPosition insertPosition(K const& key) {
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
   *  Constructs a new entry and adds it at the given `insert_position`.
   *  If a new layer is added, `insert_position` will be updated to contain
   *  the iterator to the new layer.
   */
  template<class... Args>
  constexpr void constructEntryAt(
      InsertPosition& insert_position,
      Args&&... args) {
    ++num_cells_;
    ++num_live_;
    if (insert_position.has_value()) {
      auto& layer_insert_position = insert_position.value();
      auto& layer = layer_insert_position.layer;
      if (layer_insert_position.front) {
        layer->emplace_front(std::forward<Args>(args)...);
      } else {
        layer->emplace_back(std::forward<Args>(args)...);
      }
    } else {
#if ENABLE_FUNNEL_LAYER_INDEX_OVERFLOW_CHECK == 1
      // This is disabled by default because it should not be necessary in
      // practice.

      // Reindex layers if layer_index_counter_ is about to overflow.
      if (layer_index_counter_ ==
          std::numeric_limits<layer_index_type>::max()) {
        layer_index_counter_ = 0;
        for (auto& layer : layers_) {
          layer->index = layer_index_counter_++;
        }
      }
#endif
      layers_.emplace_back(layer_index_counter_++, cell_alloc_)
          .emplace_back(std::forward<Args>(args)...);
      insert_position.emplace(std::prev(layers_.end()), false);
    }
  }

  /**
   *  @brief
   *  Inserts an entry as the first or the last element of an appropriate layer
   *  in the funnel, then returns the `LayerEnd` that points to the inserted
   *  entry.
   *
   *  This function does not check if the funnel already has a key that is
   *  equal to the key of the inserted entry.
   */
  template<class E = entry_type>
  constexpr LayerEnd insertEntry(E&& entry) {
    InsertPosition insert_position{insertPosition(cell_comp_.ext_key(entry))};
    constructEntryAt(insert_position, std::forward<E>(entry));
    return insert_position.value();
  }

  /**
   *  @brief
   *  Constructs a new entry and inserts it by calling `insertEntry()`, then
   *  returns the `LayerEnd` that points to the new entry.
   */
  template<class... Args>
  constexpr LayerEnd constructEntry(Args&&... args) {
    return insertEntry(Entry(std::forward<Args>(args)...));
  }

  /**
   *  @brief
   *  Special case of `constructEntry()` for maps and multimaps.
   *
   *  This function is similar to `constructEntry()`, but it allows `key` to
   *  maintain type `K` during comparisons.
   */
  template<class K = key_type, class... VArgs>
  constexpr LayerEnd constructEntryPair(
      K&& key,
      VArgs&&... v_args) {
    InsertPosition insert_position{insertPosition(key)};
    constructEntryAt(
        insert_position,
        std::forward<K>(key),
        std::forward<VArgs>(v_args)...);
    return insert_position.value();
  }

  /**
   *  @brief
   *  Special case of `constructEntryPair()` with `std::piecewise_construct`.
   *
   *  This function simply passes all its arguments to `constructEntry()`.
   *  It is an overload of `constructEntryPair()` that is provided just for
   *  the consistency of the interface.
   */
  template<class... KArgs, class... VArgs>
  constexpr LayerEnd constructEntryPair(
      std::piecewise_construct_t,
      std::tuple<KArgs...> k_args,
      std::tuple<VArgs...> v_args) {
    return constructEntry(std::piecewise_construct, k_args, v_args);
  }

  constexpr iterator updateAfterInsert(LayerEnd&& where) {
    CellLocation new_cell{
        where.layer,
        where.front ? 0 : where.layer->size() - 1};
    merger_.mergeOnInsert(*this, LayerSet{where.layer}, &new_cell);
    return iterator::buildAt(*this, new_cell);
  }

  /**
   *  @brief
   *  Implementation of `emplace()` for multisets.
   */
  template<class... Args>
  constexpr iterator emplace_multi(Args&&... args) {
    return updateAfterInsert(constructEntry(std::forward<Args>(args)...));
  }

  /**
   *  @brief
   *  Implementation of `insert()` for multisets.
   */
  template<class E>
  constexpr iterator insert_multi(E&& entry) {
    return updateAfterInsert(insertEntry(std::forward<E>(entry)));
  }

  /**
   *  @brief
   *  Implementation of `emplace()` for sets.
   */
  template<class... Args>
  std::pair<iterator, bool> emplace_unique(Args&&... args) {
    Entry entry{std::forward<Args>(args)...};
    std::optional<iterator> i{search(cell_comp_.ext_key(entry))};
    if (i) {
      return {*i, false};
    }
    return insert_multi(std::move(entry));
  }

  /**
   *  @brief
   *  Implementation of `insert()` for sets.
   */
  template<class E>
  constexpr iterator insert_unique(E&& entry) {
    std::optional<iterator> i{search(cell_comp_.ext_key(entry))};
    if (i) {
      return {*i, false};
    }
    return insert_multi(std::forward<E>(entry));
  }

  template<class InputIt>
  constexpr void insert_multi(InputIt first, InputIt last) {
    LayerSet touched_layers{};
    for (InputIt it = first; it != last; ++it) {
      LayerEnd where{insertEntry(*it)};
      touched_layers.insert(where.layer);
    }
    merger_.mergeOnInsert(*this, touched_layers, nullptr);
  }

  template<class InputIt>
  constexpr void insert_unique(InputIt first, InputIt last) {
    LayerSet touched_layers{};
    for (InputIt it = first; it != last; ++it) {
      if (contains(cell_comp_.ext_key(*it))) {
        continue;
      }
      LayerEnd where{insertEntry(*it)};
      touched_layers.insert(where.layer);
    }
    merger_.mergeOnInsert(*this, touched_layers, nullptr);
  }

  constexpr void insert_multi(std::initializer_list<entry_type> ilist) {
    insert_multi(ilist.begin(), ilist.end());
  }

  constexpr void insert_unique(std::initializer_list<entry_type> ilist) {
    insert_unique(ilist.begin(), ilist.end());
  }

  /**
   *  @brief
   *  Merges `layer0` with `layer1`, then erases `layer1` from `layers_`.
   *  This function will be called by `merge_policy`.
   *
   *  `layer0` should point to a layer that is positioned before `layer1` in
   *  `layers_`.
   *
   *  `tracked_entry` is an optional argument. If `tracked_entry` is not null,
   *  its value must be an `TrackedCell` of an entry to track, and after
   *  `mergeLayers()` returns, its value may be changed to a new
   *  `TrackedCell`.
   *
   *  `layer0` may become invalidated if and only if it points to a layer with
   *  no live cells and `layer1` is the past-the-end iterator. In this case,
   *  `layer0` will be removed and `mergeLayers()` will return `false`,
   *  indicating that `layer0` is no longer valid.
   *  Otherwise, `mergeLayers()` will return `true` and `layer0` will remain a
   *  valid `layer_iterator`.
   *  Note that In all cases, `layer1` is invalidated after `mergeLayers()`
   *  returns.
   *
   *  Note: The requirement that `layer0` must remain a valid iterator if the
   *  function returns `true` implies that `LayerList` must preserve iterators
   *  before the point of modification, i.e., `LayerList` may be `std::list` or
   *  `std::vector`, but not `std::deque`.
   */
  void mergeLayers(
      layer_iterator layer0,
      layer_iterator layer1,
      CellLocation* tracked_cell = nullptr) {
    if (layer1 == layers_.end()) {
      num_cells_ -= layer0->size();
      layer0->optimize(
          (tracked_cell && tracked_cell->layer == layer0) ?
          &tracked_cell->cell_index : nullptr);
      if (layer0.empty()) {
        layers_.erase(layer0);
        return;
      }
      num_cells_ += layer0->size();
      return;
    }
    bool track0{false};
    bool track1{false};
    size_type tracked_cell_index;
    if (tracked_cell) {
      if (tracked_cell.layer == layer0) {
        tracked_cell_index = tracked_cell->cell_index;
        track0 = true;
      } else if (tracked_cell.layer == layer1) {
        tracked_cell->layer = layer0;
        tracked_cell_index = tracked_cell->cell_index;
        track1 = true;
      }
    }

    deque_type& deque0{layer0->deque};
    deque_type& deque1{layer1->deque};
    size_t index0{0};
    size_t index1{0};
    layer_type merged_layer{layer0->index, cell_alloc_};
    num_cells_ -= deque0.size() + deque1.size();

    while (true) {
      if (index0 == deque0.size()) {
        for (
            ; index1 < deque1.size()
            ; index1 = layer_type::nextLiveIndex(deque1, index1)) {
          if (track1 && index1 == tracked_cell_index) {
            tracked_cell->cell_index = merged_layer.size();
          }
          merged_layer.push_back(std::move(deque1[index1]));
        }
        break;
      }
      if (index1 == deque1.size()) {
        for (
            ; index0 < deque0.size()
            ; index0 = layer_type::nextLiveIndex(deque0, index0)) {
          if (track0 && index0 == tracked_cell_index) {
            tracked_cell->cell_index = merged_layer.size();
          }
          merged_layer.push_back(std::move(deque0[index0]));
        }
        break;
      }
      if (cell_comp_(deque0[index0], deque1[index1])) {
        if (track0 && index0 == tracked_cell_index) {
          tracked_cell->cell_index = merged_layer.size();
        }
        merged_layer.push_back(std::move(deque0[index0]));
        index0 = layer_type::nextLiveIndex(deque0, index0);
      } else {
        if (track1 && index1 == tracked_cell_index) {
          tracked_cell->cell_index = merged_layer.size();
        }
        merged_layer.push_back(std::move(deque1[index1]));
        index1 = layer_type::nextLiveIndex(deque1, index1);
      }
    }
    num_cells_ += merged_layer.size();
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

  // List of layers. This is marked `mutable` because we will need non-const
  // `layer_iterator`s even in a const iterator.
  mutable layer_list_type layers_;

  // Layer index counter.
  mutable layer_index_type layer_index_counter_{0};

  // Number of cells. `iterator::index_` will not exceed `num_cells_`.
  mutable size_type num_cells_{0};

  // Number of live cells. `size()` will return this value.
  size_type num_live_{0};

  // For unit tests.
  friend class FunnelTest;
};

template<class Entry>
struct DefaultDeleteEntry {
  using entry_type = Entry;
  constexpr void operator()(entry_type& entry) const {}
};

// `std::identity` in C++20 could be used here. (The member typedefs are not
// really necessary.)
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
    template<class> class Merger = SimpleRatioMergerCurried<1, 2>::type,
    template<class, class> class Deque = std::deque,
    template<class, class> class LayerList = std::list>
class FunnelSet : public FunnelBase<
    Key,
    DefaultDeleteEntry<Key>,
    Key,
    ExtractKeyPassthrough<Key>,
    Compare,
    Allocator,
    Merger,
    Deque,
    LayerList> {
 public:
  using this_type = FunnelSet<
      Key,
      Compare,
      Allocator,
      DeleteEntry,
      Merger,
      Deque,
      LayerList>;
  using super_type = FunnelBase<
      Key,
      DefaultDeleteEntry<Key>,
      Key,
      ExtractKeyPassthrough<Key>,
      Compare,
      Allocator,
      Merger,
      Deque,
      LayerList>;

  using key_type = typename super_type::key_type;
  using key_compare = typename super_type::key_compare;
  using allocator_type = typename super_type::allocator_type;
  using delete_entry = typename super_type::delete_entry;
  using merger_type = typename super_type::merger_type;

  using extract_key = typename super_type::extract_key;
  using size_type = typename super_type::size_type;
  using iterator = typename super_type::iterator;
  using const_iterator = typename super_type::const_iterator;

  using layer_type = typename super_type::layer_type;
  using layer_list_type = typename super_type::layer_list_type;
  using layer_iterator = typename super_type::layer_iterator;
  using const_layer_iterator = typename super_type::const_layer_iterator;

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

  // For unit tests.
  friend class FunnelTest;
};

} // namespace funnel

