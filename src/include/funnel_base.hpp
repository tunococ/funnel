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
template<typename Entry, typename SizeType = std::size_t>
struct FunnelCell {
  /// `Entry`.
  using entry_type = Entry;
  /// `SizeType`.
  using size_type = SizeType;
  /// `std::make_signed_t<size_type>`.
  using offset_type = std::make_signed_t<size_type>;
  /// `FunnelCell<entry_type, size_type>`.
  using this_type = FunnelCell<entry_type, size_type>;

  template<typename... Args>
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
};

/**
 *  @brief
 *  Layer in a funnel.
 *
 *  A funnel contains a list of layers, each of which contains a list of cells
 *  sorted by their entries. The list of cells is called `deque`.
 *  The type `Deque` can be any list data structure that supports efficient
 *  random access with an index and insertion at both ends.
 */
template<typename Deque>
struct FunnelLayer {
  /// `Deque`.
  using deque_type = Deque;
  /// `deque_type::value_type`.
  using value_type = typename deque_type::value_type;
  /// `deque_type::allocator_type`.
  using allocator_type = typename deque_type::allocator_type;
  /// `deque_type::size_type`.
  using size_type = typename deque_type::size_type;
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

  /// `value_type`.
  using cell_type = value_type;
  /// `cell_type::entry_type`.
  using entry_type = typename cell_type::entry_type;
  /// `entry_type&`.
  using entry_reference = entry_type&;
  /// `entry_type const&`.
  using entry_const_reference = entry_type const&;
  /**
   *  @brief
   *  `std::allocator_traits<allocator_type>::rebind_alloc<entry_type>.
   *
   *  This is used in defining pointer types for `entry_type`.
   *  There is no concrete instance of `entry_allocator_type`.
   */
  using entry_allocator_type =
      typename std::allocator_traits<allocator_type>::
      template rebind_alloc<entry_type>;
  /// `entry_allocator_type::pointer`.
  using entry_pointer =
      typename std::allocator_traits<entry_allocator_type>::pointer;
  /// `entry_allocator_type::const_pointer`.
  using entry_const_pointer =
      typename std::allocator_traits<entry_allocator_type>::const_pointer;

  /// `FunnelLayer<deque_type>`.
  using this_type = FunnelLayer<deque_type>;

  // Static assertions
  static_assert(std::is_same<size_type, typename cell_type::size_type>::value);

  // Constructor
  FunnelLayer(allocator_type const& alloc) : deque(alloc) {}

  FunnelLayer(this_type const&) = default;
  FunnelLayer(this_type&&) = default;
  this_type& operator=(this_type const&) = default;
  this_type& operator=(this_type&&) = default;

  // Member variables

  /// Sorted list of cells.
  deque_type deque;

  /**
   *  @brief
   *  The number of dead cells in this layer.
   *
   *  This number is used to decide when to merge layers.
   */
  size_type num_dead{0};

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
    return deque[index].state.index() == 0
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
    return deque[index].state.index() == 0
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

  template<typename... Args>
  void emplace_front(Args&&... args) {
    deque.emplace_front(std::forward<Args>(args)...);
  }

  template<typename... Args>
  void emplace_back(Args&&... args) {
    deque.emplace_back(std::forward<Args>(args)...);
  }

  constexpr size_type size() const {
    return deque.size();
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
    auto& state = deque[index].state;
    assert(state.index() == 0);

    // Mark this cell as dead.
    state.template emplace<2>(0);

    if (index == 0) {
      // If the marked cell is the leftmost cell, remove all dead cells from
      // the left end of `deque`.
      size_type j = 0;
      for (; j < deque.size() && deque[j].index != 0; ++j) {}
      num_dead -= j;
      deque.erase(deque.begin(), std::next(deque.begin(), j));
      return;
    }
    if (index + 1 == deque.size()) {
      // If the marked cell is the rightmost cell, remove all dead cells from
      // the right end of `deque`.
      size_type j = 0;
      size_type const last_index = deque.size() - 1;
      for (; j < deque.size() && deque[last_index - j].index != 0; ++j) {}
      num_dead -= j;
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
 *  Base class for `FunnelSet` and `FunnelMap`.
 *
 *  A funnel stores layers in a data structure of type `LayerList`.
 *  Elements of `LayerList` should have type `FunnelLayer`.
 *  (`std::list<FunnelLayer<std::deque<FunnelCell<Entry>>>>` is a typical
 *  choice, where `Entry` is the intended type of entries in the funnel.)
 *
 *  The member typedef `value_type` of `FunnelBase`, defined as
 *  `LayerList::value_type::value_type`, is the type of entries in the funnel.
 *
 *  `ExtractKey` is a unary operator that turns `value_type const&` to
 *  `Key const&`.
 *  `Compare` is a binary operator that compares two `Key const&` objects.
 *  (It returns `true` if the first operand is strictly less than the second
 *  operand.)
 *  `ExtractKey` and `Compare` together define how entries are ordered.
 *
 *  Because entries of a funnel are lazily deleted, their actual destructions
 *  may happen at undetermined time, i.e., their destructors might not be
 *  called right when they are marked *deleted*. A custom `DeleteEntry`
 *  callback function, which will be called whenever an entry is marked 
 *  *deleted*, can be provided in the constructor of `FunnelBase`.
 *  `DeleteEntry` must support being called with one argument of type
 *  `value_type&`. The operation of `DeleteEntry` may modify the entry, but
 *  such modification must not change the return value of subsequent calls to
 *  `ExtractKey`.
 */
template<
    typename LayerList,
    typename DeleteEntry,
    typename Key,
    typename ExtractKey,
    typename Compare = std::less<Key>>
class FunnelBase {
 public:
  /// `LayerList`.
  using layer_list_type = LayerList;
  /// `layer_list_type::value_type`.
  using layer_type = typename layer_list_type::value_type;
  /// `DeleteEntry`.
  using delete_entry = DeleteEntry;
  /// `Key`.
  using key_type = Key;
  /// `ExtractKey`.
  using extract_key = ExtractKey;
  /// `Compare`.
  using key_compare = Compare;
  /// `layer_list_type::allocator_type`.
  using layer_allocator_type = typename layer_list_type::allocator_type;
  /// `layer_type::cell_type`.
  using cell_type = typename layer_type::cell_type;
  /// `layer_type::deque_type`.
  using deque_type = typename layer_type::deque_type;
  /// `layer_type::entry_type`.
  using entry_type = typename layer_type::entry_type;
  /// `entry_type`.
  using value_type = entry_type;
  /// `layer_type::size_type`.
  using size_type = typename layer_type::size_type;
  /// `layer_type::difference_type`.
  using difference_type = typename layer_type::difference_type;
  /// Rebound allocator from `layer_allocator_type` for `value_type`.
  using allocator_type =
      typename std::allocator_traits<layer_allocator_type>::
      template rebind_alloc<value_type>;
  using cell_allocator_type =
      typename std::allocator_traits<layer_allocator_type>::
      template rebind_alloc<cell_type>;

  /// `value_type&`.
  using reference = value_type&;
  /// `value_type const&`.
  using const_reference = value_type const&;
  /// `layer_type::entry_pointer`.
  using pointer = typename layer_type::entry_pointer;
  /// `layer_type::entry_const_pointer`.
  using const_pointer = typename layer_type::entry_const_pointer;

  /// This class.
  using this_type = FunnelBase<
      layer_list_type, delete_entry,
      key_type, extract_key,
      key_compare>;

  /**
   *  @brief
   *  Operator for comparing the member `entry` in `cell_type` objects.
   */
  struct cell_compare {
    key_compare key_cmp;
    extract_key ext_key;
    constexpr cell_compare(
        extract_key const& ext_key, key_compare const& key_cmp)
      : ext_key{ext_key}, key_cmp{key_cmp} {}
    cell_compare(cell_compare const&) = default;
    cell_compare(cell_compare&&) = default;
    template<typename K1, typename K2>
    constexpr bool operator()(
        K1 const& a,
        K2 const& b) const {
      return key_cmp(a, b);
    }
    template<typename K1, typename S1, typename K2>
    constexpr bool operator()(
        FunnelCell<K1, S1> const& a,
        K2 const& b) const {
      return key_cmp(ext_key(a.entry), b);
    }
    template<typename K1, typename K2, typename S2>
    constexpr bool operator()(
        K1 const& a,
        FunnelCell<K2, S2> const& b) const {
      return key_cmp(a, ext_key(b.entry));
    }
    template<typename K1, typename S1, typename K2, typename S2>
    constexpr bool operator()(
        FunnelCell<K1, S1> const& a,
        FunnelCell<K2, S2> const& b) const {
      return key_cmp(ext_key(a.entry), ext_key(b.entry));
    }
    template<typename A, typename B>
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
   *  @param extract_key
   *    Unary operator for extracting `key_type const&` from
   *    `value_type const&`.
   *  @param cmp
   *    Binary operator for comparing two `key_type const&` values.
   *    `cmp(a, b)` should return `true` if and only if its `a` is strictly
   *    less than `b`.
   *  @param allocator
   *    Allocator for constructing `layer_type` objects.
   */
  constexpr FunnelBase(
      delete_entry const& del_ent = delete_entry(),
      extract_key const& ext_key = extract_key(),
      key_compare const& key_cmp = key_compare(),
      allocator_type const& alloc = allocator_type())
    : del_ent_{del_ent},
      cell_cmp_{ext_key, key_cmp},
      value_alloc_{alloc},
      layer_alloc_{alloc},
      cell_alloc_{alloc},
      layers_{layer_alloc_} {
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

  constexpr iterator lower_bound(key_type const& key) {
    return iterator::lower_bound(*this, key);
  }

  constexpr const_iterator lower_bound(key_type const& key) const {
    return const_iterator::lower_bound(*this, key);
  }

  constexpr iterator lower_bound_async(
      std::launch policy,
      key_type const& key) {
    return iterator::lower_bound_async(*this, policy, key);
  }

  constexpr const_iterator lower_bound_async(
      std::launch policy,
      key_type const& key) const {
    return const_iterator::lower_bound_async(*this, policy, key);
  }

  constexpr iterator upper_bound(key_type const& key) {
    return iterator::upper_bound(*this, key);
  }

  constexpr const_iterator upper_bound(key_type const& key) const {
    return const_iterator::upper_bound(*this, key);
  }

  constexpr iterator upper_bound_async(
      std::launch policy,
      key_type const& key) {
    return iterator::upper_bound_async(*this, policy, key);
  }

  constexpr const_iterator upper_bound_async(
      std::launch policy,
      key_type const& key) const {
    return const_iterator::upper_bound_async(*this, policy, key);
  }

  constexpr iterator find(key_type const& key) {
    return iterator::find(*this, key);
  }

  constexpr const_iterator find(key_type const& key) const {
    return const_iterator::find(*this, key);
  }

  constexpr iterator find_async(
      std::launch policy,
      key_type const& key) {
    return iterator::find_async(*this, policy, key);
  }

  constexpr const_iterator find_async(
      std::launch policy,
      key_type const& key) const {
    return const_iterator::find_async(*this, policy, key);
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
     *  `std::priority_queue::pop()`, while `key_cmp` is a *less than*
     *  operator.
     */
    struct CompareHorizontalIterator {
      cell_compare cell_cmp;
      constexpr CompareHorizontalIterator(cell_compare const& cell_cmp)
        : cell_cmp{cell_cmp} {}
      constexpr bool operator()(
          HorizontalIterator const& a,
          HorizontalIterator const& b) const {
        if (a.cell_index >= a.layer->deque.size()) {
          return b.cell_index < b.layer->deque.size();
        }
        if (b.cell_index >= b.layer->deque.size()) {
          return false;
        }
        return cell_cmp(
            b.layer->deque[b.cell_index],
            a.layer->deque[a.cell_index]);
      }
    };

    Iterator(
        HorizontalIteratorList&& horz_it_list,
        size_type index,
        cell_compare const& cell_cmp)
      : queue_{
          CompareHorizontalIterator{cell_cmp},
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
    template<typename LayerFunc>
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
          container.cell_cmp_};
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
    template<typename LayerFunc>
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
          container.cell_cmp_};
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

    template<typename K>
    static constexpr size_type layer_lower_bound(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_cmp) {
      size_type const cell_index{
          std::distance(
            deque.begin(),
            std::lower_bound(deque.begin(), deque.end(), key, cell_cmp))};
      return cell_index < deque.size()
          ? layer_type::rightLiveIndex(deque, cell_index)
          : cell_index;
    }

    template<typename K>
    static constexpr this_type lower_bound(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_cmp = container.cell_cmp](deque_type const& deque) {
            return layer_lower_bound(deque, key, cell_cmp);
          });
    }

    template<typename K>
    static constexpr this_type lower_bound_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_cmp = container.cell_cmp](deque_type const& deque) {
            return layer_lower_bound(deque, key, cell_cmp);
          });
    }

    template<typename K>
    static constexpr size_type layer_upper_bound(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_cmp) {
      size_type const cell_index{
          std::distance(
            deque.begin(),
            std::upper_bound(deque.begin(), deque.end(), key, cell_cmp))};
      return cell_index < deque.size()
          ? layer_type::rightLiveIndex(deque, cell_index)
          : cell_index;
    }

    template<typename K>
    static constexpr this_type upper_bound(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_cmp = container.cell_cmp](deque_type const& deque) {
            return layer_upper_bound(deque, key, cell_cmp);
          });
    }

    template<typename K>
    static constexpr this_type upper_bound_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_cmp = container.cell_cmp](deque_type const& deque) {
            return layer_upper_bound(deque, key, cell_cmp);
          });
    }

    template<typename K>
    static constexpr this_type layer_find(
        deque_type const& deque,
        K const& key,
        cell_compare const& cell_cmp) {
      size_type const cell_index{
          std::distance(
            deque.begin(),
            std::lower_bound(deque.begin(), deque.end(), key, cell_cmp))};
      if (cell_index >= deque.size()) {
        return deque.size();
      }
      size_type const live_cell_index{
          layer_type::rightLiveIndex(deque, cell_index)};
      if (cell_cmp.compare(cell_cmp.ext_key(deque[live_cell_index].entry), key)
          == 0) {
        return live_cell_index;
      }
      return deque.size();
    }

    template<typename K>
    static constexpr this_type find(
        container_type& container,
        K const& key) {
      return build(
          container,
          [&key, &cell_cmp = container.cell_cmp](deque_type const& deque) {
            return layer_find(deque, key, cell_cmp);
          });
    }

    template<typename K>
    static constexpr size_type find_async(
        container_type& container,
        std::launch policy,
        K const& key) {
      return build_async(
          container, policy,
          [&key, &cell_cmp = container.cell_cmp](deque_type const& deque) {
            return layer_find(deque, key, cell_cmp);
          });
    }

  };

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

  struct LayerInsertPosition {
    layer_iterator layer;
    bool front;
    LayerInsertPosition(layer_iterator layer, bool front)
      : layer{layer}, front{front} {}
  };

  /**
   *  @brief
   *  Where to insert a new entry.
   *
   *  This is the return type of `insertPosition()`.
   *
   *  The empty state means that a new layer needs to be created.
   *  Otherwise, the value contains the layer iterator and a flag whether an
   *  insertion should take place in the front or in the back of the layer.
   */
  using InsertPosition = std::optional<LayerInsertPosition>;

  template<typename K, bool use_binary_search>
  constexpr InsertPosition insertPosition(K const& key);

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
  template<typename K>
  constexpr InsertPosition insertPosition(K const& key) {
    layer_iterator layer = layers_.begin();
    for (; layer != layers_.end(); ++layer) {
      if (!cell_cmp_(key, layer->deque.back())) {
        return InsertPosition{std::in_place, layer, false};
      }
      if (!cell_cmp_(layer->deque.front(), key)) {
        return InsertPosition{std::in_place, layer, true};
      }
    }
    return {};
  }

  /**
   *  @brief
   *  Same as `getInsertPosition()`, but use binary search.
   */
  template<typename K>
  constexpr InsertPosition insertPositionBinarySearch(K const& key) {
    size_type const num_layers = layers_.size();
    if (num_layers == 0) {
      return {};
    }
    assert((!layers_.back().deque.empty()));
    if (!cell_cmp_(key, layers_.back().deque.back())) {
      layer_iterator layer{
          std::lower_bound(layers_.begin(), layers_.end(), key,
            [&cell_cmp = cell_cmp_](layer_type const& layer, K const& k) {
              return cell_cmp(k, layer.deque.back());
            }
          )};
      if (layer == layers_.end()) {
        return {};
      }
      return InsertPosition{std::in_place, layer, false};
    }
    if (!cell_cmp_(layers_.back().deque.front(), key)) {
      layer_iterator layer{
          std::lower_bound(layers_.begin(), layers_.end(), key,
            [&cell_cmp = cell_cmp_](layer_type const& layer, K const& k) {
              return cell_cmp(layer.deque.front(), k);
            })};
      if (layer == layers_.end()) {
        return {};
      }
      return InsertPosition{std::in_place, layer, true};
    }
    return {};
  }

  template<typename E>
  constexpr void forceInsert(E&& entry) {
    InsertPosition insert_position{insertPosition(cell_cmp_.ext_key(entry))};
    if (insert_position.has_value()) {
      auto& layer_insert_position = insert_position.value();
      if (layer_insert_position.front) {
        layer_insert_position.layer->deque.emplace_front(
            std::forward<E>(entry));
      } else {
        layer_insert_position.layer->deque.emplace_back(
            std::forward<E>(entry));
      }
    } else {
      layers_.emplace_back(cell_alloc_).deque.emplace_back(
          std::forward<E>(entry));
    }
    ++size_;
  }

  template<typename E>
  constexpr void forceInsertBinarySearch(E&& entry) {
    InsertPosition insert_position{
        insertPositionBinarySearch(cell_cmp_.ext_key(entry))};
    if (insert_position.has_value()) {
      auto& layer_insert_position = insert_position.value();
      if (layer_insert_position.front) {
        layer_insert_position.layer->deque.emplace_front(
            std::forward<E>(entry));
      } else {
        layer_insert_position.layer->deque.emplace_back(
            std::forward<E>(entry));
      }
    } else {
      layers_.emplace_back(cell_alloc_).deque.emplace_back(
          std::forward<E>(entry));
    }
    ++size_;
  }

  // --- Private member variables --- //

  // del_ent_(value_type&)
  delete_entry del_ent_;

  // cell_cmp_ contains ext_key and key_cmp
  cell_compare cell_cmp_;

  // Allocators.
  allocator_type value_alloc_;
  layer_allocator_type layer_alloc_;
  cell_allocator_type cell_alloc_;

  // List of layers.
  layer_list_type layers_;

  // Number of cells. `iterator::index_` will not exceed `num_cells_`.
  size_type num_cells_{0};

  // Number of active (not deleted) cells. `size()` will return this value.
  size_type size_{0};
};

template<typename Entry>
struct DefaultDeleteEntry {
  using entry_type = Entry;
  constexpr void operator()(entry_type& entry) const {}
};

// `std::identity` in C++20 could be used here. (The member typedefs are not
// used.)
template<typename Key>
struct ExtractKeyPassthrough {
  using key_type = Key;
  using value_type = Key const;
  constexpr key_type const& operator()(value_type const& v) const {
    return v;
  }
};

template<typename Key, typename Value>
struct ExtractKeyFromPair {
  using key_type = Key;
  using mapped_type = Value;
  using value_type = std::pair<key_type const, mapped_type>;
  constexpr key_type const& operator()(value_type const& v) const {
    return v.first;
  }
};

template<
    typename Key,
    typename Compare = std::less<Key>,
    typename Allocator = std::allocator<Key>,
    typename DeleteEntry = DefaultDeleteEntry<Key>,
    template<typename, typename> class DequeTemplate = std::deque,
    template<typename, typename> class LayerListTemplate = std::vector>
class FunnelSet : public FunnelBase<
    LayerListTemplate<
      FunnelLayer<
        DequeTemplate<
          FunnelCell<Key>,
          typename std::allocator_traits<Allocator>::
            template rebind_alloc<FunnelCell<Key>>>>,
      typename std::allocator_traits<Allocator>::
        template rebind_alloc<FunnelLayer<
          DequeTemplate<
            FunnelCell<Key>,
            typename std::allocator_traits<Allocator>::
              template rebind_alloc<FunnelCell<Key>>>>>>,
    DeleteEntry,
    Key,
    ExtractKeyPassthrough<Key>,
    Compare> {
 public:
  using this_type = 
      FunnelSet<Key, Compare, Allocator, DeleteEntry, LayerListTemplate>;
  using super_type = FunnelBase<
      LayerListTemplate<
        FunnelLayer<
          DequeTemplate<
            FunnelCell<Key>,
            typename std::allocator_traits<Allocator>::
              template rebind_alloc<FunnelCell<Key>>>>,
        typename std::allocator_traits<Allocator>::
          template rebind_alloc<FunnelLayer<
            DequeTemplate<
              FunnelCell<Key>,
              typename std::allocator_traits<Allocator>::
                template rebind_alloc<FunnelCell<Key>>>>>>,
      DeleteEntry,
      Key,
      ExtractKeyPassthrough<Key>,
      Compare>;

  using key_type = typename super_type::key_type;
  using delete_entry = typename super_type::delete_entry;
  using extract_key = typename super_type::extract_key;
  using key_compare = typename super_type::key_compare;
  using allocator_type = typename super_type::allocator_type;

  constexpr FunnelSet(
      delete_entry const& del_ent = delete_entry(),
      extract_key const& ext_key = extract_key(),
      key_compare const& key_cmp = key_compare(),
      allocator_type const& alloc = allocator_type())
    : super_type{del_ent, ext_key, key_cmp, alloc} {}

  friend class FunnelTest;
};

} // namespace funnel

