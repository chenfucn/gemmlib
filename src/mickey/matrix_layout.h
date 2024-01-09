/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    matrix_layout.h
 *
 * Abstract:
 *   Utils for simplifying positioning and striding in tensors. Inspired
 *   by CUTLASS, striving for 0 runtime cost while promote safety.
 *
 *   Only supports 2D tensors (matrix) for now.
 */

#pragma once

#include <cstdint>
#include <gsl/gsl>

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/subbyte_reference.h"

//#include "cutlass/util/host_tensor.h"

#if defined(_MSC_VER)
#define M_FORCEINLINE __forceinline
#else
#define M_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace mickey {

/**
 * @brief Another impl of cutlass::TensorRef with actual bounds check.
 */
template <
    /// Data type of element stored within tensor, must be numerical types
    typename Element_,
    /// Defines a mapping from logical coordinate to linear memory offsets
    typename Layout_,
    /// If true, extra bounds checking is performed on all accesses
    bool ExtraBoundsCheck_ = false>
class MatrixRef {
 public:
  /// Data type of individual access
  using Element = Element_;

  /// Mapping function from logical coordinate to linear memory
  using Layout = Layout_;

  /// Reference type to an element
  using Reference = typename cutlass::platform::conditional<
    cutlass::sizeof_bits<Element>::value >= 8,
    Element &,
    cutlass::SubbyteReference<Element>
    >::type;

  /// Logical rank of tensor index space
  static int const kRank = Layout::kRank;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using TensorCoord = typename Layout::TensorCoord;

  /// Layout's stride vector
  using Stride = typename Layout::Stride;

  /// TensorRef to constant data
  using ConstMatrixRef = MatrixRef<
    typename cutlass::platform::remove_const<Element>::type const,
    Layout, ExtraBoundsCheck_>;

  /// TensorRef to non-constant data
  using NonConstMatrixRef = MatrixRef<
    typename cutlass::platform::remove_const<Element>::type,
    Layout, ExtraBoundsCheck_>;

  /// Require at least rank=1. Mathematically, a rank=0 tensor would be considered to be a
  /// scalar, but degenerate cases such as these are difficult to accommodate without
  /// extensive C++ metaprogramming or support for zero-length arrays.
  static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

  static constexpr bool IsNonConstRef = cutlass::platform::is_same<NonConstMatrixRef, MatrixRef<Element_, Layout_>>::value;

 private:
  /// Pointer to data
  Element* ptr_;

  /// Shape of matrix
  TensorCoord shape_;

  /// Layout object maps logical coordinates to linear offsets
  Layout layout_;

 public:
  CUTLASS_HOST_DEVICE
  MatrixRef() : ptr_(nullptr) {}

  MatrixRef(
      gsl::span<Element> const& data,  ///< pointer to start of tensor
      TensorCoord const& shape            ///< shape of tensor
      ) : ptr_(data.data()), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data.size() >= size_t(shape_.product()));
  }

  CUTLASS_HOST_DEVICE
  MatrixRef(
      Element* ptr,          ///< pointer to start of tensor
      LongIndex size,        ///< size of tensor in elements
      TensorCoord const& shape  ///< shape of tensor
      ) : ptr_(ptr), shape_(shape), layout_(Layout::packed(shape)) {
    if (size < shape_.product()) {
#if defined(__CUDA_ARCH__)
      printf("MatrixRef: size %lld is smaller than required shape: [%d, %d]\n", size, shape_[0], shape_[1]);
      assert(false);
#else
      throw std::runtime_error("MatrixRef: size " + std::to_string(size) + " is smaller than required shape: [" + std::to_string(shape_[0]) + ", " + std::to_string(shape_[1]) + "]");
#endif
    }
  }

  /// Converting constructor from MatrixRef to non-constant data.
  template <typename _Magic = int>
  CUTLASS_HOST_DEVICE
  MatrixRef(
      NonConstMatrixRef const& ref,  ///< MatrixRef to non-const data
      /// SFINAE trick to avoid creating a copy-constructor when Element_ is already non-const
      _Magic magic = (typename cutlass::platform::enable_if<!IsNonConstRef, _Magic>::type)0
      ) : ptr_(ref.data()), shape_(ref.shape()), layout_(Layout::packed(ref.shape())) {}

  CUTLASS_HOST_DEVICE
  ConstMatrixRef const_ref() const {
    return ConstMatrixRef(ptr_, shape_.product(), shape_);
  }

  CUTLASS_HOST_DEVICE
  NonConstMatrixRef non_const_ref() {
    return NonConstMatrixRef(
        const_cast<typename std::remove_const<Element>::type*>(ptr_),
        shape_.product(), shape_);
  }

  /// Returns true if the MatrixRef is non-null
  CUTLASS_HOST_DEVICE
  bool good() const { return (ptr_ != nullptr) && shape_.product() > 0; }

  CUTLASS_HOST_DEVICE
  Element* data() const { return ptr_; }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Reference data(LongIndex idx) const {
    return cutlass::ReferenceFactory<typename cutlass::platform::remove_const<Element>::type,
                            (cutlass::sizeof_bits<Element>::value < 8)>::get(ptr_, idx);
  }

  CUTLASS_HOST_DEVICE
  TensorCoord const& shape() const { return shape_; }

  CUTLASS_HOST_DEVICE
  Layout& layout() { return layout_; }

  CUTLASS_HOST_DEVICE
  Layout layout() const { return layout_; }

  CUTLASS_HOST_DEVICE
  Index stride() const { return layout_.stride(); }

  CUTLASS_HOST_DEVICE
  Index& stride() { return layout_.stride(); }

  /// Computes the offset of an index from the origin of the tensor
  CUTLASS_HOST_DEVICE
  LongIndex offset(TensorCoord const& coord) const {
    if constexpr (ExtraBoundsCheck_) {
      assert(coord[0] >= 0 && coord[0] < shape_[0]);
      assert(coord[1] >= 0 && coord[1] < shape_[1]);
    }
    return layout_(coord);
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference at(TensorCoord const& coord) const {
    return data(offset(coord));
  }

  CUTLASS_HOST_DEVICE
  Reference at(int64_t row, int64_t col) const {
    return data(offset(TensorCoord(row, col)));
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference operator[](TensorCoord const& coord) const {
    return data(offset(coord));
  }
};

/// Constructs a MatrixRef, deducing types from arguments.
template <
    typename Element,
    typename Layout,
    bool ExtraBoundsCheck = false>
M_FORCEINLINE
MatrixRef<Element, Layout, ExtraBoundsCheck>
make_MatrixRef(
    Element* ptr,
    int64_t size,
    typename Layout::TensorCoord const& shape) {
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(ptr, size, shape);
}

template <
    typename Element,
    typename Layout,
    bool ExtraBoundsCheck = false>
M_FORCEINLINE
MatrixRef<Element, Layout, ExtraBoundsCheck>
make_MatrixRef(
    const gsl::span<Element>& span,
    typename Layout::TensorCoord const& shape) {
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(span, shape);
}

//
// Converting cutlass tensor to MatrixRef
//
/*
template <
  typename Element,
  typename Layout,
  bool ExtraBoundsCheck = false>
M_FORCEINLINE
MatrixRef<Element, Layout, ExtraBoundsCheck> make_MatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.host_data());
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(ptr, tensor.capacity(), tensor.extent());
}

template <
  typename Element,
  typename Layout,
  bool ExtraBoundsCheck = false>
M_FORCEINLINE
MatrixRef<Element, Layout, ExtraBoundsCheck> make_DevMatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.device_data());
  if (ptr == nullptr) {
    return MatrixRef<Element, Layout, ExtraBoundsCheck>();
  }
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(ptr, tensor.capacity(), tensor.extent());
}

template <
  typename Element,
  typename Layout,
  bool ExtraBoundsCheck>
M_FORCEINLINE
MatrixRef<Element const, Layout, ExtraBoundsCheck> make_ConstMatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  return MatrixRef<Element const, Layout, ExtraBoundsCheck>(tensor.host_data(), tensor.capacity(), tensor.extent());
}

template <
  typename Element,
  typename Layout,
  bool ExtraBoundsCheck>
M_FORCEINLINE
MatrixRef<Element const, Layout, ExtraBoundsCheck> make_DevConstMatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.device_data());
  if (ptr == nullptr) {
    return MatrixRef<Element const, Layout, ExtraBoundsCheck>();
  }
  return MatrixRef<Element const, Layout, ExtraBoundsCheck>(ptr, tensor.capacity(), tensor.extent());
}
*/

// clang-format off

}  // namespace mickey
