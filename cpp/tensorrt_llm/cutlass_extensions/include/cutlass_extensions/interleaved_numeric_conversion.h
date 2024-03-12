#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.
template <typename T, typename S, int N>
struct FastInterleavedAndBiasedNumericArrayConverter {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    uint32_t *h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t mask_for_elt_01 = 0x5250;
    static constexpr uint32_t mask_for_elt_23 = 0x5351;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed integer as fp16.
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, N> {
  static constexpr int VEC_WIDTH = 4;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

  using result_type = Array<half_t, N>;
  using source_type = Array<uint8_t, N>;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
        convert_vector_;

    result_type result;
    using vec_result = Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = Array<scalar_source_type, VEC_WIDTH>;

    vec_result *result_ptr = reinterpret_cast<vec_result*>(&result);
    vec_source const *source_ptr = reinterpret_cast<vec_source const*>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, 4> {
  using result_type = Array<bfloat16_t, 4>;
  using source_type = Array<uint8_t, 4>;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

    uint32_t *bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t fp32_base = 0x4B000000;
    float fp32_intermediates[4];

    // Construct FP32s, bfloat does not have enough mantissa for IADD trick
    uint32_t *fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
    fp32_intermediates_casted[0] = __byte_perm(
