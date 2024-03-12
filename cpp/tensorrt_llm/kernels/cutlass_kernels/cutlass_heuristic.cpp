/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include <cuda_runtime_api.h>
#include <vector>

using namespace tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// Define a struct to represent the shape of a tile
struct TileShape
{
    int m;
    int n;
};

// Function to get the CTA shape for a given CutlassTileConfig
TileShape get_cta_shape_for_config(CutlassTileConfig tile_config)
{
    switch (tile_config)
    {
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64: return TileShape{32, 128};
    case CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64: return TileShape{64, 64};
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64: return TileShape{64, 128};
    case CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64: return TileShape{128, 64};
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64: return TileShape{128, 128};
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64: return TileShape{128, 256};
    case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64: return TileShape{256, 128};
    default: throw std::runtime_error("[TensorRT-LLm Error][get_grid_shape_for_config] Invalid config");
    }
}

// Function to check if a given split-k factor is valid for a given tile shape and problem size
bool is_valid_split_k_factor(const int64_t m, const int64_t n, const int64_t k, const TileShape tile_shape,
    const int split_k_factor, const size_t workspace_bytes, const bool is_weight_only)
{

    // All tile sizes have a k_tile of 64.
    static constexpr int k_tile = 64;

    // For weight-only quant, we need k and k_elements_per_split to be a multiple of cta_k
    if (is_weight_only)
    {
        if ((k % k_tile) != 0)
        {
            return false;
        }

        if ((k % split_k_factor) != 0)
        {
            return false;
        }

        const int k_elements_per_split = k / split_k_factor;
        if ((k_elements_per_split % k_tile) != 0)
        {
            return false;
        }
    }

    // Check that the workspace has sufficient space for this split-k factor
    const int ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
    const int ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;
    const int required_ws_bytes = split_k_factor == 1 ? 0 : sizeof(int) * ctas_in_m_dim * ctas_in_n_dim;

    if (required_ws_bytes > workspace_bytes)
    {
        return false;
    }

    return true;
}

// Function to get a vector of CutlassTileConfigs based on the given parameters
std::vector<CutlassTileConfig> get_candidate_tiles(
    const int sm, const bool is_weight_only, const bool simt_configs_only, const bool int8_configs_only)
{
    enum class CutlassGemmType : char
    {
        Default,
