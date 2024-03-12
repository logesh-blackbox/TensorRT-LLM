#pragma once

#include <cassert>
#include <cuda_runtime_api.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/common/pluginFields.h"
#include <cuda_runtime_api.h>
#include <nvinfer1/IBuilder.h>
#include <nvinfer1/IPluginV2.h>
#include <nvinfer1/IPluginV2DynamicExt.h>
#include <nvinfer1/ITensor.h>
#include <nvinfer1/ITensorDesc.h>
#include <nvinfer1/PluginField.h>
#include <nvinfer1/PluginFieldCollection.h>

namespace tensorrt_llm::plugins
{

class BertAttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    // Default constructor is deleted
    BertAttentionPlugin() = delete;

    // Constructor with required parameters
    BertAttentionPlugin(int num_heads, int head_size, float q_scaling, bool qk_half_accum,
        tensorrt_llm::kernels::ContextFMHAType context_fmha_type, nvinfer1::DataType type,
        bool do_relative_attention = false, int max_distance = 0);

    // Constructor for deserialization
    BertAttentionPlugin(const void* data, size_t length);

    // Destructor
    ~BertAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // Enqueue implementation for specific data types
    template <typename T>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    // Layer name
    const std::string mLayerName;

    // Parameters for the attention mechanism
    int mNumHeads;
    int mHeadSize;
    int mMaxInputLength;
    float mQScaling;
    nvinfer1::DataType mType;
    bool mRelativeAttention = false;
    int mMaxDistance = 0;

    // Unfused
