#ifndef ALLGATHER_PLUGIN_H
#define ALLGATHER_PLUGIN_H

#include <nccl.h>
#include <vector>
#include <unordered_set>
#include <type_traits>
#include "plugin.h"
#include "plugin_factory.h"

namespace tensorrt_llm {
namespace plugins {

class AllgatherPlugin : public IPluginV2DynamicExt {
public:
    AllgatherPlugin(std::unordered_set<int> group, nvinfer1::DataType type);
    AllgatherPlugin(const void* data, size_t length);
    ~AllgatherPlugin() override;

    int getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    std::unordered_set<int> mGroup;
    nvinfer1::DataType mType;
};

class AllgatherPluginCreator : public IPluginCreator {
public:
    AllgatherPluginCreator();
    ~AllgatherPluginCreator() override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    std::vector<PluginField> mPluginAttributes;
    constexpr static const char* mNamespace = "tensorrt_llm";
};

} // namespace plugins
} // namespace tensorrt_llm

#endif // ALLGATHER_PLUGIN_H
