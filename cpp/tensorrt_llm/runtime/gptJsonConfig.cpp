#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stringUtils.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace
{
    using Json = typename nlohmann::json::basic_json;

    template <typename FieldType>
    FieldType parseJsonField(Json const& json, std::string_view name, FieldType defaultValue)
    {
        auto value = defaultValue;
        if (json.contains(name))
        {
            auto jsonValue = json.at(name);
            if (jsonValue.is_number_integer())
            {
                value = jsonValue.template get<FieldType>();
            }
            else if (jsonValue.is_number_float())
            {
                value = jsonValue.template get<FieldType>();
            }
            else if (jsonValue.is_boolean())
            {
                value = jsonValue.template get<FieldType>();
            }
            else if (jsonValue.is_string())
            {
                value = jsonValue.template get<std::string>().c_str();
            }
           
