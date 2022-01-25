/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_COMBINEDNMS_PLUGIN_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_COMBINEDNMS_PLUGIN_H_

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/tensorrt/plugin/efficientNMSPlugin/efficientNMSPlugin.h"

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  using EfficientNMSOutputsDataType = void* const*;
#else
  using EfficientNMSOutputsDataType = void**;
#endif

namespace tensorflow {
namespace tensorrt {

class EfficientNMSImplicitPlugin : public nvinfer1::IPluginV2IOExt {
 public:
  explicit EfficientNMSImplicitPlugin(EfficientNMSParameters param);
  EfficientNMSImplicitPlugin(const void* data, size_t length);
  ~EfficientNMSImplicitPlugin() override = default;

  // IPluginV2 methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* libNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

  nvinfer1::Dims getOutputDimensions(int outputIndex,
                                     const nvinfer1::Dims* inputs,
                                     int nbInputs) noexcept override;
  size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
  int enqueue(int batchSize, void const* const* inputs, EfficientNMSOutputsDataType outputs,
              void* workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext methods
  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputType,
                                       int nbInputs) const noexcept override;
  nvinfer1::IPluginV2IOExt* clone() const noexcept override;
  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    bool const* inputIsBroadcasted,
                                    int nbInputs) const noexcept override;

  // IPluginV2IOExt methods
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) const noexcept override;
  void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInputs,
                       const nvinfer1::PluginTensorDesc* out,
                       int nbOutputs) noexcept override;

 private:
  EfficientNMSParameters mParam{};
  std::string mNamespace;
};

class CombinedNMSPluginCreator : public BaseCreator {
 public:
  CombinedNMSPluginCreator();
  ~CombinedNMSPluginCreator() override = default;

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2DynamicExt* deserializePlugin(
      const char* name, const void* serialData,
      size_t serialLength) noexcept override;

 protected:
  nvinfer1::PluginFieldCollection mFC;
  EfficientNMSParameters mParam;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mPluginName;
};

class CombinedNMSImplicitPluginCreator : public BaseCreator {
 public:
  CombinedNMSImplicitPluginCreator();
  ~CombinedNMSImplicitPluginCreator() override = default;

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2IOExt* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2IOExt* deserializePlugin(
      const char* name, const void* serialData,
      size_t serialLength) noexcept override;

 protected:
  nvinfer1::PluginFieldCollection mFC;
  EfficientNMSParameters mParam;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mPluginName;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_COMBINEDNMS_PLUGIN_H_
