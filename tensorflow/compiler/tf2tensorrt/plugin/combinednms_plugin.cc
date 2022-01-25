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

#include "tensorflow/compiler/tf2tensorrt/plugin/combinednms_plugin.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/tensorrt/plugin/efficientNMSPlugin/efficientNMSInference.h"

namespace tensorflow {
namespace tensorrt {

EfficientNMSImplicitPlugin::EfficientNMSImplicitPlugin(
    EfficientNMSParameters param)
    : mParam(param) {}

EfficientNMSImplicitPlugin::EfficientNMSImplicitPlugin(const void* data,
                                                       size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  mParam = read<EfficientNMSParameters>(d);
  ASSERT(d == a + length);
}

const char* EfficientNMSImplicitPlugin::getPluginType() const noexcept {
  return "EfficientNMS_Implicit_TRT";
}

const char* EfficientNMSImplicitPlugin::getPluginVersion() const noexcept {
  return "1";
}

int EfficientNMSImplicitPlugin::getNbOutputs() const noexcept { return 4; }

int EfficientNMSImplicitPlugin::initialize() noexcept { return STATUS_SUCCESS; }

void EfficientNMSImplicitPlugin::terminate() noexcept {}

size_t EfficientNMSImplicitPlugin::getSerializationSize() const noexcept {
  return sizeof(EfficientNMSParameters);
}

void EfficientNMSImplicitPlugin::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, mParam);
  ASSERT(d == a + getSerializationSize());
}

void EfficientNMSImplicitPlugin::destroy() noexcept { delete this; }

void EfficientNMSImplicitPlugin::setPluginNamespace(
    const char* pluginNamespace) noexcept {
  mNamespace = pluginNamespace;
}

const char* EfficientNMSImplicitPlugin::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

nvinfer1::Dims EfficientNMSImplicitPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::Dims* inputs, int nbInputs) noexcept {
  nvinfer1::Dims out_dim;

  // When pad per class is set, the output size may need to be reduced:
  // i.e.: outputBoxes = min(outputBoxes, outputBoxesPerClass * numClasses)
  ASSERT(inputs[1].nbDims == 2);
  if (mParam.padOutputBoxesPerClass && mParam.numOutputBoxesPerClass > 0) {
    const int numClasses = inputs[1].d[1];
    if (mParam.numOutputBoxesPerClass * numClasses < mParam.numOutputBoxes) {
      mParam.numOutputBoxes = mParam.numOutputBoxesPerClass * numClasses;
    }
  }

  // Standard NMS
  ASSERT(outputIndex >= 0 && outputIndex <= 3);

  // num_detections
  if (outputIndex == 0) {
    out_dim.nbDims = 0;
    out_dim.d[0] = 0;
  }
  // detection_boxes
  else if (outputIndex == 1) {
    out_dim.nbDims = 2;
    out_dim.d[0] = mParam.numOutputBoxes;
    out_dim.d[1] = 4;
  }
  // detection_scores
  else if (outputIndex == 2) {
    out_dim.nbDims = 1;
    out_dim.d[0] = mParam.numOutputBoxes;
  }
  // detection_classes
  else if (outputIndex == 3) {
    out_dim.nbDims = 1;
    out_dim.d[0] = mParam.numOutputBoxes;
  }

  return out_dim;
}

size_t EfficientNMSImplicitPlugin::getWorkspaceSize(
    int maxBatchSize) const noexcept {
  return EfficientNMSWorkspaceSize(maxBatchSize, mParam.numScoreElements,
                                   mParam.numClasses, mParam.datatype);
}

int EfficientNMSImplicitPlugin::enqueue(int batchSize,
                                        void const* const* inputs,
                                        EfficientNMSOutputsDataType outputs, void* workspace,
                                        cudaStream_t stream) noexcept {
  mParam.batchSize = batchSize;

  void const* const boxesInput = inputs[0];
  void const* const scoresInput = inputs[1];
  void const* const anchorsInput = nullptr;

  void* numDetectionsOutput = outputs[0];
  void* nmsBoxesOutput = outputs[1];
  void* nmsScoresOutput = outputs[2];
  void* nmsClassesOutput = outputs[3];

  return EfficientNMSInference(mParam, boxesInput, scoresInput, anchorsInput,
                               numDetectionsOutput, nmsBoxesOutput,
                               nmsScoresOutput, nmsClassesOutput, nullptr,
                               workspace, stream);
}

bool EfficientNMSImplicitPlugin::canBroadcastInputAcrossBatch(
    int inputIndex) const noexcept {
  return true;
}

nvinfer1::DataType EfficientNMSImplicitPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes,
    int nbInputs) const noexcept {
  // num_detections and detection_classes use integer outputs
  if (index == 0 || index == 3) {
    return nvinfer1::DataType::kINT32;
  }
  // All others should use the same datatype as the input
  return inputTypes[0];
}

nvinfer1::IPluginV2IOExt* EfficientNMSImplicitPlugin::clone() const noexcept {
  auto* plugin = new EfficientNMSImplicitPlugin(mParam);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

bool EfficientNMSImplicitPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, bool const* inputIsBroadcasted,
    int nbInputs) const noexcept {
  return true;
}

bool EfficientNMSImplicitPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) const noexcept {
  if (inOut[pos].format != nvinfer1::PluginFormat::kLINEAR) {
    return false;
  }

  ASSERT(nbInputs == 2);
  ASSERT(nbOutputs == 4);
  if (nbInputs == 2) {
    ASSERT(0 <= pos && pos <= 5);
  }

  // num_detections and detection_classes output: int
  const int posOut = pos - nbInputs;
  if (posOut == 0 || posOut == 3) {
    return inOut[pos].type == nvinfer1::DataType::kINT32 &&
           inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
  }

  // all other inputs/outputs: fp32 or fp16
  return (inOut[pos].type == nvinfer1::DataType::kHALF ||
          inOut[pos].type == nvinfer1::DataType::kFLOAT) &&
         (inOut[0].type == inOut[pos].type);
}

void EfficientNMSImplicitPlugin::configurePlugin(
    const nvinfer1::PluginTensorDesc* in, int nbInputs,
    const nvinfer1::PluginTensorDesc* out, int nbOutputs) noexcept {
  // Inputs: [0] boxes, [1] scores
  ASSERT(nbInputs == 2);
  ASSERT(nbOutputs == 4);
  mParam.datatype = in[0].type;

  // Shape of scores input should be
  // [batch_size, num_boxes, num_classes] or [batch_size, num_boxes,
  // num_classes, 1]
  ASSERT(in[1].dims.nbDims == 2 ||
         (in[1].dims.nbDims == 3 && in[1].dims.d[2] == 1));
  mParam.numScoreElements = in[1].dims.d[0] * in[1].dims.d[1];
  mParam.numClasses = in[1].dims.d[1];

  // Shape of boxes input should be
  // [batch_size, num_boxes, 4] or [batch_size, num_boxes, 1, 4] or [batch_size,
  // num_boxes, num_classes, 4]
  ASSERT(in[0].dims.nbDims == 2 || in[0].dims.nbDims == 3);
  if (in[0].dims.nbDims == 2) {
    ASSERT(in[0].dims.d[1] == 4);
    mParam.shareLocation = true;
    mParam.numBoxElements = in[0].dims.d[0] * in[0].dims.d[1];
  } else {
    mParam.shareLocation = (in[0].dims.d[1] == 1);
    ASSERT(in[0].dims.d[1] == mParam.numClasses || mParam.shareLocation);
    ASSERT(in[0].dims.d[2] == 4);
    mParam.numBoxElements = in[0].dims.d[0] * in[0].dims.d[1] * in[0].dims.d[2];
  }
  mParam.numAnchors = in[0].dims.d[0];

  if (nbInputs == 2) {
    mParam.boxDecoder = false;
  }
}

CombinedNMSPluginCreator::CombinedNMSPluginCreator() : mParam{} {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("max_output_size_per_class", nullptr,
                            nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "max_total_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "pad_per_class", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "clip_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* CombinedNMSPluginCreator::getPluginName() const {
  return "CombinedNMS_Plugin";
}

const char* CombinedNMSPluginCreator::getPluginVersion() const { return "1"; }

const nvinfer1::PluginFieldCollection*
CombinedNMSPluginCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt* CombinedNMSPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "max_output_size_per_class")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.numOutputBoxesPerClass =
          *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "max_total_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "iou_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "score_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "pad_per_class")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.padOutputBoxesPerClass =
          *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "clip_boxes")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.clipBoxes = *(static_cast<const int*>(fields[i].data));
    }
  }
  mParam.scoreBits = -1;

  auto* plugin = new EfficientNMSPlugin(mParam);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::IPluginV2DynamicExt* CombinedNMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call EfficientNMSPlugin::destroy()
  auto* plugin = new EfficientNMSPlugin(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

CombinedNMSImplicitPluginCreator::CombinedNMSImplicitPluginCreator()
    : mParam{} {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("max_output_size_per_class", nullptr,
                            nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "max_total_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "pad_per_class", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "clip_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* CombinedNMSImplicitPluginCreator::getPluginName() const {
  return "CombinedNMS_Implicit_Plugin";
}

const char* CombinedNMSImplicitPluginCreator::getPluginVersion() const {
  return "1";
}

const nvinfer1::PluginFieldCollection*
CombinedNMSImplicitPluginCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2IOExt* CombinedNMSImplicitPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "max_output_size_per_class")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.numOutputBoxesPerClass =
          *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "max_total_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "iou_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "score_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "pad_per_class")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.padOutputBoxesPerClass =
          *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "clip_boxes")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mParam.clipBoxes = *(static_cast<const int*>(fields[i].data));
    }
  }
  mParam.scoreBits = -1;

  auto* plugin = new EfficientNMSImplicitPlugin(mParam);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::IPluginV2IOExt* CombinedNMSImplicitPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call EfficientNMSPlugin::destroy()
  auto* plugin = new EfficientNMSImplicitPlugin(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
