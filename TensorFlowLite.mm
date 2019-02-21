//
//  TensorFlowLite.m
//  tflite_camera_example
//
//  Created by 納村 聡仁 on 2019/02/21.
//  Copyright © 2019 Google. All rights reserved.
//

#import "TensorFlowLite.h"

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#if TFLITE_USE_CONTRIB_LITE
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"
#include "tensorflow/contrib/lite/string_util.h"
#else
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"
#if TFLITE_USE_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
#endif

#define LOG(x) std::cerr

namespace {
    
    // If you have your own model, modify this to the file name, and make sure
    // you've added the file to your app resources too.
#if TFLITE_USE_GPU_DELEGATE
    // GPU Delegate only supports float model now.
    //NSString* model_file_name = @"mobilenet_v1_1.0_224";
    NSString* model_file_name = @"pnet";
#else
    //NSString* model_file_name = @"mobilenet_quant_v1_224";
    NSString* model_file_name = @"pnet";
#endif
    NSString* model_file_type = @"tflite";
    // If you have your own model, point this to the labels file.
    // TODO: face landmarks dont need labels
    NSString* labels_file_name = @"labels";
    NSString* labels_file_type = @"txt";
    
    // These dimensions need to match those the model was trained with.
    /*
     //mobilenet
     const int wanted_input_width = 224;
     const int wanted_input_height = 224;
     const int wanted_input_channels = 3;
     const float input_mean = 127.5f;
     const float input_std = 127.5f;
     const std::string input_layer_name = "input";
     const std::string output_layer_name = "softmax1";
    */
    // pnet
    const int wanted_input_width = 600;
    const int wanted_input_height = 800;
    const int wanted_input_channels = 3;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    const std::string input_layer_name = "input";
    
    
    NSString* FilePathForResourceName(NSString* name, NSString* extension) {
        NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
        if (file_path == NULL) {
            LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
            << "' in bundle.";
        }
        return file_path;
    }
    
    // TODO: dont need
    void LoadLabels(NSString* file_name, NSString* file_type, std::vector<std::string>* label_strings) {
        NSString* labels_path = FilePathForResourceName(file_name, file_type);
        if (!labels_path) {
            LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
            << [file_type UTF8String];
        }
        std::ifstream t;
        t.open([labels_path UTF8String]);
        std::string line;
        while (t) {
            std::getline(t, line);
            label_strings->push_back(line);
        }
        t.close();
    }
    
    // TODO: dont need
    // Returns the top N confidence values over threshold in the provided vector,
    // sorted by confidence in descending order.
    void GetTopN(
                 const float* prediction, const int prediction_size, const int num_results,
                 const float threshold, std::vector<std::pair<float, int> >* top_results) {
        // Will contain top N results in ascending order.
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
        std::greater<std::pair<float, int> > >
        top_result_pq;
        
        const long count = prediction_size;
        for (int i = 0; i < count; ++i) {
            const float value = prediction[i];
            // Only add it if it beats the threshold and has a chance at being in
            // the top N.
            if (value < threshold) {
                continue;
            }
            
            top_result_pq.push(std::pair<float, int>(value, i));
            
            // If at capacity, kick the smallest value out.
            if (top_result_pq.size() > num_results) {
                top_result_pq.pop();
            }
        }
        
        // Copy to output vector and reverse into descending order.
        while (!top_result_pq.empty()) {
            top_results->push_back(top_result_pq.top());
            top_result_pq.pop();
        }
        std::reverse(top_results->begin(), top_results->end());
    }
    
    // Preprocess the input image and feed the TFLite interpreter buffer for a float model.
    void ProcessInputWithFloatModel(
                                    uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
        for (int y = 0; y < wanted_input_height; ++y) {
            float* out_row = buffer + (y * wanted_input_width * wanted_input_channels);
            for (int x = 0; x < wanted_input_width; ++x) {
                const int in_x = (y * image_width) / wanted_input_width;
                const int in_y = (x * image_height) / wanted_input_height;
                uint8_t* input_pixel =
                input + (in_y * image_width * image_channels) + (in_x * image_channels);
                float* out_pixel = out_row + (x * wanted_input_channels);
                for (int c = 0; c < wanted_input_channels; ++c) {
                    out_pixel[c] = (input_pixel[c] - input_mean) / input_std;
                }
            }
        }
    }
    
    // Preprocess the input image and feed the TFLite interpreter buffer for a quantized model.
    void ProcessInputWithQuantizedModel(
                                        uint8_t* input, uint8_t* output, int image_width, int image_height, int image_channels) {
        for (int y = 0; y < wanted_input_height; ++y) {
            uint8_t* out_row = output + (y * wanted_input_width * wanted_input_channels);
            for (int x = 0; x < wanted_input_width; ++x) {
                const int in_x = (y * image_width) / wanted_input_width;
                const int in_y = (x * image_height) / wanted_input_height;
                uint8_t* in_pixel = input + (in_y * image_width * image_channels) + (in_x * image_channels);
                uint8_t* out_pixel = out_row + (x * wanted_input_channels);
                for (int c = 0; c < wanted_input_channels; ++c) {
                    out_pixel[c] = in_pixel[c];
                }
            }
        }
    }
    
}  // namespace

@implementation TensorFlowLite  {
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate* delegate;
}

- (void)setup {
    NSLog(@"setup");
    
    //TODO: about model settng
    NSString* graph_path = FilePathForResourceName(model_file_name, model_file_type);
    model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << graph_path;
    }
    LOG(INFO) << "Loaded model " << graph_path;
    model->error_reporter();
    LOG(INFO) << "resolved reporter";
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    LoadLabels(labels_file_name, labels_file_type, &labels);
    
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
#if TFLITE_USE_GPU_DELEGATE
    GpuDelegateOptions options;
    options.allow_precision_loss = true;
    options.wait_type = GpuDelegateOptions::WaitType::kActive;
    delegate = NewGpuDelegate(&options);
    interpreter->ModifyGraphWithDelegate(delegate);
#endif
    
    // TODO: about input data shape
    // Explicitly resize the input tensor.!!!!!!resizeをグラフ内でしてるのでは！！！！！！
    {
        int input = interpreter->inputs()[0];
        std::vector<int> sizes = {1, 224, 224, 3};
        interpreter->ResizeInputTensor(input, sizes);
    }
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter";
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }
}

// TODO: GPU setting
- (void)dealloc {
#if TFLITE_USE_GPU_DELEGATE
    if (delegate) {
        DeleteGpuDelegate(delegate);
    }
#endif
}

// TODO: model run
- (void)runModelOnFrame:(CVPixelBufferRef)pixelBuffer completion: (void (^)(NSDictionary *values))completion {
    assert(pixelBuffer != NULL);
    
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
           sourcePixelFormat == kCVPixelFormatType_32BGRA);
    
    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
    int image_height;
    unsigned char* sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    const int image_channels = 4;
    assert(image_channels >= wanted_input_channels);
    uint8_t* in = sourceStartAddr;
    //input size確認！！！！！！！ここで先にresizeしてしまう！！！！！！
    
    int input = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input);
    
    bool is_quantized;
    switch (input_tensor->type) {
        case kTfLiteFloat32:
            is_quantized = false;
            break;
        case kTfLiteUInt8:
            is_quantized = true;
            break;
        default:
            NSLog(@"Input data type is not supported by this demo app.");
            return;
    }
    
    if (is_quantized) {
        uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
        ProcessInputWithQuantizedModel(in, out, image_width, image_height, image_channels);
    } else {
        float* out = interpreter->typed_tensor<float>(input);
        ProcessInputWithFloatModel(in, out, image_width, image_height, image_channels);
    }
    
    double start = [[NSDate new] timeIntervalSince1970];
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    double end = [[NSDate new] timeIntervalSince1970];
    total_latency += (end - start);
    total_count += 1;
    NSLog(@"Time: %.4lf, avg: %.4lf, count: %d", end - start, total_latency / total_count,
          total_count);
    //TODO: get output
    // read output size from the output sensor
    /*
    const int output_tensor_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
    TfLiteIntArray* output_dims = output_tensor->dims;
    if (output_dims->size != 2 || output_dims->data[0] != 1) {
        LOG(FATAL) << "Output of the model is in invalid format.";
    }
    //TODO: image classifier has 1001 class, so output_size in classifier has 1001
    const int output_size = output_dims->data[1];
    
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    
    std::vector<std::pair<float, int> > top_results;
    
    //TODO: get top10??post process of output
    if (is_quantized) {
        uint8_t* quantized_output = interpreter->typed_output_tensor<uint8_t>(0);
        int32_t zero_point = input_tensor->params.zero_point;
        float scale = input_tensor->params.scale;
        float output[output_size];
        for (int i = 0; i < output_size; ++i) {
            output[i] = (quantized_output[i] - zero_point) * scale;
        }
        GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
        //int size = interpreter->tensor(0)->dims->data[3];
        //std::cout << "vector size: " << size << std::endl;
        //for (int i = 0; i < size; ++i){
        //    std::cout << output[i] << " ";
        //}
        //std::cout << std::endl;
     
    } else {
        float* output = interpreter->typed_output_tensor<float>(0);
        GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
        
        //int size = interpreter->tensor(0)->dims->data[0];
        ///std::cout << "vector size: " << size << std::endl;
    }
    
    //TODO: shape for putting view
    NSMutableDictionary* newValues = [NSMutableDictionary dictionary];
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        NSString* labelObject = [NSString stringWithUTF8String:labels[index].c_str()];
        NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
        [newValues setObject:valueObject forKey:labelObject];
    }
    */
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    /*
    //TODO: set output in the view
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        completion(newValues);
    });
    */
}

// TODO: mtcnn model run
- (void)runModelOnFrameMtcnn:(CVPixelBufferRef)pixelBuffer completion: (void (^)(NSDictionary *values))completion {
    assert(pixelBuffer != NULL);
    
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
           sourcePixelFormat == kCVPixelFormatType_32BGRA);
    
    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
    int image_height;
    unsigned char* sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    const int image_channels = 4;
    assert(image_channels >= wanted_input_channels);
    uint8_t* in = sourceStartAddr;
    
    int input = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input);
    
    bool is_quantized;
    switch (input_tensor->type) {
        case kTfLiteFloat32:
            is_quantized = false;
            break;
        case kTfLiteUInt8:
            is_quantized = true;
            break;
        default:
            NSLog(@"Input data type is not supported by this demo app.");
            return;
    }
    
    if (is_quantized) {
        uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
        ProcessInputWithQuantizedModel(in, out, image_width, image_height, image_channels);
    } else {
        float* out = interpreter->typed_tensor<float>(input);
        ProcessInputWithFloatModel(in, out, image_width, image_height, image_channels);
    }
    
    double start = [[NSDate new] timeIntervalSince1970];
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    double end = [[NSDate new] timeIntervalSince1970];
    total_latency += (end - start);
    total_count += 1;
    NSLog(@"Time: %.4lf, avg: %.4lf, count: %d", end - start, total_latency / total_count,
          total_count);
    NSLog(@"mtcnn model sample try");
    //TODO: get output
    // read output size from the output sensor
    const int output_tensor_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
    TfLiteIntArray* output_dims = output_tensor->dims;
    if (output_dims->size != 2 || output_dims->data[0] != 1) {
        LOG(FATAL) << "Output of the model is in invalid format.";
    }
    const int output_size = output_dims->data[1];
    
    /*
     //const int kNumResults = 5;
     //const float kThreshold = 0.1f;
     
     std::vector<std::pair<float, int> > top_results;
     
     //TODO: get top10??post process of output
     if (is_quantized) {
     uint8_t* quantized_output = interpreter->typed_output_tensor<uint8_t>(0);
     int32_t zero_point = input_tensor->params.zero_point;
     float scale = input_tensor->params.scale;
     float output[output_size];
     for (int i = 0; i < output_size; ++i) {
     output[i] = (quantized_output[i] - zero_point) * scale;
     }
     //GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
     } else {
     float* output = interpreter->typed_output_tensor<float>(0);
     //GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
     }
     
     
     //TODO: shape for putting view
     NSMutableDictionary* newValues = [NSMutableDictionary dictionary];
     for (const auto& result : top_results) {
     const float confidence = result.first;
     const int index = result.second;
     NSString* labelObject = [NSString stringWithUTF8String:labels[index].c_str()];
     NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
     [newValues setObject:valueObject forKey:labelObject];
     }
     //TODO: set output in the view
     dispatch_async(dispatch_get_main_queue(), ^(void) {
     [self setPredictionValues:newValues];
     });
     */
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

@end
