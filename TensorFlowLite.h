//
//  TensorFlowLite.h
//  tflite_camera_example
//
//  Created by 納村 聡仁 on 2019/02/21.
//  Copyright © 2019 Google. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import "TensorFlowLite.h"

#include <vector>

// TensorFlow Lite was migrated out of `contrib/` directory. The change
// wasn't reflected in newest CocoaPod release yet (1.12.0).
// Change this to 0 when using a TFLite version which is newer than 1.12.0.
// TODO(ycling): Remove the macro when we release the next version.
#ifndef TFLITE_USE_CONTRIB_LITE
//#define TFLITE_USE_CONTRIB_LITE 1
#define TFLITE_USE_CONTRIB_LITE 0
#endif

// Set TFLITE_USE_GPU_DELEGATE to 1 to use TFLite GPU Delegate.
// Note: TFLite GPU Delegate binary isn't releast yet, and we're working
// on it.
#ifndef TFLITE_USE_GPU_DELEGATE
//#define TFLITE_USE_GPU_DELEGATE 0
#define TFLITE_USE_GPU_DELEGATE 1
#endif

#if TFLITE_USE_GPU_DELEGATE && TFLITE_USE_CONTRIB_LITE
// Sanity check.
#error "GPU Delegate only works with newer TFLite " \
"after migrating out of contrib"
#endif

NS_ASSUME_NONNULL_BEGIN

@interface TensorFlowLite : NSObject {
    std::vector<std::string> labels;
    double total_latency;
    int total_count;
}

- (void)setup;
- (void)runModelOnFrame:(CVPixelBufferRef)pixelBuffer completion: (void (^)(NSDictionary *values))completion;
- (void)runModelOnFrameMtcnn:(CVPixelBufferRef)pixelBuffer completion: (void (^)(NSDictionary *values))completion;

@end

NS_ASSUME_NONNULL_END
