// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "CameraExampleViewController.h"
#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import "TensorFlowLite.h"

@interface CameraExampleViewController (InternalMethods)
- (void)setupAVCapture;
- (void)teardownAVCapture;
@end

@implementation CameraExampleViewController

- (void)setupAVCapture {
  NSError* error = nil;

  session = [AVCaptureSession new];
  if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone)
    [session setSessionPreset:AVCaptureSessionPreset640x480];
  else
    [session setSessionPreset:AVCaptureSessionPresetPhoto];

  AVCaptureDevice* device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
  AVCaptureDeviceInput* deviceInput =
      [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];

  if (error != nil) {
    NSLog(@"Failed to initialize AVCaptureDeviceInput. Note: This app doesn't work with simulator");
    assert(NO);
  }

  if ([session canAddInput:deviceInput]) [session addInput:deviceInput];

  videoDataOutput = [AVCaptureVideoDataOutput new];

  NSDictionary* rgbOutputSettings =
      [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                                  forKey:(id)kCVPixelBufferPixelFormatTypeKey];
  [videoDataOutput setVideoSettings:rgbOutputSettings];
  [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
  videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
  [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];

  if ([session canAddOutput:videoDataOutput]) [session addOutput:videoDataOutput];
  [[videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];

  previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
  [previewLayer setBackgroundColor:[[UIColor blackColor] CGColor]];
  [previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];
  CALayer* rootLayer = [previewView layer];
  [rootLayer setMasksToBounds:YES];
  [previewLayer setFrame:[rootLayer bounds]];
  [rootLayer addSublayer:previewLayer];
  [session startRunning];

  if (error) {
    NSString* title = [NSString stringWithFormat:@"Failed with error %d", (int)[error code]];
    UIAlertController* alertController =
        [UIAlertController alertControllerWithTitle:title
                                            message:[error localizedDescription]
                                     preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction* dismiss =
        [UIAlertAction actionWithTitle:@"Dismiss" style:UIAlertActionStyleDefault handler:nil];
    [alertController addAction:dismiss];
    [self presentViewController:alertController animated:YES completion:nil];
    [self teardownAVCapture];
  }
}

- (void)teardownAVCapture {
  [previewLayer removeFromSuperlayer];
}

- (AVCaptureVideoOrientation)avOrientationForDeviceOrientation:
    (UIDeviceOrientation)deviceOrientation {
  AVCaptureVideoOrientation result = (AVCaptureVideoOrientation)(deviceOrientation);
  if (deviceOrientation == UIDeviceOrientationLandscapeLeft)
    result = AVCaptureVideoOrientationLandscapeRight;
  else if (deviceOrientation == UIDeviceOrientationLandscapeRight)
    result = AVCaptureVideoOrientationLandscapeLeft;
  return result;
}

- (IBAction)takePicture:(id)sender {
  if ([session isRunning]) {
    [session stopRunning];
    [sender setTitle:@"Continue" forState:UIControlStateNormal];

    flashView = [[UIView alloc] initWithFrame:[previewView frame]];
    [flashView setBackgroundColor:[UIColor whiteColor]];
    [flashView setAlpha:0.f];
    [[[self view] window] addSubview:flashView];

    [UIView animateWithDuration:.2f
        animations:^{
          [flashView setAlpha:1.f];
        }
        completion:^(BOOL finished) {
          [UIView animateWithDuration:.2f
              animations:^{
                [flashView setAlpha:0.f];
              }
              completion:^(BOOL finished) {
                [flashView removeFromSuperview];
                flashView = nil;
              }];
        }];

  } else {
    [session startRunning];
    [sender setTitle:@"Freeze Frame" forState:UIControlStateNormal];
  }
}

- (void)captureOutput:(AVCaptureOutput*)captureOutput
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  CFRetain(pixelBuffer);
    [self.tensorFlowLite runModelOnFrame:pixelBuffer completion:^(NSDictionary * _Nonnull values) {
        [self setPredictionValues:values];
    }];
    /*
    [self.tensorFlowLite runModelOnFrameMtcnn:pixelBuffer completion:^(NSDictionary * _Nonnull values) {
        [self setPredictionValues:values];
    }];
    */
    
    
  CFRelease(pixelBuffer);
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
}

// TODO: read model??
- (void)viewDidLoad {
  [super viewDidLoad];
  labelLayers = [[NSMutableArray alloc] init];
  oldPredictionValues = [[NSMutableDictionary alloc] init];

    self.tensorFlowLite = [TensorFlowLite new];
    [self.tensorFlowLite setup];
    
  [self setupAVCapture];
}

// TODO: GPU setting
- (void)dealloc {
    [self teardownAVCapture];
}


- (void)viewDidUnload {
  [super viewDidUnload];
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];
}

- (void)viewDidDisappear:(BOOL)animated {
  [super viewDidDisappear:animated];
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
  return (interfaceOrientation == UIInterfaceOrientationPortrait);
}

- (BOOL)prefersStatusBarHidden {
  return YES;
}

// TODO: demo set output prediction to view
- (void)setPredictionValues:(NSDictionary*)newValues {
    NSLog(@"Result: %@", newValues);
//  const float decayValue = 0.75f;
//  const float updateValue = 0.25f;
//  const float minimumThreshold = 0.01f;
//  //initialize decayedPredictionValues
//  NSMutableDictionary* decayedPredictionValues = [[NSMutableDictionary alloc] init];
//  for (NSString* label in oldPredictionValues) {
//    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
//    const float oldPredictionValue = [oldPredictionValueObject floatValue];
//    const float decayedPredictionValue = (oldPredictionValue * decayValue);
//    if (decayedPredictionValue > minimumThreshold) {
//      NSNumber* decayedPredictionValueObject = [NSNumber numberWithFloat:decayedPredictionValue];
//      [decayedPredictionValues setObject:decayedPredictionValueObject forKey:label];
//    }
//  }
//  oldPredictionValues = decayedPredictionValues;
//
//  for (NSString* label in newValues) {
//    NSNumber* newPredictionValueObject = [newValues objectForKey:label];
//    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
//    if (!oldPredictionValueObject) {
//      oldPredictionValueObject = [NSNumber numberWithFloat:0.0f];
//    }
//    const float newPredictionValue = [newPredictionValueObject floatValue];
//    const float oldPredictionValue = [oldPredictionValueObject floatValue];
//    const float updatedPredictionValue = (oldPredictionValue + (newPredictionValue * updateValue));
//    NSNumber* updatedPredictionValueObject = [NSNumber numberWithFloat:updatedPredictionValue];
//    [oldPredictionValues setObject:updatedPredictionValueObject forKey:label];
//  }
//  NSArray* candidateLabels = [NSMutableArray array];
//  for (NSString* label in oldPredictionValues) {
//    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
//    const float oldPredictionValue = [oldPredictionValueObject floatValue];
//    if (oldPredictionValue > 0.05f) {
//      NSDictionary* entry = @{@"label" : label, @"value" : oldPredictionValueObject};
//      candidateLabels = [candidateLabels arrayByAddingObject:entry];
//    }
//  }
//  NSSortDescriptor* sort = [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
//  NSArray* sortedLabels =
//      [candidateLabels sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];
//
//  const float leftMargin = 10.0f;
//  const float topMargin = 10.0f;
//
//  const float valueWidth = 48.0f;
//  const float valueHeight = 18.0f;
//
//  const float labelWidth = 246.0f;
//  const float labelHeight = 18.0f;
//
//  const float labelMarginX = 5.0f;
//  const float labelMarginY = 5.0f;
//
//  [self removeAllLabelLayers];
//
//  int labelCount = 0;
//  for (NSDictionary* entry in sortedLabels) {
//    NSString* label = [entry objectForKey:@"label"];
//    NSNumber* valueObject = [entry objectForKey:@"value"];
//    const float value = [valueObject floatValue];
//    const float originY = topMargin + ((labelHeight + labelMarginY) * labelCount);
//    const int valuePercentage = (int)roundf(value * 100.0f);
//
//    const float valueOriginX = leftMargin;
//    NSString* valueText = [NSString stringWithFormat:@"%d%%", valuePercentage];
//
//    [self addLabelLayerWithText:valueText
//                        originX:valueOriginX
//                        originY:originY
//                          width:valueWidth
//                         height:valueHeight
//                      alignment:kCAAlignmentRight];
//
//    const float labelOriginX = (leftMargin + valueWidth + labelMarginX);
//
//    [self addLabelLayerWithText:[label capitalizedString]
//                        originX:labelOriginX
//                        originY:originY
//                          width:labelWidth
//                         height:labelHeight
//                      alignment:kCAAlignmentLeft];
//
//    labelCount += 1;
//    if (labelCount > 4) {
//      break;
//    }
//  }
}

// TODO: mtcnn set output prediction to view
- (void)setPredictionValuesMtcnn:(NSDictionary*)newValues {
    const float decayValue = 0.75f;
    const float updateValue = 0.25f;
    const float minimumThreshold = 0.01f;
    //initialize decayedPredictionValues
    NSMutableDictionary* decayedPredictionValues = [[NSMutableDictionary alloc] init];
    for (NSString* label in oldPredictionValues) {
        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        const float decayedPredictionValue = (oldPredictionValue * decayValue);
        if (decayedPredictionValue > minimumThreshold) {
            NSNumber* decayedPredictionValueObject = [NSNumber numberWithFloat:decayedPredictionValue];
            [decayedPredictionValues setObject:decayedPredictionValueObject forKey:label];
        }
    }
    oldPredictionValues = decayedPredictionValues;
    
    for (NSString* label in newValues) {
        NSNumber* newPredictionValueObject = [newValues objectForKey:label];
        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
        if (!oldPredictionValueObject) {
            oldPredictionValueObject = [NSNumber numberWithFloat:0.0f];
        }
        const float newPredictionValue = [newPredictionValueObject floatValue];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        const float updatedPredictionValue = (oldPredictionValue + (newPredictionValue * updateValue));
        NSNumber* updatedPredictionValueObject = [NSNumber numberWithFloat:updatedPredictionValue];
        [oldPredictionValues setObject:updatedPredictionValueObject forKey:label];
    }
    NSArray* candidateLabels = [NSMutableArray array];
    for (NSString* label in oldPredictionValues) {
        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        if (oldPredictionValue > 0.05f) {
            NSDictionary* entry = @{@"label" : label, @"value" : oldPredictionValueObject};
            candidateLabels = [candidateLabels arrayByAddingObject:entry];
        }
    }
    NSSortDescriptor* sort = [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
    NSArray* sortedLabels =
    [candidateLabels sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];
    
    const float leftMargin = 10.0f;
    const float topMargin = 10.0f;
    
    const float valueWidth = 48.0f;
    const float valueHeight = 18.0f;
    
    const float labelWidth = 246.0f;
    const float labelHeight = 18.0f;
    
    const float labelMarginX = 5.0f;
    const float labelMarginY = 5.0f;
    
    [self removeAllLabelLayers];
    
    int labelCount = 0;
    for (NSDictionary* entry in sortedLabels) {
        NSString* label = [entry objectForKey:@"label"];
        NSNumber* valueObject = [entry objectForKey:@"value"];
        const float value = [valueObject floatValue];
        const float originY = topMargin + ((labelHeight + labelMarginY) * labelCount);
        const int valuePercentage = (int)roundf(value * 100.0f);
        
        const float valueOriginX = leftMargin;
        NSString* valueText = [NSString stringWithFormat:@"%d%%", valuePercentage];
        
        [self addLabelLayerWithText:valueText
                            originX:valueOriginX
                            originY:originY
                              width:valueWidth
                             height:valueHeight
                          alignment:kCAAlignmentRight];
        
        const float labelOriginX = (leftMargin + valueWidth + labelMarginX);
        
        [self addLabelLayerWithText:[label capitalizedString]
                            originX:labelOriginX
                            originY:originY
                              width:labelWidth
                             height:labelHeight
                          alignment:kCAAlignmentLeft];
        
        labelCount += 1;
        if (labelCount > 4) {
            break;
        }
    }
}

// TODO: なんかlabel削除してる、dont need
- (void)removeAllLabelLayers {
  for (CATextLayer* layer in labelLayers) {
    [layer removeFromSuperlayer];
  }
  [labelLayers removeAllObjects];
}

// TODO: labelを画面に追加してる
- (void)addLabelLayerWithText:(NSString*)text
                      originX:(float)originX
                      originY:(float)originY
                        width:(float)width
                       height:(float)height
                    alignment:(NSString*)alignment {
  CFTypeRef font = (CFTypeRef) @"Menlo-Regular";
  const float fontSize = 12.0;
  const float marginSizeX = 5.0f;
  const float marginSizeY = 2.0f;

  const CGRect backgroundBounds = CGRectMake(originX, originY, width, height);
  const CGRect textBounds = CGRectMake((originX + marginSizeX), (originY + marginSizeY),
                                       (width - (marginSizeX * 2)), (height - (marginSizeY * 2)));

  CATextLayer* background = [CATextLayer layer];
  [background setBackgroundColor:[UIColor blackColor].CGColor];
  [background setOpacity:0.5f];
  [background setFrame:backgroundBounds];
  background.cornerRadius = 5.0f;

  [[self.view layer] addSublayer:background];
  [labelLayers addObject:background];

  CATextLayer* layer = [CATextLayer layer];
  [layer setForegroundColor:[UIColor whiteColor].CGColor];
  [layer setFrame:textBounds];
  [layer setAlignmentMode:alignment];
  [layer setWrapped:YES];
  [layer setFont:font];
  [layer setFontSize:fontSize];
  layer.contentsScale = [[UIScreen mainScreen] scale];
  [layer setString:text];

  [[self.view layer] addSublayer:layer];
  [labelLayers addObject:layer];
}

@end
