#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <thread>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <camera.h>


/*
调用相机的接口
int main()
{
    bv::CameraCapture cap("video0", "test_cam_aaa");
    cv::Mat frame = cap.readCvMat();
    cv::imshow("test_cam_aaa", frame);
}
*/

// 互斥锁
std::mutex mutex_1;

// 每个阶段的条件变量
std::condition_variable stage_1;
std::condition_variable stage_2;


using namespace nvinfer1;


std::vector<cv::Mat> imageList;
std::vector<std::string> threadSquence;


static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;


void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}


void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}


void capture_show(bv::CameraCapture& cap,int videoNumbers){
    cv::Mat frame; 
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
    }
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        auto start = std::chrono::system_clock::now();
        frame = cap.readCvMat();
        auto end = std::chrono::system_clock::now();
        std::cout << "readCvMat time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // check if we succeeded
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        std::unique_lock<std::mutex> lock(mutex_1);
        stage_1.wait(lock, [&]
                                { return imageList.size() < videoNumbers; });
        imageList.push_back(frame);
        threadSquence.push_back(cap.getCameraName());
        stage_2.notify_one();
        
        
    }
}

int main(int argc, char** argv) {
  if (argc != 3)
  {
      std::cerr << "usage: " << argv[0] << " <engine_file> <video_numbers>(maximum equals 8!!!)" << std::endl;
      return -1;
  }

  std::string engine_name = argv[1];
  int video_numbers = std::stoi(argv[2]);

  cudaSetDevice(kGpuId);

  std::vector<std::string> deviceNames;
  std::vector<std::shared_ptr<bv::CameraCapture>> capList;
  


  for (size_t i = 0; i < video_numbers; i++)
  {
    std::string temp_name = "video" + std::to_string(i);
    //如果不使用std::shared_ptr会发生内存泄漏
    std::shared_ptr<bv::CameraCapture> temp_cap = std::make_shared<bv::CameraCapture>(temp_name,temp_name);
    capList.push_back(temp_cap);
  }


//   Deserialize the engine from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[2];
  float* cpu_output_buffer = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);
  
  const int num_threads = video_numbers;
  std::vector<std::thread> threads(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(capture_show, std::ref(*capList[i]), video_numbers);
  }

  std::cout<<"come here"<<std::endl;
  while(1)
    {
        std::unique_lock<std::mutex> lock(mutex_1);
        if (imageList.size()>0)
        {
            stage_2.wait(lock, [&]
                                    { return imageList.size() == video_numbers; });
            
            std::cout<<"ready to infer"<<std::endl;
            // Preprocess
            auto start1 = std::chrono::system_clock::now();
            cuda_batch_preprocess(imageList, gpu_buffers[0], kInputW, kInputH, stream);
            auto end1 = std::chrono::system_clock::now();
            std::cout << "Preprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;

            // cv::Mat pr_img = preprocess_img(Wrap_images.front().img, INPUT_W, INPUT_H); // letterbox BGR to RGB & resize
            // Run inference
            auto start2 = std::chrono::system_clock::now();
            infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
            auto end2 = std::chrono::system_clock::now();
            std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;

            // NMS
            auto start3 = std::chrono::system_clock::now();
            std::vector<std::vector<Detection>> res_batch;
            batch_nms(res_batch, cpu_output_buffer, imageList.size(), kOutputSize, kConfThresh, kNmsThresh);
            auto end3 = std::chrono::system_clock::now();
            std::cout << "NMS time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << "ms" << std::endl;

            // Draw bounding boxes
            auto start4 = std::chrono::system_clock::now();
            draw_bbox(imageList, res_batch);
            auto end4 = std::chrono::system_clock::now();
            std::cout << "Draw bounding boxes time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4).count() << "ms" << std::endl;

            // Show images
            for (size_t i = 0; i < video_numbers; i++)
            {
                cv::imshow(threadSquence[i], imageList[i]);
            }
            threadSquence.clear();
            imageList.clear();
            stage_1.notify_one();
        }
         if (cv::waitKey(10) == 27)
        {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }
  

  // for (auto& t : threads) {
  //   t.join();
  // }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();
  return 0;
}

