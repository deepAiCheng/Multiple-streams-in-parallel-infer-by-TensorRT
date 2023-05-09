#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
const int videoNumbers = 8;
using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

// 互斥锁
std::mutex mutex_1;
// 每个阶段的条件变量
std::condition_variable stage_1;
std::condition_variable stage_2;

std::vector<cv::Mat> imageList;
std::vector<std::string> rtspSquence;//store the sequence of rtsp when showing images by opencv

class WrapData
{
public:
  cv::Mat img;
  std::string channel;
};

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


int capture_show(std::string& rtsp_address){
    // WrapData Wrap_img;
    cv::Mat frame;
    cv::VideoCapture cap;
    cap.open(rtsp_address);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        auto start = std::chrono::system_clock::now();
        cap.read(frame);
        auto end = std::chrono::system_clock::now();
        std::cout << "cap.read time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        // check if we succeeded
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        std::unique_lock<std::mutex> lock(mutex_1);
        stage_1.wait(lock, [&]
                                { return imageList.size() < videoNumbers; });
        imageList.push_back(frame);
        rtspSquence.push_back(rtsp_address);
        stage_2.notify_one();
    }
    cap.release();
    return 1;
}




int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  if (argc != 2)
  {
      std::cerr << "usage: " << argv[0] << " <engine_file> " << std::endl;
      return -1;
  }

  std::string engine_name = argv[1];

  std::string rtsp1="rtsp://192.168.70.215/live/test1";
  std::string rtsp2="rtsp://192.168.70.215/live/test2";
  std::string rtsp3="rtsp://192.168.70.215/live/test3";
  std::string rtsp4="rtsp://192.168.70.215/live/test4";
  std::string rtsp5="rtsp://192.168.70.215/live/test5";
  std::string rtsp6="rtsp://192.168.70.215/live/test6";
  std::string rtsp7="rtsp://192.168.70.215/live/test7";
  std::string rtsp8="rtsp://192.168.70.215/live/test8";

  std::vector<std::string> rtspList = {rtsp1,rtsp2,rtsp3,rtsp4,rtsp5,rtsp6,rtsp7,rtsp8};//store the whole rtsp address
  // Deserialize the engine from file
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

  std::vector<std::thread> threads(videoNumbers);
  for (int i = 0; i < videoNumbers; ++i) {
    threads[i] = std::thread(capture_show, std::ref(rtspList[i]));
  }

  while(1)
    {
        std::unique_lock<std::mutex> lock(mutex_1);
        if (imageList.size()>0)
        {
            stage_2.wait(lock, [&]
                                    { return imageList.size() == videoNumbers; });
            
            // Preprocess
            auto start1 = std::chrono::system_clock::now();
            cuda_batch_preprocess(imageList, gpu_buffers[0], kInputW, kInputH, stream);
            auto end1 = std::chrono::system_clock::now();
            std::cout << "Preprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;

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
            for (size_t i = 0; i < videoNumbers; i++)
            {
                cv::imshow(rtspSquence[i], imageList[i]);
            }
            rtspSquence.clear();
            imageList.clear();
            stage_1.notify_one();
        }
         if (cv::waitKey(10) == 27)
        {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }

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
