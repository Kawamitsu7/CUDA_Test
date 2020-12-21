// from https://yuki67.github.io/post/cuda_time/

#ifndef CUDATIMER_H
#define CUDATIMER_H

#include <iostream>
#include <stdlib.h>
#include <string>

#include "cuda_runtime.h"

using namespace std;

// cudaのエラー検出用マクロ
#define EXIT_IF_FAIL(call)                                                     \
  do {                                                                         \
    (call);                                                                    \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cout << "error in file " << __FILE__ << " line at " << __LINE__     \
                << ": " << cudaGetErrorString(err) << std::endl;               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

class CudaTimer {
private:
  cudaEvent_t start, end;

public:
  CudaTimer() {
    EXIT_IF_FAIL(cudaEventCreate(&start));
    EXIT_IF_FAIL(cudaEventCreate(&end));
  }
  ~CudaTimer() {
    EXIT_IF_FAIL(cudaEventDestroy(start));
    EXIT_IF_FAIL(cudaEventDestroy(end));
  }
  // 計測開始
  void begin() {
    EXIT_IF_FAIL(cudaEventRecord(start));
  }
  // 計測終了
  void stop() {
    EXIT_IF_FAIL(cudaEventRecord(end));
  }
  // 測定結果を出力
  float report(string s, bool flg = true) {
    // イベントendが終わるまで待つ
    EXIT_IF_FAIL(cudaEventSynchronize(end));
    float elapsed;
    EXIT_IF_FAIL(cudaEventElapsedTime(&elapsed, start, end));
    if(flg) printf("%s: %f ms\n",s.c_str(), elapsed);
    return elapsed;
  }
  float stop_and_report(string s, bool flg = true) {
    stop();
    return report(s, flg);
  }
};

#endif /* CUDATIMER_H */