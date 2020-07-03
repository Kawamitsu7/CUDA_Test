// from https://yuki67.github.io/post/cuda_time/

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"

// 何もしない
__global__ void empty() {}

int main() {
	CudaTimer timer;
	timer.begin();
	// 何もしない関数を呼ぶとどれくらい時間がかかるのだろう?
	empty<<<1024, 1024>>>();
	timer.stop_and_report();

	return 0;
}