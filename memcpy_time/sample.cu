// from https://yuki67.github.io/post/cuda_time/

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"

// �������Ȃ�
__global__ void empty() {}

int main() {
	CudaTimer timer;
	timer.begin();
	// �������Ȃ��֐����ĂԂƂǂꂭ�炢���Ԃ�������̂��낤?
	empty<<<1024, 1024>>>();
	timer.stop_and_report();

	return 0;
}