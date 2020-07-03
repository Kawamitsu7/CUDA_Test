// from https://yuki67.github.io/post/cuda_time/

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"

// ‰½‚à‚µ‚È‚¢
__global__ void empty() {}

int main() {
	CudaTimer timer;
	timer.begin();
	// ‰½‚à‚µ‚È‚¢ŠÖ”‚ğŒÄ‚Ô‚Æ‚Ç‚ê‚­‚ç‚¢ŠÔ‚ª‚©‚©‚é‚Ì‚¾‚ë‚¤?
	empty<<<1024, 1024>>>();
	timer.stop_and_report();

	return 0;
}