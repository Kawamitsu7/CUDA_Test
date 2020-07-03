// from https://yuki67.github.io/post/cuda_time/

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"

#include <assert.h>

int main() {
	// コピー用変数の準備
	float *a_h, *b_h;	// ホストデータ
	float *a_d, *b_d;	// デバイスデータ
	int n = 14, nBytes; // n:データ数 / nBytes:総データ量(bytes)

	nBytes = n * sizeof(float);
	a_h = (float *)malloc(nBytes); // これで配列として確保できる
	b_h = (float *)malloc(nBytes);
	cudaMalloc((void **) &a_d, nBytes);
	cudaMalloc((void **) &b_d, nBytes);

	for(int i = 0; i < n; i++){
		a_h[i] = 100.0f + i;
	}

	// 計測
	CudaTimer timer;
	// 1. H2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
	timer.stop_and_report();

	// 2. D2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
	timer.stop_and_report();

	// 3.D2H
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
	timer.stop_and_report();

	// 正誤チェック
	for(int i = 0; i < n; i++){
		assert(a_h[i] == b_h[i]);
	}
	// データ解放
	free(a_h);
	free(b_h);
	cudaFree(a_d);
	cudaFree(b_d);

	return 0;
}
