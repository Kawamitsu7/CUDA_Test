// from https://yuki67.github.io/post/cuda_time/

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>

#include <assert.h>

#define all(a) (a).begin(),(a).end()

// グローバル変数
// コピー用変数の準備
float *a_h, *b_h;	// ホストデータ
float *a_d, *b_d;	// デバイスデータ
int n, nBytes; // n:データ数 / nBytes:総データ量(bytes)
vector<int> n_list = {1024 / sizeof(float), 512 * 1024 / sizeof(float), 1024 * 1024 / sizeof(float), 128 * 1024 * 1024 / sizeof(float)};
// 結果格納変数
vector<float> h2d, d2d, d2h;
// その他制御変数
bool flg = false; // printf起動制御
int itr_times = 100; // 一つの条件に対する実験の繰り返し回数
ofstream ofs;

void setup(){
	a_h = (float *)malloc(nBytes); // これで配列として確保できる
	b_h = (float *)malloc(nBytes);
	cudaMalloc((void **) &a_d, nBytes);
	cudaMalloc((void **) &b_d, nBytes);

	for(int i = 0; i < n; i++){
		a_h[i] = 100.0f + i;
	}

	return;
}

void measure(){
	// 計測
	CudaTimer timer;
	// 1. H2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
	h2d.emplace_back( timer.stop_and_report("H2D",flg) );

	// 2. D2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
	d2d.emplace_back( timer.stop_and_report("D2D",flg) );

	// 3.D2H
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
	d2h.emplace_back( timer.stop_and_report("D2H",flg) );

	// 正誤チェック
	for(int i = 0; i < n; i++){
		assert(a_h[i] == b_h[i]);
	}

	return;
}

void memFree(){
	// データ解放
	free(a_h);
	free(b_h);
	cudaFree(a_d);
	cudaFree(b_d);

	return;
}

void put_csv(){
	// エラーチェック
	assert(h2d.size() == d2d.size() && d2d.size() == d2h.size());
	// 項目の入力
	ofs << "H2D" << "," << "," << "D2D" << "," << "," << "D2H" << "," << endl;
	// データ入力
	for(int ind = 0; ind < h2d.size(); ++ind){
		ofs << h2d.at(ind) << "," << "," << d2d.at(ind) << "," << "," << d2h.at(ind) << "," << endl;
	}
	// 平均データ入力
	float h2d_ave = accumulate(h2d.begin(),h2d.end(),0.0f) / h2d.size();
	float d2d_ave = accumulate(d2d.begin(),d2d.end(),0.0f) / d2d.size();
	float d2h_ave = accumulate(d2h.begin(),d2h.end(),0.0f) / d2h.size();
	ofs << "," << "Ave." <<endl;
	ofs << h2d_ave << "," << "," << d2d_ave << "," << "," << d2h_ave << "," << endl;
	// 中央値データ入力
	sort(all(h2d));	sort(all(d2d));	sort(all(d2h));
	size_t med_ind = h2d.size() / 2;
	float h2d_med = (h2d.size() % 2 == 0
    ? static_cast<float>(h2d[med_ind] + h2d[med_ind - 1]) / 2
	: h2d[med_ind]);
	float d2d_med = (d2d.size() % 2 == 0
    ? static_cast<float>(d2d[med_ind] + d2d[med_ind - 1]) / 2
	: d2d[med_ind]);
	float d2h_med = (d2h.size() % 2 == 0
    ? static_cast<float>(d2h[med_ind] + d2h[med_ind - 1]) / 2
	: d2h[med_ind]);
	ofs << "," << "Med." <<endl;
	ofs << h2d_med << "," << "," << d2d_med << "," << "," << d2h_med << "," << endl;
}

int main() {
	for(auto ele : n_list){
		n = ele;
		nBytes = n * sizeof(float);
		cout << "transport data size : " << nBytes / 1024 << "[K Bytes]" << endl;
		string data_name = to_string(nBytes / 1024) + "Bytes_measure.csv";
		ofs.open(data_name);

		h2d.clear();
		d2d.clear();
		d2h.clear();

		for(int i = 0; i < itr_times; ++i){
			setup();
			measure();
			memFree();
		}
	
		cout << "finished" << "\n";

		put_csv();
		ofs.close();
	}

	return 0;
	/*
	nBytes = n * sizeof(float);
	a_h = (float *)malloc(nBytes); // これで配列として確保できる
	b_h = (float *)malloc(nBytes);
	cudaMalloc((void **) &a_d, nBytes);
	cudaMalloc((void **) &b_d, nBytes);

	for(int i = 0; i < n; i++){
		a_h[i] = 100.0f + i;
	}
	*/

	/*
	// 計測
	CudaTimer timer;
	// 1. H2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
	timer.stop_and_report("H2D");

	// 2. D2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
	timer.stop_and_report("D2D");

	// 3.D2H
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
	timer.stop_and_report("D2H");

	// 正誤チェック
	for(int i = 0; i < n; i++){
		assert(a_h[i] == b_h[i]);
	}
	*/

	/*
	// データ解放
	free(a_h);
	free(b_h);
	cudaFree(a_d);
	cudaFree(b_d);
	*/
}
