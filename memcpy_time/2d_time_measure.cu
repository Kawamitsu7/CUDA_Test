// from https://yuki67.github.io/post/cuda_time/

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>

#include <assert.h>

#define all(a) (a).begin(),(a).end()
#define ll long long

// グローバル変数
// コピー用変数の準備
float *a_h, *b_h;	// ホストデータ
float *a_d, *b_d;	// デバイスデータ
size_t pitch, width, height, cut; ll nBytes, ele; 
// width, height:データ数/ cut : 一行のデータ数 / pitch : 横方向のピッチ(bytes) /nBytes:総データ量(bytes)

// vector<long long> n_list = {160LL, 1024, 3000, 160LL*180*360, 1024LL*1024*1200, /*3000LL*3500*360*/};
// vector<int> n_list = {1024 / sizeof(float), 512 * 1024 / sizeof(float), 1024 * 1024 / sizeof(float), 128 * 1024 * 1024 / sizeof(float)};

// 結果格納変数
vector<float> h2d, d2d, d2h;
// その他制御変数
bool flg = false; // printf起動制御
int itr_times = 100; // 一つの条件に対する実験の繰り返し回数
ofstream ofs;
random_device seed_gen;
uniform_real_distribution<float> dist(-1.0,1.0);

#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

void setup(){
	// mt19937 engine(seed_gen());

	a_h = (float *)malloc(height * width * sizeof(float)); // これで配列として確保できる
	b_h = (float *)malloc(height * width * sizeof(float));
	cudaMallocPitch((void **) &a_d, &pitch, width * sizeof(float), height);
	cudaMallocPitch((void **) &b_d, &pitch, width * sizeof(float), height);

	// fill(a_h, a_h + n, 1.3);
	/*
	for(long long i = 0; i < n; i++){
		a_h[i] = dist(engine);
	}
	*/
	for(ll i=0;i<height;i++){
		for(ll j=0;j<width;j++){
			a_h[i * width + j] = 1.0f + 0.01f * i + 0.0001f * j;
		}
	}

	return;
}

void measure(){
	// 計測
	CudaTimer timer;
	// 1. H2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	CUDA_SAFE_CALL( cudaMemcpy2D(a_d, pitch, a_h, nBytes, nBytes, height, cudaMemcpyHostToDevice) );
	h2d.emplace_back( timer.stop_and_report("H2D",flg) );

	// 2. D2D
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy2D(b_d, pitch, a_d, pitch, pitch, height, cudaMemcpyDeviceToDevice);
	d2d.emplace_back( timer.stop_and_report("D2D",flg) );

	// 3.D2H
	timer.begin();
	// ~~~計測対象の動作を記述~~~
	cudaMemcpy2D(b_h, nBytes, b_d, pitch, nBytes, height, cudaMemcpyDeviceToHost);
	d2h.emplace_back( timer.stop_and_report("D2H",flg) );

	// 正誤チェック
	/*
	for(long long i = 0; i < n; i++){
		assert(a_h[i] == b_h[i]);
	}
	
	if(ele==256){
		for(ll i=0; i<height; i++) for(ll j=0;j<width;j++) cout << a_h[i * width + j] << "/" << b_h[i * width + j] << "\n";
	}
	*/
	for(ll i=0; i<height; i++) for(ll j=0;j<width;j++) assert(a_h[i * width + j] == b_h[i * width + j]);

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

void csv_out() {ofs << endl;}

template <typename Head, typename... Tail>
void csv_out(Head H, Tail... T){
	ofs << H;
	csv_out(T...);
}
#define csvo(...) csv_out(__VA_ARGS__)

void put_csv(long long Bytes){
	// エラーチェック
	assert(h2d.size() == d2d.size() && d2d.size() == d2h.size());
	
	/*
	// データ入力
	for(long long ind = 0; ind < h2d.size(); ++ind){
		ofs << h2d.at(ind) << "," << "," << d2d.at(ind) << "," << "," << d2h.at(ind) << "," << endl;
	}
	*/
	
	// 平均データ入力
	float h2d_ave = accumulate(h2d.begin(),h2d.end(),0.0f) / h2d.size();
	float d2d_ave = accumulate(d2d.begin(),d2d.end(),0.0f) / d2d.size();
	float d2h_ave = accumulate(d2h.begin(),d2h.end(),0.0f) / d2h.size();
	//ofs << "," << "Ave." <<endl;
	//ofs << h2d_ave << "," << "," << d2d_ave << "," << "," << d2h_ave << "," << endl;
	
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
	//ofs << "," << "Med." <<endl;
	//ofs << h2d_med << "," << "," << d2d_med << "," << "," << d2h_med << "," << endl;

	// データ書き込み
	csvo(Bytes/1024,",,",h2d_ave,",",d2d_ave,",",d2h_ave,",,",h2d_med,",",d2d_med,",",d2h_med);
}

int main() {

	cut = 1;

	while(cut <= 256){
		cout << to_string(cut) << "cut_processing\n";
		string data_name = to_string(cut) + "cut_time_plot_data_" +  ".csv";
		ofs.open(data_name);
		// 項目の入力
		csvo("(KBytes)\\(msec.),","<Ave.>,","H2D,","D2D,","D2H,","<Med.>,","H2D,","D2D,","D2H");
		//ofs << "H2D" << "," << "," << "D2D" << "," << "," << "D2H" << "," << endl;
		// for(long long ele : n_list){
		ele = 256LL;
		long long add = 256LL;
		long long base = 10LL;
		long long div = 1024LL;
		bool flg = false;
		while(ele < 1024LL*1024*512){	//1GBまで?
			/*
			n = ele;
			nBytes = n * sizeof(float);
			*/
			width = (size_t)ele / cut;
			height = cut;
			nBytes = width * sizeof(float);
			pitch = nBytes;
			ll sum_Bytes = nBytes * height;
			if(sum_Bytes / (1024 * 1024) > 0){
				cout << "transport data size : " << sum_Bytes / (1024 * 1024) << "[M Bytes]" << endl;
				// data_name = to_string(nBytes / (1024 * 1024)) + "M_Bytes_measure.csv";
			}
			else if(sum_Bytes / 1024 > 0){
				cout << "transport data size : " << sum_Bytes / 1024 << "[K Bytes]" << endl;
				// data_name = to_string(nBytes / 1024) + "K_Bytes_measure.csv";
			}
			else{
				cout << "transport data size : " << sum_Bytes << "[Bytes]" << endl;
				// data_name = to_string(nBytes) + "Bytes_measure.csv";
			}
			h2d.clear();
			d2d.clear();
			d2h.clear();

			setup();
			for(int i = 0; i < itr_times; ++i){
				measure();
			}
			memFree();
	
			cout << "finished" << "\n";
			put_csv(nBytes);
			if(ele * sizeof(float) / div >= base){
				base *= 10;
				add *= 10;
			}
			ele += add;
			if(!flg && ele * sizeof(float) / 1024 > 1000){
				flg = true;
				div *= 1024;
				base = 10;
				add = 256LL * 1024;
				ele = 256LL * 1024;
			}
		}
		ofs.close();

		cut *= 2;
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
