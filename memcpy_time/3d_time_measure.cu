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
cudaPitchedPtr a_d, b_d;	// デバイスデータ
size_t pitch, s[3], cut[2], sum_Bytes; ll nBytes, ele; 
// s[3] : データ数 [0]width [1]height [2]depth
// cut : 一行のデータ数 / pitch : 横方向のピッチ(bytes) /nBytes:総データ量(bytes)

cudaExtent ext;
cudaMemcpy3DParms p = {0};

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

	a_h = (float *)malloc(s[0] * s[1] * s[2] * sizeof(float)); // これで配列として確保できる
	b_h = (float *)malloc(s[0] * s[1] * s[2] * sizeof(float));
	
	ext = make_cudaExtent(s[0]*sizeof(float),s[1],s[2]);

	p.extent.width = s[0] * sizeof(float);
	p.extent.height = s[1];
	p.extent.depth = s[2];

	cudaMalloc3D(&a_d, ext);
	cudaMalloc3D(&b_d, ext);
	/*
	cudaMallocPitch((void **) &a_d, &pitch, width * sizeof(float), height);
	cudaMallocPitch((void **) &b_d, &pitch, width * sizeof(float), height);
	*/

	// fill(a_h, a_h + n, 1.3);
	/*
	for(long long i = 0; i < n; i++){
		a_h[i] = dist(engine);
	}
	*/
	for(size_t i=0;i<s[0];i++){
		for(size_t j=0;j<s[1];j++){
			for(size_t k=0;k<s[2];k++){
				a_h[i * s[1] * s[2] + j * s[2] + k] = 1.0f * i + 0.01f * j + 0.0001f * k;
			}
		}
	}

	return;
}

void measure(){
	// 計測
	CudaTimer timer;
	// 1. H2D
	p.srcPtr.ptr = a_h;
	p.srcPtr.pitch = s[0] * sizeof(float);
	p.srcPtr.xsize = s[0];
	p.srcPtr.ysize = s[1];
	p.dstPtr.ptr = a_d.ptr;
	p.dstPtr.pitch = a_d.pitch;
	p.dstPtr.xsize = s[0];
	p.dstPtr.ysize = s[1];
	p.kind = cudaMemcpyHostToDevice;

	timer.begin();
	// ~~~計測対象の動作を記述~~~
	CUDA_SAFE_CALL( cudaMemcpy3D(&p) );
	//CUDA_SAFE_CALL( cudaMemcpy2D(a_d, pitch, a_h, nBytes, nBytes, height, cudaMemcpyHostToDevice) );
	h2d.emplace_back( timer.stop_and_report("H2D",flg) );

	// 2. D2D
	p.srcPtr.ptr = a_d.ptr;
	p.srcPtr.pitch = a_d.pitch;
	p.srcPtr.xsize = s[0];
	p.srcPtr.ysize = s[1];
	p.dstPtr.ptr = b_d.ptr;
	p.dstPtr.pitch = b_d.pitch;
	p.dstPtr.xsize = s[0];
	p.dstPtr.ysize = s[1];
	p.kind = cudaMemcpyDeviceToDevice;

	timer.begin();
	// ~~~計測対象の動作を記述~~~
	CUDA_SAFE_CALL( cudaMemcpy3D(&p) );
	//cudaMemcpy2D(b_d, pitch, a_d, pitch, pitch, height, cudaMemcpyDeviceToDevice);
	d2d.emplace_back( timer.stop_and_report("D2D",flg) );

	// 3.D2H
	p.srcPtr.ptr = b_d.ptr;
	p.srcPtr.pitch = b_d.pitch;
	p.srcPtr.xsize = s[0];
	p.srcPtr.ysize = s[1];
	p.dstPtr.ptr = b_h;
	p.dstPtr.pitch = s[0] * sizeof(float);
	p.dstPtr.xsize = s[0];
	p.dstPtr.ysize = s[1];
	p.kind = cudaMemcpyDeviceToHost;

	timer.begin();
	// ~~~計測対象の動作を記述~~~
	CUDA_SAFE_CALL( cudaMemcpy3D(&p) );
	//cudaMemcpy2D(b_h, nBytes, b_d, pitch, nBytes, height, cudaMemcpyDeviceToHost);
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

	size_t error = 0, success = 0;
	for(size_t i=0;i<s[0];i++){
		for(size_t j=0;j<s[1];j++){
			for(size_t k=0;k<s[2];k++){
				if(a_h[i * s[1] * s[2] + j * s[2] + k] != b_h[i * s[1] * s[2] + j * s[2] + k]){
					cout << "not match by (" << i << "," << j << "," << k << ")\n";
					cout <<  a_h[i * s[1] * s[2] + j * s[2] + k] << "/" << b_h[i * s[1] * s[2] + j * s[2] + k] << "\n";
				}
				/*
				assert(a_h[i * s[1] * s[2] + j * s[2] + k] == b_h[i * s[1] * s[2] + j * s[2] + k]);
				*/
				if(a_h[i * s[1] * s[2] + j * s[2] + k] != b_h[i * s[1] * s[2] + j * s[2] + k]) error++;
				else success++;
			}
		}
	}
	//for(ll i=0; i<height; i++) for(ll j=0;j<width;j++) assert(a_h[i * width + j] == b_h[i * width + j]);

	if(error != 0){
		cout << error << "/" << success << "\n";
		exit(1);
	}

	return;
}

void memFree(){
	// データ解放
	free(a_h);
	free(b_h);
	cudaFree(a_d.ptr);
	cudaFree(b_d.ptr);

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
	csvo(sum_Bytes/1024,",,",h2d_ave,",",d2d_ave,",",d2h_ave,",,",h2d_med,",",d2d_med,",",d2h_med);
}

int main() {

	cut[0] = 1; cut[1] = 1;

	while(cut[0] <= 256){
		cut[1] = 1;

		while(cut[0]*cut[1] <= 256){
			string data_name = to_string(cut[0]) + "by" + to_string(cut[1]) + "_cut_time_plot_data_" +  ".csv";
			cout << data_name << " <-- processing\n";
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
				s[2] = cut[1];
				s[1] = cut[0];
				s[0] = (size_t)ele / (s[1] * s[2]);
				nBytes = s[0] * sizeof(float);
				pitch = nBytes;
				sum_Bytes = nBytes * s[1] * s[2];
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

			cut[1] *= 2;
		}

		cut[0] *= 2;
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
