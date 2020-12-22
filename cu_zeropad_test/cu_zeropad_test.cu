#pragma region header

#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaTimer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>

#include <windows.h>

#include <assert.h>

#define all(a) (a).begin(),(a).end()
#define ll long long
#define eb emplace_back

#pragma endregion header

using namespace std;

// グローバル変数
// コピー用変数の準備
float *org, *pad_1d, *pad_2d;	// ホストデータ
cudaPitchedPtr org_d;	// デバイスデータ
size_t pitch, s[3], padded_sz, pad_pitch, sum_Bytes;
ll nBytes, ele; 
// s[3] : データ数 [0]width [1]height [2]depth
// cut : 一行のデータ数 / pitch : 横方向のピッチ(bytes) /nBytes:総データ量(bytes)

cudaExtent ext;
cudaMemcpy3DParms p = {0};

// 結果格納変数
vector<double> res1d, res2d;
// その他制御変数
bool flg = false; // printf起動制御
int itr_times; // 一つの条件に対する実験の繰り返し回数
ofstream ofs;

#define CUDA_SAFE_CALL(func) \
do { \
	 cudaError_t err = (func); \
	 if (err != cudaSuccess) { \
		 fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
		 exit(err); \
	 } \
} while(0)

void setup(){ 
	// CPU側の領域確保
	
	org = (float *)malloc(s[0]*s[1]*s[2]*sizeof(float));
	pad_1d = (float *)malloc(padded_sz*s[1]*s[2]*sizeof(float));
	pad_2d = (float *)malloc(padded_sz*(s[1]+1)*s[2]*sizeof(float));
	/*
	org = new float[s[0]*s[1]*s[2]];
	pad_1d = new float[padded_sz*s[1]*s[2]];
	pad_2d = new float[padded_sz*s[1]*s[2]];
	*/

	for(size_t i=0;i<s[0];i++){
		for(size_t j=0;j<s[1];j++){
			for(size_t k=0;k<s[2];k++){
				/*org[i*s[2]*s[1] + j*s[2] + k]
					= 1.0f*(i+1)+ 0.01f*(j+1) + 0.0001f*(k+1);
				*/
				org[i + j*s[0] + k*s[1]*s[0]]
					= 1*(i+1)+ 100*(j+1) + 10*(k+1);
			}
		}
	}

	for(size_t i=0;i<padded_sz;i++){
		for(size_t j=0;j<s[1];j++){
			for(size_t k=0;k<s[2];k++){
				pad_1d[i + j*padded_sz + k*s[1]*padded_sz] = 0;
				pad_2d[i + j*padded_sz + k*s[1]*padded_sz] = 0;
			}
		}
	}

	// GPUの元データ領域確保
	ext = make_cudaExtent(s[0]*sizeof(float),s[1],s[2]);
	p.extent.width = s[0]*sizeof(float);
	p.extent.height = s[1];
	p.extent.depth = s[2];

	cudaMalloc3D(&org_d, ext);

	cout << org_d.pitch << "\n";

	p.srcPtr.ptr = org;
	p.srcPtr.pitch = s[0]*sizeof(float);
	p.srcPtr.xsize = s[0];
	p.srcPtr.ysize = s[1];
	p.dstPtr.ptr = org_d.ptr;
	p.dstPtr.pitch = s[0]*sizeof(float);
	p.dstPtr.xsize = s[0];
	p.dstPtr.ysize = s[1];
	p.kind = cudaMemcpyHostToDevice;

	CUDA_SAFE_CALL( cudaMemcpy3D(&p) );

	// データコピーデバッグ
	float * debug;
	debug = new float[s[0]*s[1]*s[2]];

	cudaMemcpy(debug, org_d.ptr, sizeof(float)*s[0]*s[1]*s[2], cudaMemcpyDeviceToHost);

	if(s[0] == 3 && s[2] == 9 && itr_times == 1){
		cout << "org_img\n";
		for(size_t k=0; k<s[2]; k++){
			for(size_t i=0; i<s[0]; i++){
				if(org[i + k*s[1]*s[0]] == 0) cout << ".\t";
				else cout << org[i + k*s[1]*s[0]] << "\t";
			}
			cout << "\n";
		}

		cout << "debug_img\n";
		for(size_t k=0; k<s[2]; k++){
			for(size_t i=0; i<s[0]; i++){
				if(debug[i + k*s[1]*s[0]] == 0) cout << ".\t";
				else cout << debug[i + k*s[1]*s[0]] << "\t";
			}
			cout << "\n";
		}
	}

	for(size_t i=0;i<s[0];i++){
		for(size_t j=0;j<s[1];j++){
			for(size_t k=0;k<s[2];k++){
				if(org[i + j*s[0] + k*s[1]*s[0]] != debug[i + j*s[0] + k*s[1]*s[0]]){
					cout << "3DMemcpy Error" << org[i + j*s[0] + k*s[1]*s[0]] << "/" << debug[i + j*s[0] + k*s[1]*s[0]] << "\n";
					exit(1);
				}
			}
		}
	}
	cout << "3Dmemcpy ok\n";

	delete[] debug;

	return;
}

void doJob(const float * org_dev, float * pad_1d_h, float * pad_2d_h){
	// 必要な変数の用意
	size_t projCount = s[1];
	pad_pitch = padded_sz / 2;
	size_t pad_w_mem = padded_sz * sizeof(float);
	size_t buffMemSize = sizeof(float) * projCount * padded_sz;

// 実装すべきこと
// 計測変数用意
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	
	LARGE_INTEGER start, end;
	double result = 0.0;

// 1行ずつゼロパディング
	float* pad_d = NULL;
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&pad_d, buffMemSize));
	CUDA_SAFE_CALL(cudaMemset(pad_d, 0, buffMemSize));
	
	/*
	CUDA_SAFE_CALL(cudaMallocPitch((void **)&pad_d, &pad_w_mem, pad_w_mem, s[1]));
	CUDA_SAFE_CALL(cudaMemset2D(pad_d, pad_w_mem, 0, pad_w_mem, s[1]));
	*/
	QueryPerformanceCounter(&start);

	for(int p_idx=0; p_idx < projCount; p_idx++){
		const float * src_loc = org_dev + p_idx * s[0];
		float * dst_loc = pad_d + p_idx * padded_sz;

		CUDA_SAFE_CALL(cudaMemcpy(dst_loc, src_loc, sizeof(float)*s[0], cudaMemcpyDeviceToDevice));
	}

	QueryPerformanceCounter(&end);

	result = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	res1d.emplace_back(result);
	// データのCPU移動 *2Dのデータ移動じゃない?*
	
	/* CUDA_SAFE_CALL(cudaMemcpy2D(pad_1d_h, pad_w_mem,
		pad_d, pad_w_mem, pad_w_mem, s[1],
		cudaMemcpyDeviceToHost));
	*/
	CUDA_SAFE_CALL(cudaMemcpy(pad_1d_h, pad_d, buffMemSize, cudaMemcpyDeviceToHost));

	/*
	p.extent.width = padded_sz;
	p.extent.height = s[1];
	p.extent.depth = s[2];

	p.srcPtr.ptr = (void *)pad_d;
	p.srcPtr.pitch = padded_sz * sizeof(float);
	p.srcPtr.xsize = padded_sz;
	p.srcPtr.ysize = s[1];
	p.dstPtr.ptr = (void *)pad_1d;
	p.dstPtr.pitch = padded_sz * sizeof(float);
	p.dstPtr.xsize = padded_sz;
	p.dstPtr.ysize = s[1];
	p.kind = cudaMemcpyDeviceToHost;

	CUDA_SAFE_CALL(cudaMemcpy3D(&p));
	*/

	// メモリ開放
	cudaFree(pad_d);

// 2Dでゼロパディング
	pad_d = NULL;

	/*
	CUDA_SAFE_CALL(cudaMallocPitch((void **)&pad_d, &pad_w_mem, pad_w_mem, s[1]));
	CUDA_SAFE_CALL(cudaMemset2D(pad_d, pad_w_mem, 0.0f, pad_w_mem, s[1]));
	*/

	CUDA_SAFE_CALL(cudaMalloc((void **)&pad_d, buffMemSize));
	CUDA_SAFE_CALL(cudaMemset(pad_d, 0, buffMemSize));

	QueryPerformanceCounter(&start);

	// データコピー
	CUDA_SAFE_CALL(cudaMemcpy2D(pad_d, pad_w_mem,
		org_dev, s[0]*sizeof(float),
		s[0]*sizeof(float), s[1],
		cudaMemcpyDeviceToDevice));

	QueryPerformanceCounter(&end);

	result = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	res2d.emplace_back(result);

	// データのCPU移動 *2Dのデータ移動じゃない?*
	CUDA_SAFE_CALL(cudaMemcpy2D(pad_2d_h, pad_w_mem,
		pad_d, pad_w_mem,
		pad_w_mem, s[1],
		cudaMemcpyDeviceToHost));

	cudaFree(pad_d);

	return;
}

void check(){
	// ゼロパディングになってるのか確認
	if(s[0] == 3 && s[2] == 9 && itr_times == 1){
		
		cout << "1d_pad\n";
		for(size_t k=0; k<s[2]; k++){
			for(size_t i=0; i<padded_sz; i++){
				if(pad_1d[i + k*s[1]*padded_sz] == 0) cout << ".\t";
				else cout << pad_1d[i + k*s[1]*padded_sz] << "\t";
			}
			cout << "\n";
		}

		cout << "2d_pad\n";
		for(size_t k=0; k<s[2]; k++){
			for(size_t i=0; i<padded_sz; i++){
				if(pad_2d[i + k*s[1]*padded_sz] == 0) cout << ".\t";
				else cout <<  pad_2d[i + k*s[1]*padded_sz] << "\t";
			}
			cout << "\n";
		}
	}

	// 正誤確認
	size_t err = 0, success = 0;
	for(size_t i=0;i<padded_sz;i++){
		for(size_t j=0;j<s[1];j++){
			for(size_t k=0;k<s[2];k++){
//				if(pad_1d[i * s[1] * s[2] + j * s[2] + k] != pad_2d[i * s[1] * s[2] + j * s[2] + k]){
//					cout << "not match by (" << i << "," << j << "," << k << ")\n";
//					cout <<  pad_1d[i * s[1] * s[2] + j * s[2] + k] << "/" << pad_2d[i * s[1] * s[2] + j * s[2] + k] << "\n";
//				}
				/*
				assert(a_h[i * s[1] * s[2] + j * s[2] + k] == b_h[i * s[1] * s[2] + j * s[2] + k]);
				*/
				if(pad_1d[i + j*padded_sz + k*s[1]*padded_sz] != pad_2d[i + j*padded_sz + k*s[1]*padded_sz]) err++;
				else success++;
			}
		}
	}

	if(err != 0){
		cout << "error " << err << "/" << success << "\n";
		exit(1);
	}
	return;
}

void memFree(){
	
	free(org); 
	free(pad_1d); cout << "pass\n";
	free(pad_2d); 
	
	//delete[] org; delete[] pad_1d; delete[] pad_2d;
	// メモリ開放
	cudaFree(org_d.ptr);
	return;
}

#pragma region csvout

void csv_out() {ofs << endl;}

template <typename Head, typename... Tail>
void csv_out(Head H, Tail... T){
	ofs << H;
	csv_out(T...);
}
#define csvo(...) csv_out(__VA_ARGS__)

#pragma endregion csvout

void put_csv(){
	// エラーチェック
	assert(res1d.size() == res2d.size());
	// 平均データ
	double ave_1d = accumulate(res1d.begin(),res1d.end(),(double)0.0f) / itr_times;
	double ave_2d = accumulate(res2d.begin(),res2d.end(),(double)0.0f) / itr_times;

	// データ書き出し
	csvo(s[0], " x ", s[2], ",", ave_1d, ",", ave_2d, "\n");

	printf("put_csv\n");
	return;
}

size_t nextPowerOfTwo(size_t x){
	size_t res = 1;
	while(res < x){
		res *= 2;
	}
	return res;
}

void paraSet(){
	nBytes = s[0] * sizeof(float);
	pitch = nBytes;
	sum_Bytes = nBytes * s[1] * s[2];

	padded_sz = nextPowerOfTwo(2 * s[0]);
	pad_pitch = padded_sz / 2;

	return;
}

int main(){
	// 実装すべきこと
	itr_times = 10;
	
	// 画像サイズ、枚数の決定 (枚数はとりま360枚固定?)
	int width = 128; int height = 128; int projnum = 360;
	
	s[0] = width; s[2] = height;
	s[1] = projnum;

	ofs.open("result.csv");

	while(s[0] <= 2048){
		s[2] = height;

		while(s[2] <= 2048){
			string data_name = to_string(s[0])
			+ "x" + to_string(s[2]) + "_zeropad";
		
			cout << "processing " << data_name << "\n";
			csvo("img_size,1d_ave,2d_ave"); // 項目の入力

			paraSet(); // 各種パラメータ設定

			res1d.clear(); res2d.clear();

			setup();
			for(int i=0; i < itr_times; ++i){
				float * src = (float*)org_d.ptr;
				float * dst_1d = pad_1d;
				float * dst_2d = pad_2d;
				for(int v=0; v < s[2]; ++v){
					doJob(src, dst_1d, dst_2d);
					src += (s[1] * s[0]);			// アドレス移動は正しい
					dst_1d += (s[1] * padded_sz);	// 正しい
					dst_2d += (s[1] * padded_sz);	// 正しい
					//cout << v << " ";
				}
				//cout << "\n";
				check();
			}
			memFree();
			put_csv();

			s[2] *= 2;
		}
		s[0] *= 2;
	}

	ofs.close();
		// 最初の2^n計算
		// setup呼び出し
		// measure呼び出し
		// memFree呼び出し
		// put_csv呼び出し

	// ASTRAでは、x*yピクセルの画像がt枚に対して、pythonの配列に格納
	// arr[x][t][y]でアクセスできる
	return 0;
}