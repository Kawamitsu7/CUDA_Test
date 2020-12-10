#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
typedef long long ll;

//マクロ
//forループ
//引数は、(ループ内変数,動く範囲)か(ループ内変数,始めの数,終わりの数)、のどちらか
//Dがついてないものはループ変数は1ずつインクリメントされ、Dがついてるものはループ変数は1ずつデクリメントされる
//FORAは範囲for文(使いにくかったら消す)
#define REP(i,n) for(ll i=0;i<ll(n);i++)
#define REPD(i,n) for(ll i=n-1;i>=0;i--)
#define FOR(i,a,b) for(ll i=a;i<=ll(b);i++)
#define FORD(i,a,b) for(ll i=a;i>=ll(b);i--)
#define FORA(i,I) for(const auto& i:I)
//xにはvectorなどのコンテナ
#define ALL(x) x.begin(),x.end() 
#define SIZE(x) ll(x.size()) 
//定数
#define INF 1000000000000 //10^12:∞
#define MOD 1000000007 //10^9+7:合同式の法
#define MAXR 100000 //10^5:配列の最大のrange
//略記
#define PB push_back //挿入
#define MP make_pair //pairのコンストラクタ
#define F first //pairの一つ目の要素
#define S second //pairの二つ目の要素

int calcNextPowerOfTwo(int _iValue){
	int iOutput = 1;
	while (iOutput < _iValue)
		iOutput *= 2;
	return iOutput;
}


void main() {
	int w, h;
	const int lp_lmt = 100;
	const int img_amount = 360;

	cout << "input width of img\n";
	cin >> w;
	cout << "input height of img\n";
	cin >> h;

	int paddedcount = calcNextPowerOfTwo(2 * w);

	vector<int> time;

	cv::Mat img(h, w, CV_64F, cv::Scalar(65535.0));
	cv::Mat pad(h, paddedcount - w, CV_64F, cv::Scalar(45000.0));

	cv::Mat dst;

	REP(li, lp_lmt) {
		cv::hconcat(img, pad, dst);

		if (li == 0) {
			/*
			cv::Range slice[ndims];
			slice[0] = cv::Range::all();
			slice[1] = cv::Range::all();
			slice[2] = cv::Range(0, 0);
			cv::imwrite("test.png", dst(slice));
			*/
			dst.convertTo(dst, CV_16UC1);
			cv::imwrite("test.tif", dst);
		}
	}
}