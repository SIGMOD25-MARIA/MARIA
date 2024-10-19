#pragma once
#include "StructType.h"
#include "basis.h"
#include <cmath>
#include <assert.h>
#include <string>
#include <vector>
#include <queue>
#include <cfloat>

#include <fstream>
#include <assert.h>
#include <random>
#include <iostream>
#include <fstream>
#include <map>
#include <ctime>
#include <sstream>
#include <numeric>
#include<algorithm>

#define CANDIDATES 100
// #define E 2.718281746
// #define PI 3.1415926
//#define MAXSIZE 4096

#define MAXSIZE INT32_MAX

//#define min(a,b)           (((a) < (b)) ? (a) : (b))

typedef struct Tuple
{
	int id;
	float inp;
	bool operator < (const Tuple& rhs) {
		return inp < rhs.inp;
	}
}Tuple;

inline bool comp(const Tuple& a, const Tuple& b)
{
	return a.inp > b.inp;
}

struct Dist_id
{
	int id = -1;
	float dist = 0.0f;
	//Dist_id() = default;
	Dist_id(int id_, float dist_) :id(id_), dist(dist_) {}
	bool operator < (const Dist_id& rhs) {
		return dist < rhs.dist;
	}
};

class Preprocess
{
	//int for_varied_n = 0;
	public:
	Data data;
	Data queries;
	float* SquareLen;
	Ben benchmark;
	float MaxLen;
	std::string data_file;
	std::string ben_file;
	public:
	Preprocess(const std::string& path, const std::string& ben_file_, int varied_n = 0) {
		//for_varied_n = varied_n;
		lsh::timer timer;
		std::cout << "LOADING DATA..." << std::endl;
		timer.restart();
		if (path.find("deep1B") != std::string::npos || path.find("yandex") != std::string::npos) {
			load_fbin(path + "_query.fbin", queries);
			load_fbin(path + ".fbin", data, varied_n);

		}
		else load_data(path, varied_n);
		std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
		cal_SquareLen();

		data_file = path;
		ben_file = ben_file_;
		if (varied_n > 0) ben_file += std::to_string(varied_n);
		ben_create();
	}

	void load_data(const std::string& path, int varied_n = 0) {
		std::string file = path + ".data_new";
		std::ifstream in(file.c_str(), std::ios::binary);
		if (!in) {
			printf("Fail to open the file!:%s\n", file.c_str());
			exit(-1);
		}



		unsigned int header[3] = {};
		assert(sizeof header == 3 * 4);
		in.read((char*)header, sizeof(header));
		assert(header[1] != 0);
		data.N = header[1] - 200;
		data.dim = header[2];

		while (varied_n > 0) {
			data.N /= 10;
			varied_n--;
		}

		queries.N = 200;
		queries.dim = data.dim;

		std::cout << "Load from new file: " << file << "\n";
		std::cout << "Nq =  " << queries.N << "\n";
		std::cout << "N  =  " << data.N << "\n";
		std::cout << "dim=  " << data.dim << "\n\n";

		queries.val = new float* [queries.N];
		data.val = new float* [data.N];

		//data.offset=data.dim+1;
		data.base = new float[(size_t)data.N * data.dim];
		queries.base = new float[(size_t)queries.N * queries.dim];

		for (size_t i = 0; i < queries.N; ++i) {
			// queries.val[i] = new float[queries.dim + 1];
			// in.read((char*)queries.val[i], sizeof(float) * header[2]);
			// queries.val[i][queries.dim - 1] = 0.0f;

			queries.val[i] = queries.base + i * queries.dim;
			in.read((char*)queries.val[i], sizeof(float) * header[2]);
		}

		for (size_t i = 0; i < data.N; ++i) {
			// data.val[i] = new float[data.dim + 1];
			// in.read((char*)data.val[i], sizeof(float) * header[2]);
			// data.val[i][data.dim - 1] = 0.0f;

			data.val[i] = data.base + i * data.dim;
			in.read((char*)data.val[i], sizeof(float) * header[2]);
		}

		std::cout << "Finish loading! " << "\n";

		in.close();
	}

	void load_fbin(const std::string& path, Data& data, int varied_n = 0) {
		std::string file = path;
		std::ifstream in(file.c_str(), std::ios::binary);
		if (!in) {
			printf("Fail to open the file: %s\n", file.c_str());
			exit(-1);
		}

		unsigned int header[2] = {};
		assert(sizeof header == 2 * 4);
		in.read((char*)header, sizeof(header));
		assert(header[0] != 0);
		data.N = header[0];
		data.dim = header[1];

		while (varied_n > 0) {
			data.N /= 10;
			varied_n--;
		}
		size_t size = ((size_t)data.N) * data.dim;
		std::cout << "Load from fbin: " << file << "\n";
		std::cout << "N   =  " << data.N << "\n";
		std::cout << "dim =  " << data.dim << "\n";
		std::cout << "size=  " << size << "\n\n";

		data.base = new float[size];
		data.val = new float* [data.N];
		for (size_t i = 0; i < data.N; ++i) {
			data.val[i] = data.base + i * data.dim;
		}

		in.read((char*)data.base, sizeof(float) * size);
		// #pragma omp parallel for schedule(dynamic, 256)
		// 		for (int i = 0; i < data.N; ++i) {
		// 			in.read((char*)data.val[i], sizeof(float) * data.dim);
		// 		}

		std::cout << "Finish Reading File! " << "\n";
		in.close();
	}

	void cal_SquareLen() {
		SquareLen = new float[data.N];
		for (int i = 0; i < data.N; ++i) SquareLen[i] = cal_inner_product(data.val[i], data.val[i], data.dim);

		MaxLen = *std::max_element(SquareLen, SquareLen + data.N);
	}

	void ben_make() {
		benchmark.N = 100, benchmark.num = 100;
		benchmark.indice = new int* [benchmark.N];
		benchmark.innerproduct = new float* [benchmark.N];
		for (int j = 0; j < benchmark.N; j++) {
			benchmark.indice[j] = new int[benchmark.num];
			benchmark.innerproduct[j] = new float[benchmark.num];
		}

		Tuple a;

		lsh::progress_display pd(benchmark.N);
		for (int j = 0; j < benchmark.N; j++)
		{
			std::vector<Tuple> dists;
			dists.clear();
			for (int i = 0; i < data.N; i++)
			{
				a.id = i;
				a.inp = cal_inner_product(data.val[i], queries.val[j], data.dim);
				dists.push_back(a);
			}

			sort(dists.begin(), dists.end(), comp);
			for (int i = 0; i < benchmark.num; i++)
			{
				benchmark.indice[j][i] = (int)dists[i].id;
				benchmark.innerproduct[j][i] = dists[i].inp;
			}
			++pd;
		}

	}

	void ben_save() {
		std::ofstream out(ben_file.c_str(), std::ios::binary);
		out.write((char*)&benchmark.N, sizeof(int));
		out.write((char*)&benchmark.num, sizeof(int));

		for (int j = 0; j < benchmark.N; j++) {
			out.write((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
		}

		for (int j = 0; j < benchmark.N; j++) {
			out.write((char*)&benchmark.innerproduct[j][0], sizeof(float) * benchmark.num);
		}

		out.close();
	}

	void ben_load() {
		std::ifstream in(ben_file.c_str(), std::ios::binary);
		in.read((char*)&benchmark.N, sizeof(int));
		in.read((char*)&benchmark.num, sizeof(int));

		benchmark.indice = new int* [benchmark.N];
		benchmark.innerproduct = new float* [benchmark.N];
		for (int j = 0; j < benchmark.N; j++) {
			benchmark.indice[j] = new int[benchmark.num];
			in.read((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
		}

		for (int j = 0; j < benchmark.N; j++) {
			benchmark.innerproduct[j] = new float[benchmark.num];
			in.read((char*)&benchmark.innerproduct[j][0], sizeof(float) * benchmark.num);
		}
		in.close();
	}

	void ben_create() {
		int a_test = data.N + 1;
		lsh::timer timer;
		std::ifstream in(ben_file.c_str(), std::ios::binary);
		in.read((char*)&a_test, sizeof(int));
		in.close();
		if (a_test > 0 && a_test <= data.N)
		{
			std::cout << "LOADING BENMARK..." << std::endl;
			timer.restart();
			ben_load();
			std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
		}
		else
		{
			std::cout << "MAKING BENMARK..." << std::endl;
			timer.restart();
			ben_make();
			std::cout << "MAKING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

			std::cout << "SAVING BENMARK..." << std::endl;
			timer.restart();
			ben_save();
			std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
		}
	}

	~Preprocess() {
		//clear_2d_array(data.val, data.N);
		delete[] data.base;
		delete[] data.val;
		delete[] queries.base;
		delete[] queries.val;
		clear_2d_array(benchmark.indice, benchmark.N);
		clear_2d_array(benchmark.innerproduct, benchmark.N);
		delete[] SquareLen;
	}
};



class Partition
{
	private:
	float ratio;

	void make_chunks_maria(Preprocess& prep) {
		std::vector<Dist_id> distpairs;
		std::vector<int> bucket;
		//Dist_id pair;
		int N_ = prep.data.N;
		int n;
		int minsize = 32;
		for (int j = 0; j < N_; j++) {
			distpairs.emplace_back(j, -prep.SquareLen[j]);
		}
		std::sort(distpairs.begin(), distpairs.end());

		float norm_range = distpairs[0].dist / (distpairs[N_ - 1].dist);
		std::cout << "Norm Range is: " << norm_range << std::endl;
		// exit(-1);

		numChunks = 0;
		chunks.resize(N_);
		int j = 0;
		int lastn = 0;
		while (j < N_) {
			float M = distpairs[j].dist * ratio;
			//lastn = n;
			n = 0;
			bucket.clear();
			while (j < N_) {
				if ((n >= minsize) && (distpairs[j].dist > M || n >= MAXSIZE)
					&& (N_ - j > minsize)) break;

				chunks[distpairs[j].id] = numChunks;
				bucket.push_back(distpairs[j].id);
				j++;
				n++;
			}
			nums.push_back(n);
			MaxLen.push_back(-M / ratio);
			EachParti.push_back(bucket);
			bucket.clear();
			numChunks++;
			minsize *= 2;
		}
	}

	public:
	int numChunks;
	std::vector<float> MaxLen;

	std::vector<int> chunks;

	std::vector<int> nums;

	std::vector<std::vector<int>> EachParti;


	Partition(float c_, Preprocess& prep) {
		ratio = 0.85;
		make_chunks_maria(prep);
	}

	//Partition() {}
	~Partition() {}
};

// class Parameter //N,dim,S, L, K, M, W;
// {
// public:
// 	int N;
// 	int dim;
// 	// Number of hash functions
// 	int S;
// 	//#L Tables; 
// 	int L;
// 	// Dimension of the hash table
// 	int K;
// 	//
// 	int MaxSize = -1;
// 	//
// 	int KeyLen = -1;

// 	int M = 1;

// 	int W = 0;

// 	float U;
// 	Parameter(Preprocess& prep, int L_, int K_, int M);
// 	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_);
// 	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_, float W_);
// 	Parameter(Preprocess& prep, float c_, float S0);
// 	Parameter(Preprocess& prep, float c0_);
// 	bool operator = (const Parameter& rhs);
// 	~Parameter();
// };

#include <unordered_set>
class queryN
{
	public:
	// the parameter "c" in "c-ANN"
	float c;
	//which chunk is accessed
	//int chunks;

	//float R_min = 4500.0f;//mnist
	//float R_min = 1.0f;
	float init_w = 1.0f;

	float* queryPoint = NULL;
	float* hashval = NULL;
	uint16_t srpval[4] = { 0,0,0,0 }; //Initialize !!!!
	//float** myData = NULL;
	//int dim = 1;

	//int UB = 0;
	//float minKdist = FLT_MAX;
	// Set of points sifted
	std::priority_queue<Res> top_candidates;
	std::vector<bool> visited;
	std::unordered_set<int> visited_set;
	//std::vector<int> keys;

	public:
	// k-NN
	unsigned k = 1;
	// Indice of query point in dataset. Be equal to -1 if the query point isn't in the dataset.
	unsigned qid = -1;

	float beta = 0;
	float norm = 1.0f;
	unsigned cost = 0;

	//#access;
	int maxHop = -1;
	//
	unsigned prunings = 0;
	//cost of each partition
	std::vector<int> costs;
	//
	float time_total = 0;
	//
	float timeHash = 0;
	//
	float time_sift = 0;

	float time_verify = 0;
	// query result:<indice of ANN,distance of ANN>
	std::vector<Res> res;

	public:
	// queryN(unsigned id, float c_, unsigned k_, Preprocess& prep, float beta_) {
	// 	qid = id;
	// 	c = c_;
	// 	k = k_;
	// 	beta = beta_;
	// 	//myData = prep.data.val;
	// 	//dim = prep.data.dim + 1;
	// 	queryPoint = prep.queries[id];
	// 	float dim = prep.data.dim;
	// 	norm = sqrt(cal_inner_product(queryPoint, queryPoint, prep.data.dim));
	// 	// for (int i = 0;i < dim;++i) {
	// 	// 	queryPoint[i] /= norm;
	// 	// }
	// 	//search();
	// }

	queryN(unsigned id, float c_, unsigned k_, float* query, int dim, float beta_) {
		qid = id;
		c = c_;
		k = k_;
		beta = beta_;
		//myData = prep.data.val;
		//dim = dim;
		//std::vector<float> qvec()
		queryPoint = new float[dim];
		memcpy(queryPoint, query, sizeof(float) * dim);
		//queryPoint = query;

		norm = sqrt(cal_inner_product(queryPoint, queryPoint, dim));
		for (int i = 0;i < dim;++i) {
			queryPoint[i] /= norm + 1e-30f;
		}
		//float norm1 = sqrt(cal_inner_product(queryPoint, queryPoint, dim));

		//search();
	}
	//void search();

	~queryN() {
		delete[] hashval;
		//delete queryPoint;
	}
};

class Parameter //N,dim,S, L, K, M, W;
{
	public:
	unsigned N = 0;
	unsigned dim = 0;
	// Number of hash functions
	unsigned S = 0;
	//#L Tables; 
	unsigned L = 0;
	// Dimension of the hash table
	unsigned K = 0;

	float W = 1.0f;
	int MaxSize = 0;

	float R_min = 0.3f;

	Parameter(Data& data, unsigned L_, unsigned K_, float rmin_) {
		N = data.N;
		dim = data.dim;
		L = L_;
		K = K_;
		MaxSize = 5;
		R_min = rmin_;
	}
	~Parameter() {}
};

