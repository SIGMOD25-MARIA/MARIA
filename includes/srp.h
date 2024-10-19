#pragma once
#include "utils/StructType.h"
#include "utils/Preprocess.h"
#include <cmath>
#include <assert.h>
#include <vector>
#include <queue>
#include <cfloat>

// #define USE_BLAS

// #ifdef USE_BLAS
#if defined(__GNUC__) && defined(USE_BLAS)
#include <cblas.h>
#endif

namespace lsh
{
	//The prority_queue only for building APG
	struct  fixxed_length_priority_queue
	{
		private:
		int size_ = 0;
		int capacity = 0;
		Res* data_ = nullptr;

		public:
		fixxed_length_priority_queue() = default;
		// fixxed_length_priority_queue(int K) {
		//     reset(K);
		// }

		fixxed_length_priority_queue(int K, Res* data) {
			capacity = K;
			data_ = data;
		}

		Res& operator[](int i) { return data_[i]; }
		// void reset(int K) {
		//     //capacity = K;
		//     delete[] data_;
		//     data_ = new Res[capacity];
		// }

		inline void emplace(int id, float dist) {
			if (size_ == capacity) {
				if (dist > data_[0].dist) return;
				pop();
				data_[size_] = Res(id, dist);
				std::push_heap(data_, data_ + size_);
			}
			data_[size_++] = Res(id, dist);
			std::push_heap(data_, data_ + size_);
		}

		void push(Res res) {
			if (size_ == capacity) {
				if (res.dist > data_[0].dist) return;
				pop();
				data_[size_] = res;
				std::push_heap(data_, data_ + size_);
			}
			data_[size_++] = res;
			std::push_heap(data_, data_ + size_);
		}

		void emplace_with_duplication(int id, float dist) {
			for (int i = 0;i < size_;++i) {
				if (data_[i].id == id) return;
			}
			emplace(id, dist);
		}

		void pop() {
			std::pop_heap(data_, data_ + size_);
			size_--;
		}

		int size() {
			return size_;
		}

		bool empty() {
			return size_ == 0;
		}

		Res& top() {
			return data_[0];
		}

		Res*& data() {
			return data_;
		}

		~fixxed_length_priority_queue() {}
	};

	struct srpPair {
		int id = -1;
		uint16_t val = 0;
		srpPair() = default;
		srpPair(int id_, uint16_t hashval) : id(id_), val(hashval) {}

		bool operator<(const srpPair& rhs) const { return val < rhs.val; }
	};

	struct hash_t {
		void reset(int N_) {
			delete[] base;
			N = N_;
			base = new uint16_t[N * 4];
			memset(base, 0, sizeof(uint16_t) * N * 4);
		}

		uint16_t* operator[](size_t i) { return base + i * 4; }
		size_t size() const { return N; }

		~hash_t() {
			delete[] base;
		}
		private:
		uint16_t* base = nullptr;
		size_t N = 0;
		//const int L = 4;
	};

	// My implement for a simple sign random prejection LSH function class
	class srp
	{
		// int N=0;

		//N * L;



		std::vector<std::vector<srpPair>> hash_tables;
		std::vector<std::vector<int>>& part_map;
		Data data;
		std::string index_file;
		std::atomic<size_t> cost{ 0 };
		float* rndAs = nullptr;
		int dim = 0;
		// Number of hash functions
		int S = 0;
		// #L Tables;
		int L = 0;
		// Dimension of the hash table
		int K = 0;
		float indexing_time = 0.0f;
		public:
		const std::string alg_name = "srp";
		//std::vector<std::vector<uint16_t>> hashvals;
		//std::vector<uint16_t[4]> hashvals;

		hash_t hashvals;
		size_t getCost() {
			return cost;
		}

		int numBlocks() {
			return part_map.size();
		}

		std::vector<int>& getIds(int i) {
			return part_map[i];
		}

		srp() = default;

		srp(Data& data_, std::vector<std::vector<int>>& part_map_, const std::string& index_file_,
			int N_, int dim_, int L_ = 4, int K_ = 16, bool isbuilt = 1) :part_map(part_map_)
		{
			data = data_;
			// N=N_;
			dim = dim_;
			L = L_;
			K = K_;
			S = L * K;
			hashvals.reset(N_);
			index_file = index_file_;
			if (L > 4 || K > 16) {
				std::cerr << "The valid ranges of L and K are: 1<=L<=4, 1<=K<=16" << std::endl;
				exit(-1);
			}

			//std::ifstream in(index_file, std::ios::binary);
			lsh::timer timer;
			if (!(isbuilt && exists_test(index_file))) {
				float mem = (float)getCurrentRSS() / (1024 * 1024);
				buildIndex();
				float memf = (float)getCurrentRSS() / (1024 * 1024);
				indexing_time = timer.elapsed();
				std::cout << "SRP Building time:" << indexing_time << "  seconds.\n";

				saveIndexInfo(index_file, "./indexes/maria_info.txt", memf - mem, indexing_time);
				// FILE* fp = nullptr;
				// fopen_s(&fp, "./indexes/maria_info.txt", "a");
				// if (fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), memf - mem, indexing_time);

				saveIndex();
			}
			else {
				//in.close();
				std::cout << "Loading index from " << index_file << ":\n";
				float mem = (float)getCurrentRSS() / (1024 * 1024);
				loadIndex();
				float memf = (float)getCurrentRSS() / (1024 * 1024);
				std::cout << "Actual memory usage: " << memf - mem << " Mb \n";

			}
		}

		srp(Data& data_, std::vector<std::vector<int>>& part_map_, int L_ = 2, int K_ = 2) :part_map(part_map_)
		{
			data = data_;
			// N=N_;
			dim = data.dim;
			L = L_;
			K = K_;
			S = L * K;
		}

		void buildIndex() {
			std::cout << std::endl
				<< "START HASHING..." << std::endl
				<< std::endl;
			lsh::timer timer;

			std::cout << "SETTING HASH PARAMETER..." << std::endl;
			timer.restart();
			SetHash();
			std::cout << "SETTING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;

			std::cout << "COMPUTING HASH..." << std::endl;
			timer.restart();
			GetHash(data);
			std::cout << "COMPUTING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;

			std::cout << "BUILDING INDEX..." << std::endl;
			std::cout << "THERE ARE " << L << " " << K << "-D HASH TABLES." << std::endl;
			timer.restart();

			if (part_map.empty())
				GetTables();
			else
				GetTables(part_map);

			std::cout << "BUILDING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;
		}

		void SetHash()
		{
			rndAs = new float[S * dim];
			// hashpar.rndAs2 = new float* [S];

			std::mt19937 rng(int(std::time(0)));
			// std::mt19937 rng(int(0));
			std::normal_distribution<float> nd;
			for (int i = 0; i < S * dim; ++i)
				rndAs[i] = (nd(rng));
		}

		void GetHash(Data& data)
		{
#if defined(__GNUC__) && defined(USE_BLAS)
			int m = hashvals.size();
			int k = dim;
			int n = S;

			float* A = data.base;
			float* B = rndAs;
			float* C = new float[m * n];

			memset(C, 0.0f, m * n * sizeof(float));
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				m, n, k, 1.0, A, k, B, k, 0.0, C, n);

			for (int i = 0; i < hashvals.size(); ++i)
			{
				hashvals[i].resize(L, 0);
				for (int j = 0; j < L; ++j)
				{
					for (int l = 0; l < K; ++l)
					{
						float val = C[i * S + j * K + l];
						// cal_inner_product(data[i],rndAs+(j*K+l)*dim,dim);
						if (val > 0)
							hashvals[i][j] |= (1 << l);
					}
				}
			}
#else

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = 0; i < hashvals.size(); ++i)
			{
				//hashvals[i].resize(L, 0);
				hashvals[i][0] = 0;
				hashvals[i][1] = 0;
				hashvals[i][2] = 0;
				hashvals[i][3] = 0;
				for (int j = 0; j < L; ++j)
				{

					for (int l = 0; l < K; ++l)
					{
						float val = cal_inner_product(data[i], rndAs + (j * K + l) * dim, dim);
						if (val > 0)
							hashvals[i][j] |= (1 << l);
					}
				}
			}
#endif

			// for(int i=0;i<10;++i){
			// 	for(int j=0;j<L;++j){
			// 		std::cout<<hashvals[i][j]<<" ";

			// 	}
			// 	std::cout<<std::endl;
			// }
		}

#include <algorithm>
		void myitoa(int n, char s[], int b = 10)
		{
			int i, sign;

			if ((sign = n) < 0)  /* record sign */
				n = -n;          /* make n positive */
			i = 0;
			do {       /* generate digits in reverse order */
				s[i++] = n % b + '0';   /* get next digit */
			} while ((n /= b) > 0);     /* delete it */
			if (sign < 0)
				s[i++] = '-';
			s[i] = '\0';
			std::reverse(s, s + i - 1);
		}

		void display() {
			for (int i = 0;i < S;++i) {
				printf("a[%d][%d]=%2.2f\n", i / L, i % L, rndAs[i]);
			}
			for (auto& part : part_map) {
				std::cout << "A block:" << std::endl;
				for (auto& id : part) {
					char s1[10], s2[10], s3[10];
					int quant = hashvals[id][0] * 4 + hashvals[id][1];
					myitoa(hashvals[id][0], s1, 2);
					myitoa(hashvals[id][1], s2, 2);
					myitoa(quant, s3, 2);
					printf("data[%2d]=(%2.2f,%2.2f), its hashvals={%1d(%s),%1d(%s)},its quantization=%2d(%s)\n",
						id, data[id][0], data[id][1], hashvals[id][0], s1, hashvals[id][1], s2, quant, s3);
				}
				std::cout << std::endl;
			}

			for (int j = 0;j < part_map.size();++j) {
				std::cout << "The block:" << std::endl;
				printf("The block:%d\n", j + 1);
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[j * L + i];
					printf("T%d,%d:\n", j + 1, i + 1);
					for (auto& x : table) {
						printf("hashpair:(%d,x_%d)\n", (int)x.val, part_map[j][x.id]);
					}
				}
				std::cout << std::endl;
			}
		}

		void GetTables(std::vector<std::vector<int>>& part_map)
		{
			int num_parti = part_map.size();
			hash_tables.resize(num_parti * L);
			for (int i = 0; i < num_parti; ++i)
			{
				auto& part = part_map[i];
				// for (auto& id : part) {
				for (int l = 0; l < part.size(); ++l)
				{
					int id = part[l];
					for (int j = 0; j < L; ++j) {
						hash_tables[i * L + j].emplace_back(l, hashvals[id][j]);
					}
				}
			}

			for (auto& table : hash_tables)
			{
				std::sort(table.begin(), table.end());
			}
		}

		void GetTables()
		{
			hash_tables.resize(L);
			for (int i = 0; i < hashvals.size(); ++i)
			{
				int id = i;
				for (int j = 0; j < L; ++j)
				{
					hash_tables[j].emplace_back(id, hashvals[id][j]);
				}
			}

			for (auto& table : hash_tables)
			{
				std::sort(table.begin(), table.end());
			}
		}

		void saveIndex() {

			std::string file = index_file;
			std::ofstream out(file, std::ios::binary);

			out.write((char*)(&L), sizeof(int));
			out.write((char*)(&K), sizeof(int));
			out.write((char*)(&dim), sizeof(int));
			S = L * K;
			//save hashpar
			out.write((char*)(rndAs), sizeof(float) * S * dim);

			//save hashvals
			int N = hashvals.size();
			out.write((char*)(&N), sizeof(int));
			out.write((char*)(hashvals[0]), sizeof(uint16_t) * N * 4);
			//for (int i = 0;i < N;++i) {
			//	out.write((char*)&(hashvals[i][0]), sizeof(uint16_t) * L);
			//}

			//save hash tables
			int ntb = hash_tables.size();
			out.write((char*)(&ntb), sizeof(int));
			for (int j = 0; j < ntb; ++j) {
				int np = hash_tables[j].size();
				out.write((char*)(&np), sizeof(int));
				out.write((char*)(hash_tables[j].data()), sizeof(srpPair) * np);
			}
		}

		void loadIndex() {

			std::string file = index_file;
			std::ifstream in(file, std::ios::binary);

			in.read((char*)(&L), sizeof(int));
			in.read((char*)(&K), sizeof(int));
			in.read((char*)(&dim), sizeof(int));
			S = L * K;

			//load hashpar
			rndAs = new float[S * dim];
			in.read((char*)(rndAs), sizeof(float) * S * dim);

			//load hashvals
			int N = 0;
			in.read((char*)(&N), sizeof(int));
			hashvals.reset(N);
			in.read((char*)(hashvals[0]), sizeof(uint16_t) * N * 4);
			//for (int i = 0;i < N;++i) {
			//	//hashvals[i].resize(L);
			//	in.read((char*)(hashvals[i]), sizeof(uint16_t) * L);
			//}

			//load hash tables
			int ntb = 0;
			in.read((char*)(&ntb), sizeof(int));
			hash_tables.resize(ntb);
			for (int j = 0; j < ntb; ++j) {
				int np = 0;
				in.read((char*)(&np), sizeof(int));
				hash_tables[j].resize(np);
				in.read((char*)(hash_tables[j].data()), sizeof(srpPair) * np);
			}
		}

		void kjoin(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width)
			{
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			knns.resize(n);
			for (auto& nnset : knns) nnset.reserve(2 * width * L);

			for (int i = np * L; i < np * L + L; ++i)
			{
				auto& table = hash_tables[i];
				for (int j = 0; j < width; ++j)
				{
					for (int l = 0; l < j + width; ++l)
					{
						if (j != l)
							knns[table[j].id].emplace_back(table[l].id, 1.0f);
					}
				}
				for (int j = width; j < n - width; ++j)
				{
					for (int l = j - width; l < j + width; ++l)
					{
						if (j != l)
							knns[table[j].id].emplace_back(table[l].id, 1.0f);
					}
				}
				for (int j = n - width; j < n; ++j)
				{
					for (int l = j - width; l < n; ++l)
					{
						if (j != l)
							knns[table[j].id].emplace_back(table[l].id, 1.0f);
					}
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
				pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < knns.size(); ++i)
			{
				auto& pool = knns[i];
				for (auto& x : pool)
				{
					x.dist = calInnerProductReverse(data[ids[x.id]], data[ids[i]], dim);
#if defined(COUNT_CC)
					cost++;
#endif
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				std::sort(pool.begin(), pool.end());
				if (pool.size() > K)
					pool.resize(K);
			}
		}

		void kjoin1(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width) {
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}

			int lc = width * 2 + 1;
			knns.resize(n, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = np * L; i < np * L + L; ++i) {
				auto& table = hash_tables[i];
				int bias = (i - np * L) * lc + width;
				for (int j = 0; j < width; ++j)
				{
					for (int l = 0; l < j; ++l)
					{
						{
							// float inp = calInnerProductReverse(data[ids[table[j].id]],
							// 	data[table[l].id], dim);

							float inp = calInnerProductReverse(data[ids[table[j].id]],
								data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
							cost++;
#endif
							knns[table[j].id][j - l + bias] = Res(table[l].id, inp);
							knns[table[l].id][l - j + bias] = Res(table[j].id, inp);
							// knns[table[j].id].emplace_back(table[l].id, inp);
							// knns[table[l].id].emplace_back(table[j].id, inp);
						}
					}
				}

#pragma omp parallel for schedule(dynamic, 256)
				for (int j = width; j < n; ++j)
				{
					for (int l = j - width; l < j; ++l)
					{
						{
							float inp = calInnerProductReverse(data[ids[table[j].id]],
								data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
							cost++;
#endif
							knns[table[j].id][j - l + bias] = Res(table[l].id, inp);
							knns[table[l].id][l - j + bias] = Res(table[j].id, inp);
							// knns[table[j].id].emplace_back(table[l].id, inp);
							// knns[table[l].id].emplace_back(table[j].id, inp);
						}
					}
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				std::sort(pool.begin(), pool.end());
				// if (pool.size() > K) pool.resize(K);
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
				pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				// std::sort(pool.begin(), pool.end());
				if (pool.back().id == -1)
					pool.pop_back();
				if (pool.size() > K)
					pool.resize(K);
			}
		}

		void kjoin1_new(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width) {
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}

			int lc = width * 2 + 1;
			knns.resize(n, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = np * L; i < np * L + L; ++i) {
				auto& table = hash_tables[i];
				int bias = (i - np * L) * lc + width;
				for (int j = 0; j < width; ++j)
				{
					for (int l = 0; l < j; ++l)
					{
						{
							// float inp = calInnerProductReverse(data[ids[table[j].id]],
							// 	data[table[l].id], dim);

							float inp = calInnerProductReverse(data[ids[table[j].id]],
								data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
							cost++;
#endif
							knns[table[j].id][j - l + bias] = Res(ids[table[l].id], inp);
							knns[table[l].id][l - j + bias] = Res(ids[table[j].id], inp);
							// knns[table[j].id].emplace_back(table[l].id, inp);
							// knns[table[l].id].emplace_back(table[j].id, inp);
						}
					}
				}

#pragma omp parallel for schedule(dynamic, 256)
				for (int j = width; j < n; ++j)
				{
					for (int l = j - width; l < j; ++l)
					{
						{
							float inp = calInnerProductReverse(data[ids[table[j].id]],
								data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
							cost++;
#endif
							knns[table[j].id][j - l + bias] = Res(ids[table[l].id], inp);
							knns[table[l].id][l - j + bias] = Res(ids[table[j].id], inp);
							// knns[table[j].id].emplace_back(table[l].id, inp);
							// knns[table[l].id].emplace_back(table[j].id, inp);
						}
					}
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				std::sort(pool.begin(), pool.end());
				// if (pool.size() > K) pool.resize(K);
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
				pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				// std::sort(pool.begin(), pool.end());
				while (pool.back().id == -1)
					pool.pop_back();
				if (pool.size() > K)
					pool.resize(K);
			}
		}

#ifdef COUNT_PD
		lsh::progress_display* pd = nullptr;
		void resetPD() {
			pd = new lsh::progress_display((size_t)L * hashvals.size());
		}

		void updatePD(int k) {
			(*pd) += L * k;
		}

		void dropPD() {
			delete pd;
		}
#endif
		void kjoin2_new(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width) {
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			knns.resize(n);
			std::vector<mp_mutex> locks(n);
			//std::vector<std::priority_queue<Res>> top_candidates;


			//Caonot initialize them as std::vector<lsh::priority_queue> top_candidates(n);
			//Otherwise, all heaps have have the same momory position.
			std::vector<lsh::priority_queue> top_candidates(n);
			for (int i = 0; i < n; ++i) top_candidates[i].reset(K);
			// int lc = width * 2 + 1;
			// knns.resize(n, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = np * L; i < np * L + L; ++i) {
				auto& table = hash_tables[i];
				//int bias = (i - np * L) * lc + width;
				for (int j = 0; j < width; ++j) {
					for (int l = 0; l < j; ++l) {
						float inp = calInnerProductReverse(data[ids[table[j].id]],
							data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
						cost++;
#endif
						{
							write_lock lock(locks[table[j].id]);
							top_candidates[table[j].id].emplace_with_duplication(ids[table[l].id], inp);
							if (top_candidates[table[j].id].size() > K) top_candidates[table[j].id].pop();
						}

						{
							write_lock lock(locks[table[l].id]);
							top_candidates[table[l].id].emplace_with_duplication(ids[table[j].id], inp);
							if (top_candidates[table[l].id].size() > K) top_candidates[table[l].id].pop();
						}

					}
#ifdef COUNT_PD
					++(*pd);
#endif
				}

#pragma omp parallel for schedule(dynamic, 256)
				for (int j = width; j < n; ++j)
				{
					for (int l = j - width; l < j; ++l)
					{
						float inp = calInnerProductReverse(data[ids[table[j].id]],
							data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
						cost++;
#endif
						{
							write_lock lock(locks[table[j].id]);
							top_candidates[table[j].id].emplace_with_duplication(ids[table[l].id], inp);
							if (top_candidates[table[j].id].size() > K) top_candidates[table[j].id].pop();
						}

						{
							write_lock lock(locks[table[l].id]);
							top_candidates[table[l].id].emplace_with_duplication(ids[table[j].id], inp);
							if (top_candidates[table[l].id].size() > K) top_candidates[table[l].id].pop();
						}
					}
#ifdef COUNT_PD
					++(*pd);
#endif
				}


			}

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = 0;i < n;++i) {
				knns[i].resize(top_candidates[i].size());
				memcpy(knns[i].data(), top_candidates[i].data(), sizeof(Res) * top_candidates[i].size());
				top_candidates[i].clear();
			}
		}

		void kjoin3_new(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width) {
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			knns.resize(n);
			std::vector<mp_mutex> locks(n);
			//std::vector<std::priority_queue<Res>> top_candidates;


			//Caonot initialize them as std::vector<lsh::priority_queue> top_candidates(n);
			//Otherwise, all heaps have have the same momory position.
			std::vector<lsh::priority_queue> top_candidates(n);
			for (int i = 0; i < n; ++i) top_candidates[i].reset(K);
			// int lc = width * 2 + 1;
			// knns.resize(n, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = np * L; i < np * L + L; ++i) {
				auto& table = hash_tables[i];
				//int bias = (i - np * L) * lc + width;
				for (int j = 0; j < width; ++j) {
					for (int l = 0; l < j; ++l) {
						float inp = calInnerProductReverse(data[ids[table[j].id]],
							data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
						cost++;
#endif
						if (table[j].id > table[l].id) {
							write_lock lock(locks[table[j].id]);
							top_candidates[table[j].id].emplace_with_duplication(ids[table[l].id], inp);
							if (top_candidates[table[j].id].size() > K) top_candidates[table[j].id].pop();
						}
						else {
							write_lock lock(locks[table[l].id]);
							top_candidates[table[l].id].emplace_with_duplication(ids[table[j].id], inp);
							if (top_candidates[table[l].id].size() > K) top_candidates[table[l].id].pop();
						}

					}
#ifdef COUNT_PD
					++(*pd);
#endif
				}

#pragma omp parallel for schedule(dynamic, 256)
				for (int j = width; j < n; ++j)
				{
					for (int l = j - width; l < j; ++l)
					{
						float inp = calInnerProductReverse(data[ids[table[j].id]],
							data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
						cost++;
#endif
						if (table[j].id > table[l].id) {
							write_lock lock(locks[table[j].id]);
							top_candidates[table[j].id].emplace_with_duplication(ids[table[l].id], inp);
							if (top_candidates[table[j].id].size() > K) top_candidates[table[j].id].pop();
						}
						else {
							write_lock lock(locks[table[l].id]);
							top_candidates[table[l].id].emplace_with_duplication(ids[table[j].id], inp);
							if (top_candidates[table[l].id].size() > K) top_candidates[table[l].id].pop();
						}
					}
#ifdef COUNT_PD
					++(*pd);
#endif
				}


			}

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = 0;i < n;++i) {
				knns[i].resize(top_candidates[i].size());
				memcpy(knns[i].data(), top_candidates[i].data(), sizeof(Res) * top_candidates[i].size());
				top_candidates[i].clear();
			}
		}

		void kjoin4_new(std::vector<fixxed_length_priority_queue>& knng, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width) {
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			std::vector<mp_mutex> locks(n);
			//std::vector<std::priority_queue<Res>> top_candidates;


			//Caonot initialize them as std::vector<lsh::priority_queue> top_candidates(n);
			//Otherwise, all heaps have have the same momory position.
			//std::vector<fixxed_length_priority_queue> top_candidates(n);
			//for (int i = 0; i < n; ++i) top_candidates[i].reset(K);
			// int lc = width * 2 + 1;
			// knns.resize(n, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = np * L; i < np * L + L; ++i) {
				auto& table = hash_tables[i];
				//int bias = (i - np * L) * lc + width;
				for (int j = 0; j < width; ++j) {
					for (int l = 0; l < j; ++l) {
						float inp = calInnerProductReverse(data[ids[table[j].id]],
							data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
						cost++;
#endif
						if (table[j].id > table[l].id) {
							write_lock lock(locks[table[j].id]);
							knng[ids[table[j].id]].emplace_with_duplication(ids[table[l].id], inp);
							//if (top_candidates[table[j].id].size() > K) top_candidates[table[j].id].pop();
						}
						else {
							write_lock lock(locks[table[l].id]);
							knng[ids[table[l].id]].emplace_with_duplication(ids[table[j].id], inp);
							//top_candidates[table[l].id].emplace_with_duplication(ids[table[j].id], inp);
							//if (top_candidates[table[l].id].size() > K) top_candidates[table[l].id].pop();
						}

					}
#ifdef COUNT_PD
					++(*pd);
#endif
				}

#pragma omp parallel for schedule(dynamic, 256)
				for (int j = width; j < n; ++j)
				{
					for (int l = j - width; l < j; ++l)
					{
						float inp = calInnerProductReverse(data[ids[table[j].id]],
							data[ids[table[l].id]], dim);
#if defined(COUNT_CC)
						cost++;
#endif
						if (table[j].id > table[l].id) {
							write_lock lock(locks[table[j].id]);
							knng[ids[table[j].id]].emplace_with_duplication(ids[table[l].id], inp);
							//if (top_candidates[table[j].id].size() > K) top_candidates[table[j].id].pop();
						}
						else {
							write_lock lock(locks[table[l].id]);
							knng[ids[table[l].id]].emplace_with_duplication(ids[table[j].id], inp);
							//top_candidates[table[l].id].emplace_with_duplication(ids[table[j].id], inp);
							//if (top_candidates[table[l].id].size() > K) top_candidates[table[l].id].pop();
						}
					}
#ifdef COUNT_PD
					++(*pd);
#endif
				}


			}

			// #pragma omp parallel for schedule(dynamic, 256)
			// 			for (int i = 0;i < n;++i) {
			// 				knns[i].resize(top_candidates[i].size());
			// 				memcpy(knns[i].data(), top_candidates[i].data(), sizeof(Res) * top_candidates[i].size());
			// 				top_candidates[i].clear();
			// 			}
		}

		//find NNS in np1 for the points in np2
		void kjoin(std::vector<std::vector<Res>>& knns, std::vector<int>& ids1, int np1,
			std::vector<int>& ids2, int np2, int K, int width)
		{
			int n1 = hash_tables[np1 * L].size();
			if (n1 < 2 * width)
			{
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			int n2 = hash_tables[np2 * L].size();
			int lc = width * 2 + 1;
			knns.resize(n2, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));
			// knns.resize(n2);
			// for (auto& nnset : knns) nnset.reserve(2 * lc);

#pragma omp parallel for
			for (int i = 0; i < L; ++i) {
				auto& table1 = hash_tables[i + np1 * L];
				auto& table2 = hash_tables[i + np2 * L];
				int bias = i * lc;

				int pos1 = 0, pos2 = 0;

				while (pos2 < n2) {
					while (pos1 < n1 && table1[pos1].val < table2[pos2].val) pos1++;

					int start = std::max(pos1 - width, 0);
					int end = std::min(pos1 + width, n1 - 1);

					auto& vec2 = data[ids2[table2[pos2].id]];
					for (int j = start;j <= end;++j) {
						auto& vec1 = data[ids1[table1[j].id]];
						float dist = calInnerProductReverse(vec1, vec2, data.dim);
						//knns[table2[pos2].id][bias + j - start] = Res(table1[j].id, dist);
						knns[table2[pos2].id][bias + j - start] = Res(ids1[table1[j].id], dist);
					}
					pos2++;
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) {
				std::sort(pool.begin(), pool.end());
				// if (pool.size() > K) pool.resize(K);
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) {
				if (pool.back().id == -1) pool.pop_back();
				if (pool.size() > K) pool.resize(K);
			}
		}


		//find NNS in np1 for the points in np2
		void kjoin_new(std::vector<std::vector<Res>>& knns, std::vector<int>& ids1, int np1,
			std::vector<int>& ids2, int np2, int K, int width)
		{
			int n1 = hash_tables[np1 * L].size();
			if (n1 < 2 * width)
			{
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			int n2 = hash_tables[np2 * L].size();
			int lc = width * 2 + 1;
			knns.resize(n2, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));
			// knns.resize(n2);
			// for (auto& nnset : knns) nnset.reserve(2 * lc);

#pragma omp parallel for
			for (int i = 0; i < L; ++i) {
				auto& table1 = hash_tables[i + np1 * L];
				auto& table2 = hash_tables[i + np2 * L];
				int bias = i * lc;

				int pos1 = 0, pos2 = 0;

				while (pos2 < n2) {
					while (pos1 < n1 && table1[pos1].val < table2[pos2].val) pos1++;

					int start = std::max(pos1 - width, 0);
					int end = std::min(pos1 + width, n1 - 1);

					auto& vec2 = data[ids2[table2[pos2].id]];
					for (int j = start;j <= end;++j) {
						auto& vec1 = data[ids1[table1[j].id]];
						float dist = calInnerProductReverse(vec1, vec2, data.dim);
						knns[table2[pos2].id][bias + j - start] = Res(ids1[table1[j].id], dist);
					}
					pos2++;
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) {
				std::sort(pool.begin(), pool.end());
				// if (pool.size() > K) pool.resize(K);
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) {
				if (pool.back().id == -1) pool.pop_back();
				if (pool.size() > K) pool.resize(K);
			}
		}

		void calQHash(queryN* q) {
			//hashvals[i].resize(L, 0);
			auto& vals = q->srpval;
			for (int j = 0; j < L; ++j)
			{
				for (int l = 0; l < K; ++l)
				{
					float val = cal_inner_product(q->queryPoint, rndAs + (j * K + l) * dim, dim);
					if (val > 0)
						vals[j] |= (1 << l);
				}
			}
		}

		void knn(queryN* q) {
			lsh::timer timer;
			//int np = part_map.size() - 1;
			int cnt = 0;
			int ub = 200;
			std::vector<bool>& visited = q->visited;
			visited.resize(data.N);

			//std::cerr << "here!" << std::endl;

			calQHash(q);

			for (int j = 0; j < part_map.size(); ++j) {
				float max_norm = sqrt(cal_inner_product(data[part_map[j][0]], data[part_map[j][0]], data.dim));
				if ((q->top_candidates.size() > q->k) && (-(q->top_candidates.top().dist)) >
					max_norm)
					break;
				int size = part_map[j].size();
				if (part_map[j].size() < 256) {
					for (auto& u : part_map[j]) {
						visited[u] = true;
						q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
						if (q->top_candidates.size() > q->k) {
							q->top_candidates.pop();
						}
					}

					continue;
				}

				ub = 0.9 * size;
				int num_candidates = 0;
				uint32_t diff = 1;
				int lpos[4];
				int rpos[4];
				uint16_t lval[4], rval[4];
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + j * L];
					//rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
					rpos[i] = std::distance(table.begin(), std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])));
					lpos[i] = rpos[i] - 1;
					while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
						lpos[i]--;
					}
					num_candidates += rpos[i] - lpos[i] - 1;
				}

				while (num_candidates < ub) {
					num_candidates = 0;
					diff *= 2;
					for (int i = 0;i < L;++i) {
						auto& table = hash_tables[i + j * L];
						lval[i] = q->srpval[i] / diff * diff;
						rval[i] = lval[i] + diff;
						while (lpos[i] >= 0 && table[lpos[i]].val >= lval[i]) {
							lpos[i]--;
						}

						while (rpos[i] < size && table[rpos[i]].val <= rval[i]) {
							rpos[i]++;
						}
						num_candidates += rpos[i] - lpos[i] - 1;
					}

				}

				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + j * L];
					for (int l = lpos[i] + 1;l < rpos[i];++l) {
						int u = part_map[j][table[l].id];
						if (visited[u]) continue;
						visited[u] = true;
						q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
						cnt++;
						//if (cnt > ub) break;
						if (q->top_candidates.size() > q->k) {
							q->top_candidates.pop();
						}
					}
				}

				q->cost += cnt;

				while (q->top_candidates.size()) {
					q->res.emplace_back(q->top_candidates.top());
					q->top_candidates.pop();
				}
				std::reverse(q->res.begin(), q->res.end());
				// if ((!q->top_candidates.empty()) && (-(q->top_candidates.top().dist)) >
				// 	q->norm * sqrt(parti.MaxLen[i]))
				// 	break;
			}






		}

		void knnInner(queryN* q, int np = 0) {
			// int np = part_map.size() - 1;
			// np = 0;
			int cnt = 0;
			int ub = 200;
			std::vector<bool>& visited = q->visited;

			int size = part_map[np].size();
			if (part_map[np].size() < ub) {
				for (auto& u : part_map[np]) {
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
				}

				return;
			}

			calQHash(q);

			int num_candidates = 0;
			uint32_t diff = 1;
			int lpos[4];
			int rpos[4];
			uint16_t lval[4], rval[4];
			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				//rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
				rpos[i] = std::distance(table.begin(), std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])));
				lpos[i] = rpos[i] - 1;
				while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
					lpos[i]--;
				}
				num_candidates += rpos[i] - lpos[i] - 1;
			}

			while (num_candidates < ub) {
				num_candidates = 0;
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + np * L];
					lval[i] = q->srpval[i] / diff * diff;
					rval[i] = lval[i] + diff;
					while (lpos[i] >= 0 && table[lpos[i]].val >= lval[i]) {
						lpos[i]--;
					}

					while (rpos[i] < size && table[rpos[i]].val <= rval[i]) {
						rpos[i]++;
					}
					num_candidates += rpos[i] - lpos[i] - 1;
				}
				diff *= 2;
			}

			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				for (int j = lpos[i] + 1;j < rpos[i];++j) {
					int u = part_map[np][table[j].id];
					if (visited[u]) continue;
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
					cnt++;
					if (cnt > ub) break;
				}
			}

			q->cost += ub;
			while (q->top_candidates.size() > q->k) q->top_candidates.pop();
		}

		void knnMaria(queryN* q, std::priority_queue<Res>& ep, int np = 0) {

			// int np = part_map.size() - 1;
			// np = 0;
			int cnt = 0;
			int ub = 200;
			std::vector<bool>& visited = q->visited;
			//auto& visited = q->visited_set;
			int size = part_map[np].size();
			if (part_map[np].size() < ub) {
				for (auto& u : part_map[np]) {
					visited[u] = true;
					//visited.emplace(u);
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
				}

				return;
			}

			calQHash(q);

			int num_candidates = 0;
			uint32_t diff = 1;
			int lpos[4];
			int rpos[4];
			uint16_t lval[4], rval[4];
			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				//rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
				rpos[i] = std::distance(table.begin(), std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])));
				lpos[i] = rpos[i] - 1;
				while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
					lpos[i]--;
				}
				num_candidates += rpos[i] - lpos[i] - 1;
			}

			while (num_candidates < ub) {
				num_candidates = 0;
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + np * L];
					lval[i] = q->srpval[i] / diff * diff;
					rval[i] = lval[i] + diff;
					while (lpos[i] >= 0 && table[lpos[i]].val >= lval[i]) {
						lpos[i]--;
					}

					while (rpos[i] < size && table[rpos[i]].val <= rval[i]) {
						rpos[i]++;
					}
					num_candidates += rpos[i] - lpos[i] - 1;
				}
				diff *= 2;
			}

			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				for (int j = lpos[i] + 1;j < rpos[i];++j) {
					int u = part_map[np][table[j].id];
					if (visited[u]) continue;
					visited[u] = true;
					float dist = cal_inner_product(q->queryPoint, data[u], data.dim);
					q->top_candidates.emplace(u, -dist);
					ep.emplace(u, dist);
					cnt++;
					if (cnt > ub) break;
				}
			}

			q->cost += ub;
			while (q->top_candidates.size() > q->k) q->top_candidates.pop();
			while (ep.size() > q->k) ep.pop();
		}

		void knnMaria1(queryN* q, std::priority_queue<Res>& ep, int np = 0) {

			// int np = part_map.size() - 1;
			// np = 0;
			int cnt = 0;
			int ub = 200;
			//std::vector<bool>& visited = q->visited;
			auto& visited = q->visited_set;
			int size = part_map[np].size();
			if (part_map[np].size() < ub) {
				for (auto& u : part_map[np]) {
					//visited[u] = true;
					visited.emplace(u);
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
				}

				return;
			}

			calQHash(q);

			int num_candidates = 0;
			uint32_t diff = 1;
			int lpos[4];
			int rpos[4];
			uint16_t lval[4], rval[4];
			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				//rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
				rpos[i] = std::distance(table.begin(), std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])));
				lpos[i] = rpos[i] - 1;
				while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
					lpos[i]--;
				}
				num_candidates += rpos[i] - lpos[i] - 1;
			}

			while (num_candidates < ub) {
				num_candidates = 0;
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + np * L];
					lval[i] = q->srpval[i] / diff * diff;
					rval[i] = lval[i] + diff;
					while (lpos[i] >= 0 && table[lpos[i]].val >= lval[i]) {
						lpos[i]--;
					}

					while (rpos[i] < size && table[rpos[i]].val <= rval[i]) {
						rpos[i]++;
					}
					num_candidates += rpos[i] - lpos[i] - 1;
				}
				diff *= 2;
			}

			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				for (int j = lpos[i] + 1;j < rpos[i];++j) {
					int u = part_map[np][table[j].id];
					if (visited.find(u) != visited.end()) continue;
					//visited[u] = true;
					visited.emplace(u);
					float dist = cal_inner_product(q->queryPoint, data[u], data.dim);
					q->top_candidates.emplace(u, -dist);
					ep.emplace(u, dist);
					cnt++;
					if (cnt > ub) break;
				}
			}

			q->cost += ub;
			while (q->top_candidates.size() > q->k) q->top_candidates.pop();
			while (ep.size() > q->k) ep.pop();
		}

		void knnFalse(queryN*& q) {
			int np = part_map.size() - 1;
			np = 0;
			int cnt = 0;
			int ub = 200;
			std::vector<bool>& visited = q->visited;

			int size = part_map[np].size();
			if (part_map[np].size() < ub) {
				for (auto& u : part_map[np]) {
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
				}

				return;
			}

			//calQHash(q);

			int num_candidates = 0;
			uint32_t diff = 1;
			int lpos[4];
			int rpos[4];
			uint16_t lval[4], rval[4];
			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				//rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
				rpos[i] = std::distance(table.begin(), std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])));
				lpos[i] = rpos[i] - 1;
				while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
					lpos[i]--;
				}
				num_candidates += rpos[i] - lpos[i] - 1;
			}

			while (num_candidates < ub) {
				num_candidates = 0;
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + np * L];
					lval[i] = q->srpval[i] / diff * diff;
					rval[i] = lval[i] + diff;
					while (lpos[i] >= 0 && table[lpos[i]].val >= lval[i]) {
						lpos[i]--;
					}

					while (rpos[i] < size && table[rpos[i]].val <= rval[i]) {
						rpos[i]++;
					}
					num_candidates += rpos[i] - lpos[i] - 1;
				}
				diff *= 2;
			}

			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				for (int j = lpos[i] + 1;j < rpos[i];++j) {
					int u = part_map[np][table[j].id];
					if (visited[u]) continue;
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
					cnt++;
					if (cnt > ub) break;
				}
			}

			q->cost += ub;
		}

		int getEntryPoint(queryN*& q) {
			int np = part_map.size() - 1;
			np = 0;
			//int cnt = 0;
			//int ub = 200;

			//int size = part_map[np].size();

			calQHash(q);
			auto& table = hash_tables[np * L];
			int pos = std::distance(table.begin(), std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[0])));
			if (pos > 0) return part_map[np][table[pos - 1].id];
			//if (pos - 1 >= 0) return part_map[np][table[pos - 1].id];
			return part_map[np][table[pos].id];

		}
	};
}
