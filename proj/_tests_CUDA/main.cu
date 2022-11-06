#include <iostream>
#ifndef __Main_cpp__
#define __Main_cpp__
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "Common.h"
#include "GeometryInit.h"
/*
template<class T,int d1, int d2>
class MyMatrix {
public:
	T x00, x01, x02;
	T x10, x11, x12;
	T x20, x21, x22;
	static MyMatrix<T, d1, d2> __device__ __host__ GetMatrix(T x00) { 
		MyMatrix<T, d1, d2> tem; 
		tem.x00 = x00; 
		return tem; 
	}
	static MyMatrix<T, d1, d2> __device__ __host__ GetMatrix(T x00, T x10) {
		MyMatrix<T, d1, d2> tem; 
		tem.x00 = x00; tem.x10 = x10; 
		return tem; 
	}
	static MyMatrix<T, d1, d2> __device__ __host__ GetMatrix(T x00, T x10, T x20) {
		MyMatrix<T, d1, d2> tem; 
		tem.x00 = x00; tem.x10 = x10; tem.x20 = x20; 
		return tem;
	}
	static MyMatrix<T, d1, d2> __device__ __host__ GetMatrix(T x00, T x01, T x10, T x11) {
		MyMatrix<T, d1, d2> tem; 
		tem.x00 = x00; tem.x01 = x01; 
		tem.x10 = x10; tem.x11 = x11; 
		return tem;
	}
	static MyMatrix<T, d1, d2> __device__ __host__ GetMatrix(T x00, T x01, T x02, T x10, T x11, T x12, T x20, T x21, T x22) {
		MyMatrix<T, d1, d2> tem;  
		tem.x00 = x00; tem.x01 = x01; tem.x02 = x02;
		tem.x10 = x10; tem.x11 = x11; tem.x12 = x12;
		tem.x20 = x20; tem.x21 = x21; tem.x22 = x22;
		return tem;
	}
	MyMatrix<T,d1,1> __device__ __host__ cols(int i) {
		Assert(d1 == d2, "cols only for square matrix");
		if constexpr (d1 == 3) {
			Assert(i >= 0 && i <= 2, "Wrong dimension for cols in Matrix");
			if (i == 0) return MyMatrix<d1, 1>::GetMatrix(x00, x10, x20);
			else if (i == 1) return MyMatrix<d1, 1>::GetMatrix(x01, x11, x21);
			else if (i == 2) return MyMatrix<d1, 1>::GetMatrix(x02, x12, x22);
		}
		else if constexpr (d1 == 2) {
			Assert(i >= 0 && i <= 1, "Wrong dimension for cols in Matrix");
			if (i == 0) return MyMatrix<d1, 1>::GetMatrix(x00, x10);
			else if (i == 1) return MyMatrix<d1, 1>::GetMatrix(x01, x11);
		}
		else if constexpr (d1 == 1) {
			Assert(i >= 0 && i <= 0, "Wrong dimension for cols in Matrix");
			return MyMatrix<d1, 1>::GetMatrix(x00);
		}
	}
	T& __device__ __host__ operator() (int i) {
		Assert(d1 >= 1 && d2 == 1, "operator (int) only for vector");
		Assert(i < d1&& i >= 0, "index in operator(int) out of range");
		if (i == 0) return x00;
		else if (i == 1) return x10;
		else if (i == 2) return x20;
	}
	T& __device__ __host__ operator() (int i,int j) {
		Assert(d1 == d2 , "operator (int,int) only for matrix");
		Assert(i < d1&& i >= 0, "index in operator(int) out of range");
		if (i == 0) {
			if (j == 0) return x00;
			else if (j == 1) return x01;
			else if (j == 2) return x02;
		}
		else if (i == 1) {
			if (j == 0) return x10;
			else if (j == 1) return x11;
			else if (j == 2) return x12;
		}
		else if (i == 2) {
			if (j == 0) return x20;
			else if (j == 1) return x21;
			else if (j == 2) return x22;
		}
	}
	MyMatrix<T, d1, d2>& __device__ __host__ operator + (MyMatrix<T, d1, d2>& _one) {
		return MyMatrix<T, d1, d2>::GetMatrix(
			x00 + _one.x00, x01 + _one.x01, x02 + _one.x02,
			x10 + _one.x10, x11 + _one.x11, x12 + _one.x12,
			x20 + _one.x20, x21 + _one.x21, x22 + _one.x22);
	}
	MyMatrix<T, d1, d2>& __device__ __host__ operator - (MyMatrix<T, d1, d2>& _one) {
		return MyMatrix<T, d1, d2>::GetMatrix(
			x00 - _one.x00, x01 - _one.x01, x02 - _one.x02,
			x10 - _one.x10, x11 - _one.x11, x12 - _one.x12,
			x20 - _one.x20, x21 - _one.x21, x22 - _one.x22);
	}

	//TODO :transpose
	//TODO :inverse
	//TODO :dot
	//TODO :cross
	//TODO :mul
	//TODO :mul by scalar
	//TOFIX : add ASSERT
};
template<class T,int d1,int d2>
std::ostream& __device__ __host__ operator<<(std::ostream& out, MyMatrix<T, d1, d2>& A) {
	if (d1 == 1 && d2 == 1) {
		out << "Matrix 1*1" << std::endl;
		out << A.x00 << std::endl;
	}
	else if (d1 == 2 && d2 == 2) {
		out << "Matrix 2*2" << std::endl;
		out << A.x00 <<","<< A.x01 << std::endl;
		out << A.x10 << "," << A.x11 << std::endl;
	}
	else if (d1 == 3 && d2 == 3) {
		out << "Matrix 3*3" << std::endl;
		out << A.x00 << "," << A.x01<< "," << A.x02 << std::endl;
		out << A.x10 << "," << A.x11 << "," << A.x12 << std::endl;
		out << A.x20 << "," << A.x21 << "," << A.x22 << std::endl;
	}
	else if (d1 == 2 && d2 == 1) {
		out << "Matrix 2*1" << std::endl;
		out << A.x00 << std::endl;
		out << A.x10 << std::endl;
	}
	else if (d1 == 3 && d2 == 1) {
		out << "Matrix 3*1" << std::endl;
		out << A.x00 <<  std::endl;
		out << A.x10 << std::endl;
		out << A.x20 <<  std::endl;
	}
	return out;
}*/
class AAA {
public:
	int x;
	 void __device__ __host__ f() {
		printf("this is A\n");
	}
};
class BBB:public AAA {
public:
	void __device__ __host__ f() {
		printf("this is B\n");
	}
};
class CCC :public AAA {
public:
	 void __device__ __host__  f() {
		printf("this is C\n");
	}
};
class DDD {
public:
	void __host__ f() {
		thrust::device_vector<int> a(10);
		thrust::counting_iterator<int> idxfirst(0);
		thrust::counting_iterator<int> idxlast = idxfirst + a.size();
		//printf("%d %p-=+= \n", test_real.size(), thrust::raw_pointer_cast(&test_real[0]));
		auto g_f = [] __device__ __host__(const int idx)->int {
			return 0;// MatrixT();// MatrixT::Zero();// Calculate_Tensor(idx, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr);
		};
		thrust::transform(
			idxfirst,
			idxlast,
			a.begin(),
			g_f
		);

	}
};
class vec {
public:
	thrust::device_vector<double> a;
	double x = 1.4;
	vec() {
		for (int i = 0; i < 10; i++)
			a.push_back(10);
	}
	void f() {
		thrust::counting_iterator<int> idxfirst(0);
		thrust::counting_iterator<int> idxlast = idxfirst + a.size();
		thrust::transform(
			idxfirst,
			idxlast,
			a.begin(),
			[x=this->x]__device__(int index) {
				return x;
			}
		);
	}
	void g() {
		for (int i = 0; i < 10; i++) {
			int x = a[i];
			printf("%d  ", x);
		}
			

	}
};
int main(int argc,char* argv[])
{
	Typedef_VectorDii(3)
	InitSphere sphere(1, 0.1);
	Particles<3, DEVICE> particles;
	particles.initParameter(std::unordered_map<std::string, real>());
	particles.initAttribute(
		sphere.points, sphere.v, sphere.m,
		sphere.h, sphere.normals, sphere.Vol,
		sphere.GM, sphere.vo, sphere.maxPosition,
		sphere.minPosition
	);
	particles.updateG();

	DDD d;
	d.f();
	vec tem;
	tem.g();
	tem.f();
	tem.g();
	using namespace ACG;
	Typedef_VectorDii(3);
	thrust::device_vector<BBB> BBBB(10000);
	thrust::counting_iterator<int> idxfirst(0);
	thrust::counting_iterator<int> idxlast = idxfirst + BBBB.size();
	BBB* BBBB_ptr = thrust::raw_pointer_cast(&BBBB[0]);
	thrust::transform(
		idxfirst,
		idxlast,
		BBBB.begin(),
		[BBBB_ptr]__device__ __host__(int index) {
			printf("****\n");
			BBBB_ptr[index].f();
			return BBB();
		}
	);
	
	/*VectorDi aaa(1, 2);
	std::cout << aaa << std::endl;
	thrust::host_vector<thrust::device_vector<real>> aaaaa;
	thrust::device_vector<Array<MatrixD,DEVICE>> b;
	thrust::device_vector<MyMatrix<int, 3, 3>> b;
	thrust::counting_iterator<int> idxfirst(0);
	thrust::counting_iterator<int> idxlast = idxfirst + b.size();
	b.push_back(MyMatrix<int, 3, 3>::GetMatrix(1, 2, 3, 4, 521, 63, 745, 86, 94));
	b.push_back(MyMatrix<int, 3, 3>::GetMatrix(6, 5, 7, 1, 512, 61, 75, 85, 921));
	b.push_back(MyMatrix<int, 3, 3>::GetMatrix(2, 0, 4, 40, 5234, 654, 73, 82, 94));
	b.push_back(MyMatrix<int, 3, 3>::GetMatrix(3, 2, 8, 413, 51, 643, 756, 82, 91));
	MyMatrix<int, 3, 3>* b_ptr= thrust::raw_pointer_cast(&b[0]);
	*//*thrust::transform(
		idxfirst,
		idxlast,
		b.begin(),
		[b_ptr](int idx) {
			MyMatrix<int, 3, 3> tem1 = MyMatrix<int, 3, 3>::GetMatrix(51, 12, 33, 46, 5271, 263, 7145, 386, 694);
			MyMatrix<int, 3, 3> tem2 = MyMatrix<int, 3, 3>::GetMatrix(1, 1, 1, 1, 1, 1, 1, 1, 1);
			std::cout << idx << ":" << b_ptr[idx];
			std::cout << idx << ":" << tem1;
			std::cout << idx << ":" << tem2;
			MyMatrix<int, 3, 3> tem = tem1 + b_ptr[idx] + tem2;
			std::cout << idx << ":" << tem;
		}
	);
	*/
	Eigen::Matrix2d a;
	a << 1, 2, 3, 4;
	std::cout << a << std::endl;
}

#endif