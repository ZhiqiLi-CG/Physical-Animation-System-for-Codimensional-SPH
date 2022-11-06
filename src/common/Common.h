#ifndef COMMON_H
#define COMMON_H
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector_functions.h>
#include "driver_types.h"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Geometry"
#include <vector>
#include <list>
#include <queue>
#include <array>
#include <memory>
#include <iostream>
#include <cmath>
#include <iostream>

#include <boost/filesystem.hpp>
namespace ACG {
	int frame_number = 60;
	bool debug_x = false, debug_v = false, debug_h = false, debug_visc_f = false;
	bool debug_update = true;
#define checkUpdateReal(func,var,promp,force) \
		Array<real> New##var=var;\
		if(debug_update||force){\
		printf(promp);\
		printf("--------------------------------------------\n");\
		for (int i = 0; i < New##var.size(); i++) {\
			printf("%.12f ", New##var[i]);\
		}printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");}\
		func;\
		New##var = var;\
		if(debug_update||force){for (int i = 0; i < New##var.size(); i++) {\
			printf("%.12f ", New##var[i]);\
		}printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");\
		printf("--------------------------------------------\n\n");}
#define checkUpdateReal2(func,var,promp,force) \
		New##var=var;\
		if(debug_update||force){\
		printf(promp);\
		printf("--------------------------------------------\n");\
		for (int i = 0; i < New##var.size(); i++) {\
			printf("%.12f ", New##var[i]);\
		}printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");}\
		func;\
		New##var = var;\
		if(debug_update||force){for (int i = 0; i < New##var.size(); i++) {\
			printf("%.12f ", New##var[i]);\
		}printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");\
		printf("--------------------------------------------\n\n");}
#define checkUpdateThisReal(func,var,promp) \
		Array<real> New##var=this->var;\
		if(debug_update){\
		printf(promp);\
		printf("--------------------------------------------\n");\
		for (int i = 0; i < New##var.size(); i++) {\
			printf("%.12f ", New##var[i]);\
		}printf("\n");}\
		func;\
		New##var = this->var;\
		if(debug_update){for (int i = 0; i < New##var.size(); i++) {\
			printf("%.12f ", New##var[i]);\
		}printf("\n");\
		printf("--------------------------------------------\n\n");}
#define checkUpdateVector(func,var,promp,force) \
		Array<Vectord<3>> New##var=var;\
		if(debug_update||force){printf(promp);\
printf(promp);\
		printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");\
		for (int i = 0; i < New##var.size(); i++) {\
			printf("(%.12f,%.12f,%.12f) ", New##var[i][0],New##var[i][1],New##var[i][2]);\
		}printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");}\
		func;\
		New##var = var;\
		if(debug_update||force){for (int i = 0; i < New##var.size(); i++) {\
			printf("(%.12f,%.12f,%.12f) ", New##var[i][0],New##var[i][1],New##var[i][2]);\
		}printf("\n\n\n\n");\
		printf("--------------------------------------------\n\n");}
#define checkUpdateMatrix(func,var,promp) \
		Array<Matrixd<3>> New##var=var;\
		if(debug_update){printf(promp);\
		printf("--------------------------------------------\n");\
		for (int i = 0; i < New##var.size(); i++) {\
			printf("(%.12f,%.12f,%.12f;%.12f,%.12f,%.12f;%.12f,%.12f,%.12f)\t", New##var[i](0,0),New##var[i](0,1),New##var[i](0,2),New##var[i](1,0),New##var[i](1,1),New##var[i](1,2),New##var[i](2,0),New##var[i](2,1),New##var[i](2,2));\
		}printf("\n");}\
		func;\
		New##var = var;\
		if(debug_update){for (int i = 0; i < New##var.size(); i++) {\
			printf("(%.12f,%.12f,%.12f;%.12f,%.12f,%.12f;%.12f,%.12f,%.12f)\t", New##var[i](0,0),New##var[i](0,1),New##var[i](0,2),New##var[i](1,0),New##var[i](1,1),New##var[i](1,2),New##var[i](2,0),New##var[i](2,1),New##var[i](2,2));\
		}printf("\n");\
		printf("--------------------------------------------\n\n");}
#define checkUpdateVector2(func,var,promp,force) \
		New##var=var;\
		if(debug_update||force){printf(promp);\
printf(promp);\
		printf("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n");\
		for (int i = 0; i < New##var.size(); i++) {\
			printf("(%.12f,%.12f,%.12f) ", New##var[i][0],New##var[i][1],New##var[i][2]);\
		}printf("\n\n\n\n");}\
		func;\
		New##var = var;\
		if(debug_update||force){for (int i = 0; i < New##var.size(); i++) {\
			printf("(%.12f,%.12f,%.12f) ", New##var[i][0],New##var[i][1],New##var[i][2]);\
		}printf("\n\n\n\n");\
		printf("--------------------------------------------\n\n");}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
		}
	}
	#define InitSizeNum(array1,size)\
		array1.resize(size);
	#define InitSizeArray(array1,array2)\
		array1.resize(array2.size());
	#define Assert(boolVar,str) \
		if(!(boolVar)){\
			printf("----------------Aseert failed!---------------------\n");\
			std::cout << str << std::endl;\
			exit(1);\
		}
	#define checkCuda(boolVal,str)\
		if ((boolVal)) {\
			printf("----------------CUDA ERROR--------------------------\n");\
			std::cout << str << std::endl; \
			exit(1);\
		}
	//#define printMatrix()
	#define HOST 0
	#define DEVICE 1
	template<class T>
	using IOArray = std::vector<T>;
	template<class T, int side = 0>
	using Array = typename std::conditional<side == 0, thrust::host_vector<T>, thrust::device_vector<T>>::type;
	template<class T, int side = 0>
	using ArrayIter = typename Array<T, side>::iterator;
	using real = float;
	template<int d>
	using Vectord= typename Eigen::Matrix<real, d, 1>;
	template<int d>
	using Vectori = typename Eigen::Matrix<int, d, 1>;
	template<int d>
	using Matrixd = typename Eigen::Matrix<real, d, d>;
	using VectorX = Eigen::Matrix<real, Eigen::Dynamic, 1>; //Eigen::VectorXd;
	using MatrixX = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
	#define Typedef_VectorDii(d) \
		using VectorD=Vectord<d>; \
		using VectorDi=Vectori<d>; \
		using VectorT=Vectord<d-1>; \
		using VectorTi = Vectori<d-1>; \
		using MatrixD = Matrixd<d>; \
		using MatrixT = Matrixd<d-1>;
	template<int d,int side>
	class Frame {
	public:
		Typedef_VectorDii(d);
		Array<VectorD>points;
		Array<VectorD>normal;
		Array<real>h;
		Frame() {}
		Frame(Array<VectorD,side>&points, Array<VectorD, side>& normal, Array<real, side>& h) {
			this->points = points;
			this->normal = normal;
			this->h = h;
		}
		Frame(Array<VectorD, side>& points, Array<MatrixD, side>& e, Array<real, side>& h) {
			try {
				//thrust::copy(points.begin(), points.end(), this->points.begin());
				this->points = points;
			}
			catch (thrust::system_error& ee) {
				std::cerr << "CUDA error after cudaSetDevice1: " << ee.what() << std::endl;
			}Array<MatrixD> local_e;
			try {
				//thrust::copy(e.begin(), e.end(), local_e);
				local_e = e;
			}
			catch (thrust::system_error& ee) {
				std::cerr << "CUDA error after cudaSetDevice2: " << ee.what() << std::endl;
			}
			
			for (int i = 0; i < local_e.size(); i++) {
				normal.push_back(local_e[i].col(d - 1));
			}
			this->h = h;
			//thrust::copy(h.begin(), h.end(), this->h);
		}
	};
}
#endif