#ifndef __Common_h__
#define __Common_h__
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <vector>
#include <list>
#include <queue>
#include <array>
#include <memory>
#include <iostream>
#include <cmath>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
namespace ACG {
	#define Assert(boolVar,str) \
		if(!boolVar){\
			printf("----------------Aseert failed!---------------------\n");\
			std::cout << str << std::endl;\
			exit(1);\
		}
	#define HOST 0
	#define DEVICE 1
	template<class T>
	using IOArray = std::vector<T>;
	template<class T, int side = 0>
	using Array = typename std::conditional<side == 0, thrust::host_vector<T>, thrust::device_vector<T>>::type;
	template<class T, int side = 0>
	using ArrayIter = typename Array<T, side>::iterator;
	using real = double;
	template<int d>
	using Vectord= typename Eigen::Matrix<real, d, 1>;
	template<int d>
	using Vectori = typename Eigen::Matrix<int, d, 1>;
	template<int d>
	using Matrixd = typename Eigen::Matrix<real, d, d>;
	using VectorX = Eigen::VectorXd;        
	using MatrixX = Eigen::MatrixXd;        
#define Typedef_VectorDii(d) using VectorD=Vectord<d>; using VectorDi=Vectori<d>; using VectorT=Vectord<d-1>; using VectorTi = Vectori<d-1>; using MatrixD = Matrixd<d>; using MatrixT = Matrixd<d-1>;

}


#endif