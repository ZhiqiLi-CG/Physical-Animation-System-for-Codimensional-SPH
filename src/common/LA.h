#ifndef MATH_H
#define MATH_H
#include"Common.h"
#include "constant.h"
namespace ACG {
	/*
	template<int d>
	real __device__ Determinant(const Eigen::Matrix<real, d, d>& M) {
		real sum = 0;
		if constexpr (d == 6) {
			for (int i = 0; i < 720; i++) {
				real tem = 1;
				for (int j = 0; j < 6; j++) {
					tem *= M(j, permDevice6[i][j]);
				}
				tem *= permDevice6[i][6];
				sum += tem;
			}
		}
		else if constexpr (d == 5) {
			for (int i = 0; i < 120; i++) {
				real tem = 1;
				for (int j = 0; j < 5; j++) {
					tem *= M(j, permDevice5[i][j]);
				}
				tem *= permDevice5[i][5];
				sum += tem;
			}
		}
		return sum;
	}*/

}
#endif