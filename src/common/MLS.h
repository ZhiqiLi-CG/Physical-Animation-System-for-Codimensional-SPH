#ifndef MLS_H
#define MLS_H
#include"Common.h"
#include"Algorithm.h"
#include "LA.h"
#include<unordered_map>
#include "constant.h"
namespace ACG {	
		template<int d>
		Eigen::Matrix<real, d - 1, d - 1> __device__ __host__ coMatrix(const Eigen::Matrix<real, d, d>& M, int row_idx, int col_idx) {
			Eigen::Matrix<real, d - 1, d - 1> coM;
			for (int i = 0; i < d; i++) {
				for (int j = 0; j < d; j++) {
					if (i != row_idx && j != col_idx) {
						int i_idx = i > row_idx ? i - 1 : i;
						int j_idx = j > col_idx ? j - 1 : j;
						coM(i_idx, j_idx) = M(i, j);
					}
				}
			}
			return coM;
		}
		template<int d>
		real __device__ __host__ Determinant(const Eigen::Matrix<real, d, d>& M) {
			if constexpr (d == 1) return M(0, 0);
			else if constexpr (d == 2) return M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
			else {
				real sum = 0;
				for (int i = 0; i < d; i++) {
					sum += Determinant<d - 1>(coMatrix<d>(M, 0, i)) * (i % 2 == 0 ? 1 : -1) * M(0, i);
				}
				return sum;
			}
		}
		template<int d>
		real __device__ __host__ coDet(const Eigen::Matrix<real, d, d>& M, int row_idx, int col_idx) {
			Eigen::Matrix<real, d - 1, d - 1> coM = coMatrix<d>(M, row_idx, col_idx);
			real det = Determinant<d - 1>(coM);
			return det * ((row_idx + col_idx) % 2 == 0 ? 1 : -1);
		}
		template<int d>
		Eigen::Matrix<real, d, d> __device__ __host__ Inverse(const Eigen::Matrix<real, d, d>& M) {
			real det = Determinant<d>(M);
			Eigen::Matrix<real, d, d> ans;
			for (int i = 0; i < d; i++) {
				for (int j = 0; j < d; j++) {
					real co = coDet<d>(M, j, i);
					ans(i, j) = co / det;
				}
			}
			return ans;
		}
		real __device__ __host__ LSDot(int m, real* v1, real* v2) {
			real ans = 0;
			for (int i = 0; i < m; i++) ans += v1[i] * v2[i];
			return ans;
		}
		template<int d,int side>
		Vectord<3*(d-1)> __device__ __host__ myLS(int m, real* B, real* f) {
			Matrixd<3 * (d - 1)> Ma;
			Vectord<3 * (d - 1)> fa;
			for (int i = 0; i < 3 * (d - 1); i++){
				for (int j = 0; j < 3 * (d - 1); j++) {
					Ma(i, j) = LSDot(m,B+i*m, B+j * m);
				}
			}
			for (int i = 0; i < 3 * (d - 1); i++) {
				fa[i] = LSDot(m, B+i*m, f);
			}
			Matrixd<3 * (d - 1)> tem = Ma;
			Ma =Inverse<3 * (d - 1)>(Ma);
			Vectord<3 * (d - 1)> ans = Ma * fa;
			return ans;
		}
		real __device__ __host__ MLS_Weight(const real& x0, const real& y0, const real& x1, const real& y1)
		{
			real dis2 = SlowPower2(x0 - x1) + SlowPower2(y0 - y1);
			return exp(-dis2 / 2);
		}
	/// For MLS
	////For 3 dimension:f(x,y)=c0+c1*x+c2*y+c3*x^2+c4*y^2+c5*xy
	/// For 2 dimension:f(x,y)=c0+c1*x+c2*x^2	
		template<int d, int side>
		Vectord<3*(d-1)> __host__ __device__ MLS_Fit(real* cood, real* fj, int m, Eigen::Matrix<real, d - 1, 1> origin) {
			real* B;
			real* f;
			f = new real[m];
			B= new real[m*3*(d-1)];
			for (int i = 0; i < m; i++) {
				real w = MLS_Weight(cood[i * 2], cood[i * 2 + 1], origin[0], origin[1]);
				if constexpr (d == 3) {
					B[i] = w * (real)1;
					B[i+m] = w * cood[i * 2];
					B[i + m*2] = w * cood[i * 2 + 1];
					B[i + m*3] = w * SlowPower2(cood[i * 2]);
					B[i + m*4] = w * SlowPower2(cood[i * 2 + 1]);
					B[i + m*5] = w * cood[i * 2] * cood[i * 2 + 1];
				}
				else {
					B[i] = w * (real)1;
					B[i+m] = w * cood[i];
					B[i + m*2] = w * SlowPower2(cood[i]);
				}
				f[i] = w * fj[i];
			}
			Vectord<3 * (d - 1)> ans;
			ans = myLS<d, side>(m,B, f);
			delete[] f;
			delete[] B;
			//printf("here is the mls_fit get ans %f %f %f %f %f %f\n", ans[0], ans[1], ans[2], ans[3], ans[4], ans[5]);
			return ans;
		}
		/*
		Un-used function
		VectorX __device__ __host__ myLS(const MatrixX& B , const VectorX& f) {
			MatrixX t = B.transpose() * B;
			return (B.transpose() * B).inverse() * (B.transpose() * f);
		}
				template<int d, int side>
		*/
		template<int d, int side>
		VectorX __host__ __device__ MLS_Fit2(real* cood,real* fj,int m,Eigen::Matrix<real,d-1,1> origin) {
			MatrixX B(m, 3*(d-1));
			VectorX f(m);
				for (int i = 0; i < m; i++) {
					real w= MLS_Weight(cood[i * 2], cood[i * 2 + 1], origin[0], origin[1]);
					if constexpr (d == 3) {
						B.coeffRef(i, 0) = w * (real)1;
						B.coeffRef(i, 1) = w * cood[i * 2];
						B.coeffRef(i, 2) = w * cood[i * 2 + 1];
						B.coeffRef(i, 3) = w * SlowPower2(cood[i * 2]);
						B.coeffRef(i, 4) = w * SlowPower2(cood[i * 2 + 1]);
						B.coeffRef(i, 5) = w * cood[i * 2] * cood[i * 2 + 1];
					}
					else {
						B.coeffRef(i, 0) = w * (real)1;
						B.coeffRef(i, 1) = w * cood[i];
						B.coeffRef(i, 2) = w * SlowPower2(cood[i]);
					}
					f[i] = w * fj[i];
			}
				VectorX ans;
				ans = B.colPivHouseholderQr().solve(f);
				//printf("here is the mls_fit get ans %d %d,%f %f %f %f %f %f\n", ans.rows(),ans.cols(), ans[0], ans[1], ans[2], ans[3], ans[4], ans[5]);
				return ans;
		}
		
}
#endif