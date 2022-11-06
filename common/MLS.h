#ifndef __ALGORITHM__
#define __ALGORITHM__
#include"Common.h"
#include<unordered_map>
namespace ACG {	
	/// For MLS
	////For 3 dimension:f(x,y)=c0+c1*x+c2*y+c3*x^2+c4*y^2+c5*xy
	/// For 2 dimension:f(x,y)=c0+c1*x+c2*x^2
	template<int d,int side> 
	class MLS {
		Typedef_VectorDii(d);
		static VectorX __device__ __host__ Fit(real* cood,real* fj,int n,VectorT origin) {
			MatrixX B(m, 3*(d-1));
			VectorX f(m);
			
				for (int i = 0; i < m; i++) {
					real w= sqrt(Weight(cood[i * 2], cood[i * 2 + 1], origin[0], origin[1]));
					if constexpr (d == 3) {
						B.coeffRef(i, 0) = w * (real)1;
						B.coeffRef(i, 1) = w * data[i * 2];
						B.coeffRef(i, 2) = w * data[i * 2 + 1];
						B.coeffRef(i, 3) = w * Power2(data[i * 2]);
						B.coeffRef(i, 4) = w * Power2(data[i * 2 + 1]);
						B.coeffRef(i, 5) = w * data[i * 2] * data[i * 2 + 1];
					}
					else {
						B.coeffRef(i, 0) = w * (real)1;
						B.coeffRef(i, 1) = w * data[i];
						B.coeffRef(i, 3) = w * Power2(data[i]);
					}
					f[i] = w * fj[i];
				
			}
			return B.colPivHouseholderQr().solve(f);
		}
		static real __device__ __host__ Weight(const real& x0, const real& y0, const real& x1, const real& y1) const
		{
			real dis2 = Power2(x0 - x1) + Power2(y0 - y1);
			return exp(-dis2/2);
		}
	};

}

#endif