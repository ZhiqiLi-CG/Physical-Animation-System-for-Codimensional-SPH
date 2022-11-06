#ifndef __PARTICLES__
#define __PARTICLES__
#include"Common.h"
#include"Algorithm.h"
#include "MLS.h"
namespace ACG{
	/// We need to launch the kernel function as much as possible,
	/// because launching the function need time
	template<int d,int side=HOST>
	class Particles {
		/// Note the procedure:
		///		M-->V by divide rho
		///		V-->h by SPH 
		/// In SPH, V is like m and h is like rho
		Typedef_VectorDii(d);
	/// Constant
	public:
		real t_dot = (real).2;										////threshold for the dot product between two normals
		real t_r = (real)1;											////local tangential radius, the supporting radius to search for the tangential neighbors, initialized based on _dx
		real v_r = (real)1;											////volumetric radius, the supporting radius to search for the volumetric neighbors, initialized based on _dx
		SPH* v_kernel;												////volumetric SPH kernel
		SPH* t_kernel;												////tangential SPH kernel
		real rho;
		/// Attribute
	public:
		Array<VectorD, side> x;
		Array<VectorD, side> v;
		Array<VectorD, side> f;
		Array<real, side> m;
		Array<real, side> h;
		Array<real, side> nden;
		Array<int, side> idx;
		Array<MatrixD, side> e;
		Array<MatrixT, side> g;
		Array<real, side> vol;
		Array<Array<int>, HOST> nbs;
		// here is the nbs transfered to GPU
		int**  nbs_ptr;
		int* nbs_ptr_num;
		Neighbor nb;
	/// Function for procedure
	public:
		void init(Araay<int>& Position) {
			v_kernel = SPH(v_r);
			t_kernel = SPH(t_r);
		}
		void update(real dt) {
			thrust::fill(f.begin(), f.end(), VectorD::Zero());
			updateLowercaseGamma();
			updatePressure();
			updateAdvectedHeight(dt);
			updateVorticity(dt);
			updateConcentration();

			updateExternalForce();
			updateVorticityForce();
			updatePressureForce();
			updateMarangoniForce(); 
			updateCapillaryForces();
			updateViscosityForces();
			updateVelocity(dt);
			updatePosition(dt);
			updateFrame();
			updateG();
			updateH();
		}
		void updateNb() {
			/// Here we need calculate nbs in CPU, then send it to GPU


		}

		void updateLowercaseGamma() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* GM_ptr = thrust::raw_pointer_cast(&GM[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				gm.begin(),
				[GM_ptr, this] __device__ __host__(const int idx)->real {
				return gamma_0 + gamma_a * GM_ptr[idx];
			}
			);
		}

		void updatePressure() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			MatrixD* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				p.begin(),
				[h_0, alpha_h, alpha_k, alpha_d, gm_ptr, x_ptr, e_ptr, g_ptr, Vol_ptr, v_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return alpha_h * (h_ptr[i] / h_0) + alpha_k * gm_ptr[idx] * calculateCurvature(idx, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr) +
					alpha_d * calculateCurvature(idx, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr);
			}
			);
		}


		void updateAdvectedHeight(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			MatrixD* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* ah_ptr = thrust::raw_pointer_cast(&ah[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				ah.begin(),
				[dt, x_ptr, e_ptr, g_ptr, Vol_ptr, v_ptr, h_ptr, ah_ptr, this] __device__ __host__(const int idx)->real {
				return ah_ptr[idx] - ah_ptr[idx] * calculateCurvature(idx, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr) * dt;
			}
			);
		}

		void updateVorticity(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			MatrixD* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* vo_ptr = thrust::raw_pointer_cast(&vo[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				vo.begin(),
				[dt, alpha_c, vo_ptr, x_ptr, e_ptr, g_ptr, Vol_ptr, v_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return alpha_c * dt * LapTang(
					idx,
					[vo_ptr] __device__ __host__(const int i, const int j)->real {
					return vo_ptr[j] - vo_ptr[i];
				},
					t_kernel, 0, nbs_num[idx], nbs_ptr[idx], x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr
					);
			}
			);
		}

		void updateConcentration() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			MatrixD* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* GM_ptr = thrust::raw_pointer_cast(&GM[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				GM.begin(),
				[dt, alpha_c, GM_ptr, x_ptr, e_ptr, g_ptr, Vol_ptr, v_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return alpha_c * dt * LapTang(
					idx,
					[GM_ptr] __device__ __host__(const int i, const int j)->real {
					return GM_ptr[j] - GM_ptr[i];
				},
					t_kernel, 0, nbs_num[idx], nbs_ptr[idx], x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr
					);
			}
			);
		}

		void updateExternalForce() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, f_ptr, e_ptr, h_ptr, p_a, Vol_0, m_ptr, gravity, this] __device__ __host__(const int idx)->real {
				real Vol_b = calculateVolume();
				real p_b = (Vol_0 / Vol_b) * p_0;
				return f_ptr[idx] + m_ptr[idx] * gravity + (p_0 - p_b) / (2 * h_ptr[idx]) * (e_ptr[idx]).col(d - 1);
			}
			);
		}

		void updateVorticityForce() {
			// TODO: waiting for interface
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, f_ptr, h_ptr, p_a, Vol_0, m_ptr, gravity] __device__ __host__(const int idx)->real {

			}
			);
		}

		void updatePressureForce() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* p_ptr = thrust::raw_pointer_cast(&p[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, f_ptr, vol_ptr, h_ptr, nbs_ptr_num, nbs_ptr, e_ptr, g_ptr, this] __device__ __host__(const int idx)->real {
				return f_ptr[idx] + 2 * vol_ptr[idx] *
					GradTang_Symmetric(
						idx,
						[p_ptr, h_ptr] __device__ __host__(const int i)->real {
					return p_ptr[i] / (h_ptr[i] * h_ptr[i]);
				},
						t_kernel, 0, nbs_ptr_num[i], nbs_ptr[i], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr
					);
			}
			);
		}

		void updateMarangoniForce() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, e_ptr, g_ptr, f_ptr, vol_ptr, h_ptr, nbs_ptr_num, nbs_ptr, this] __device__ __host__(const int idx)->real {
				return f_ptr[idx] + (vol_ptr[idx] / h_ptr[idx]) *
					GradTang_Difference(
						idx,
						[gm_ptr] __device__ __host__(const int i)->real {
					return gm[i];
				},
						t_kernel, 0, nbs_ptr_num[idx], nbs_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr
					);
			}
			);
		}

		void updateCapillaryForces() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, e_ptr, g_ptr, f_ptr, vol_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return f_ptr[idx] + (vol_ptr[idx] / h_ptr[idx]) *
					LapTang(
						idx,
						[x_ptr, e_ptr] __device__ __host__(const int i, const int j)->real {
					return -(x_ptr[i] - x_ptr[j]).dot(e_ptr[i].col(d - 1));
				},
						t_kernel, 0, nbs_num[idx], nbs_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr
					);
			}
			);
		}

		void updateViscosityForces() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			MatrixD* g_ptr = thrust::raw_pointer_cast(&g[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, e_ptr, g_ptr, f_ptr, vol_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return f_ptr[idx] + (vol_ptr[idx] * mu) *
					LapTang(
						idx,
						[x_ptr, v_ptr, e_ptr] __device__ __host__(const int i, const int j)->real {
					VectorD u_ij = v_ptr[j] - v_ptr[i];
					return u_ij - u_ij.dot(e_ptr[i].col(d - 1)) * e_ptr[i].col(d - 1);
				},
						t_kernel, 0, nbs_num[idx], nbs_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr
					);
			}
			);
		}

		void updateVelocity(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				v.begin(),
				[f_ptr, v_ptr, m_ptr, dt] __device__ __host__(const int idx)->real {
				return v_ptr[idx] + (f_ptr[idx] / m_ptr[idx]) * dt;
			}
			);
		}

		void updatePosition(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				x.begin(),
				[x_ptr, v_ptr, dt] __device__ __host__(const int idx)->real {
				return x_ptr[idx] + v_ptr[idx] * dt;
			}
			);
		}

		void updateFrame() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + e.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				e.begin(),
				[nbs_ptr, nbs_ptr_num,x_ptr,e_ptr, this] __device__ __host__ (const int idx)->MatrixD {
					return update_Frame(idx, nbs_ptr_num[idx], nbs_ptr[idx], x_ptr, e_ptr);
				}
			);
		}
		void updateG() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + h.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				g.begin(),
				[nbs_ptr, nbs_ptr_num, x_ptr, e_ptr, this] __device__ __host__(const int idx)->real {
					return Calculate_Tensor(idx, nbs_num[idx], nbs[idx], x, e);
				}
			);
		}
		void updateH() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + h.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			SPH* local_sph = t_kernel;
			int kernel_select=0; /// here need to be modified!
			thrust::transform(
				idxfirst,
				idxlast,
				h.begin(),
				[nbs_ptr, nbs_ptr_num, x_ptr, e_ptr, g_ptr, local_sph,kernel_select,this] __device__ __host__(const int idx)->real {
					return RealSPH_H(idx, local_sph, kernel_select, nbs_ptr_num[idx], nbs_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr);
				}
			);
		}
	/// Main function for SPH
	public:
		template<typename F>
		real __device__ __host__ RealSPH(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			real phi_real = 0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				VectorT rt_ij = coord[0] * e[i].cols(0) + coord[1] * e[i].cols(1);
				phi_real += Vol[j] / h[j] * phi[j]* sph->Weight<d - 1>(rt_ij.norm(), kernelID);
			}
			return phi_real;
		}
		real __device__ __host__ RealSPH_H(int i, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol) {
			real phi_real = 0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				VectorT rt_ij = coord[0] * e[i].cols(0) + coord[1] * e[i].cols(1);
				phi_real += Vol[j] * sph->Weight<d - 1>(rt_ij.norm(), kernelID);
			}
			return phi_real;
		}
		template<typename F>
		VectorT __device__ __host__ GradTang_Symmetric(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			VectorT grad_phi = VectorT::Zero();
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorT kernel_grad = Kernel_Grad_Tangential(i, j, sph, kernelID, x, e, g).norm();
				grad_phi += Vol[j] * h[i] *  kernel_grad*(phi(i)/h[i]/ h[i]+ phi(j) / h[j] / h[j]);
			}
			return grad_phi;
		}
		template<typename F>
		VectorT __device__ __host__  GradTang_Difference(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			VectorT grad_phi = VectorT::Zero();
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorT kernel_grad = Kernel_Grad_Tangential(i, j, sph, kernelID, x, e, g).norm();
				grad_phi += Vol[j] / h[j] *  kernel_grad * (phi(j) - phi(i));
			}
			return grad_phi;
		}
		template<typename F>
		real __device__ __host__  DivTang(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			real div_phi = (real)0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorT kernel_grad = Kernel_Grad_Tangential(i, j, sph, kernelID, x, e, g).norm();
				div_phi += Vol[j] / h[j] * kernel_grad.dot(phi(i,j));
			}
			return div_phi;
		}
		template<typename F>
		real __device__ __host__ LapTang(int i, const F& phi, SPH* sph, int kernelID,int nbs_num, 
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g,real* Vol, real* h) {
			real lap_phi = (real)0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				real kernel_lap = Kernel_Lap_Tangential(i, j, sph, kernelID, x, e, g).norm();
				lap_phi += Vol[j]/h[j]*phi(i,j)* kernel_lap;
			}
			return lap_phi;
		}
		VectorT __device__ __host__ Kernel_Grad_Tangential(int i, int j, SPH* sph,int kernelID, VectorD* x, MatrixD* e,MatrixT* g) {
			VectorD r_ij = x[i] - x[j];
			VectorT coord=projectPlane(r_ij, e[i]);
			VectorT rt_ij = coord[0] * e[i].cols(0) + coord[1] * e[i].cols(1);
			return (g[i].inverse())* sph->Grad<d - 1>(rt_ij.norm(), kernelID)*rt_ij.normalised();
		}
		real __device__ __host__ Kernel_Lap_Tangential(int i, int j, SPH* sph, int kernelID, VectorD* x, MatrixD* e, MatrixT* g) {
			VectorD r_ij = x[i] - x[j];
			VectorT coord = projectPlane(r_ij, e[i]);
			VectorT rt_ij = coord[0] * e[i].cols(0) + coord[1] * e[i].cols(1);
			VectorT grad_kerenel=(g[i].inverse()) * sph->Grad<d - 1>(rt_ij.norm(), kernelID) * rt_ij.normalised();
			return 2*grad_kerenel.norm() / rt_ij.norm();
		}
	/// Helper function
	public:
	/// 1. project to local frame
		VectorT __host__ __device__ projectPlane(const VectorD& u, const MatrixD& e) {
			VectorT t;
			for (int i = 0; i < d - 1; i++) {
				t[i] = u.dot(e.col(i));
			}
			return t;
		}
	/// 2. check tang --- only host
		bool __host__ isTangNeighbor(const VectorD& pos, const MatrixD& e_local, const int idx) const
		{
			////check angle
			VectorD n= e_local.col(d - 1);
			VectorD n_p = e[idx].col(d - 1);
			real angle = n.dot(n_p);
			if (angle < t_dot) return false;	////skip the points with large angles
			////check distance
			VectorD u = x[idx] - pos;
			VectorT t = projectPlane(u, e_local);
			return t.norm() < t_r;
		}
	/// 3. define tang neigbor --- only host
		Array<int> __host__ findTangNeighbor(const VectorD& pos, const MatrixD& e) const
		{
			Array<int> nbs,ans;
			nb.searchNeibor(pos, nbs);
			for (int i = 0; i < nbs.size(); i++) {
				if (isTangNeighbor(pos, e, nbs[i])) ans.push_back(nbs[i]);
			}
			return ans;
		}
	/// 4. function to find normal by PCA
		VectorD __host__ __device__ Normal(const VectorD& pos) const
		{
			int closest_p = nb.closestPoint(pos);
			Array<int> nbs = findTangNeighbor(pos, E(closest_p));
			if (nbs.size() == 0) {
				return Normal(closest_p);
			}
			VectorD nml = VectorD::Zero();
			for (int i = 0; i < nbs.size(); i++) {
				int p = nbs[i];
				real dis = (pos - X(p)).norm();
				real w0 = WPCA(dis);
				nml += w0 * Normal(p);
			}
			return nml.normalized();
		}
		VectorD __host__ __device__ Normal(int i, MatrixD* e) const { return e[i].cols(d - 1); }
		VectorD __host__ init_Normal_All(const Array<MatrixD,HOST>& e) {
			this->e = e;
		}
		MatrixD __host__ __device__ init_Frame(int idx,int nbs_num,int* nbs,VectorD* x,MatrixD* e) {
			/// find the neighbors must be called before this.
			VectorD x_tidle = VectorD::Zero();
			real w_total=0;
			MatrixD Cp = MatrixD::Zero();
			for (int i = 0; i < nbs_num; i++) {
				real d = (x[nbs[i]] - x[idx]).norm();
				real wij = WPCA(d);
				x_tidle += WPCA(d) * x[nbs[i]];
				w_total += wij;
			}
			if (w_total == 0) { printf("Init normal error! invalid w_total"); exit(1); }
			x_tidle /= w_total;
			for (int i = 0; i < nbs_num; i++) {
				real d = (x[nbs[i]] - x[idx]).norm();
				real wij = WPCA(d);
				Cp += wij * (x[nbs[i]] - x_tidle) * (x[nbs[i]] - x_tidle).transpose();
			}Cp /= w_total;
			/// Note that: the normal cannot change dramastically!
			/// the direction for new normal must be close to the old one
			VectorD normal = Min_Eigenvector(Cp);
			MatrixD new_e;
			if ((e[idx]->col(d - 1)).dot(normal) < 0) normal = -normal;
			if constexpr (d == 2) {
				new_e.col(0) = Orthogonal_Vector(normal).normalized();
				new_e.col(1) = normal.normalized();
				return new_e;
			}
			else if constexpr (d == 3) {
				new_e.col(0) = Orthogonal_Vector(normal).normalized();
				new_e.col(2) = normal.normalized();
				new_e.col(1) = (e[index].col(0)).cross(e[index].col(2));
				return new_e;
			}
		}
		MatrixD __host__ __device__ update_Frame(int i, int nbs_num, int* nbs, VectorD* x, MatrixD* e){
			/// TODO, which may not be implemented and be substitute by init_normal
			init_Frame(i,nbs_num,nbs,x,e);
		}
		real __host__ __device__ WPCA(const real r) const
		{
			if (r < v_r)return (real)1 - pow(r / v_r, 3);
			else return (real)0;
		}
	/// 5. approxmate local shape  and calculate metrix tensor
		template<typename F>
		VectorX __host__ __device__ Fit_Local_MLS(int idx,F phi, int nbs_num, int* nbs, VectorD* x, MatrixD* e) {
			real* data = new real[(d-1)*nbs_num];
			real* f = new real[nbs_num];
			for (int i = 0; i < nbs_num; i++) {
				int nb = nbs[i];
				VectorT tang= projectPlane(x[nb] - x[idx],e[nb])；
				if constexpr (d == 2) {
					data[i] = tang[0];
				}
				else if constexpr(d == 3) {
					data[2*i] = tang[0];
					data[2 * i+1] = tang[1];
				}
				f[i] = phi(nb);
			}
			return MLS::Fit(data, fit, nbs_num, VectorD::zero());
		}
		VectorX __host__ __device__ Fit_Local_Shape(int idx, int nbs_num, int* nbs, VectorD* x, MatrixD* e) {
			auto phi = [this,e,x,idx](int j) {
				return (e[j].col(d-1)).dot(x[j] - x[idx]);
			};
			return Fit_Local_MLS(idx, phi, nbs_num, nbs, x, e);

		}
		MatrixT __host__ __device__ Calculate_Tensor(int idx, int nbs_num, int* nbs, VectorD* x, MatrixD* e) {
			VectorX c = Fit_Local_Shape(idx, nbs_num, nbs, x, e);
			MatrixT metrix_tensor;
			if constexpr (d==2) metrix_tensor<<1+c[1]* c[1]；
			else if constexpr (d == 3) {
				metrix_tensor << 1 + c[1] * c[1], c[2] * c[1], c[2] * c[1], 1 + c[2] * c[2];
			}
			return metrix_tensor;
		}
	/// 5. Vol
		real Vol(const int idx) const { return M(idx) / pho; }
		real 
		//real Vol(const int idx) const { return (real)1 / nden[idx]; }
	};
}
#endif