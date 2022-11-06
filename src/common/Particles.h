#ifndef PARTICLES_H
#define PARTICLES_H
#include"Common.h"
#include"Algorithm.h"
#include "MLS.h"
#include <omp.h>
#include "IO.h"
namespace ACG {
	template<int d, class T>
	class DDD {
	public:
		void __host__ f() {
			thrust::device_vector<T> a(2);
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + a.size();
			//printf("%d %p-=+= \n", test_real.size(), thrust::raw_pointer_cast(&test_real[0]));
			auto g_f = [] __device__ __host__(const int idx)->T {
				printf("????");
				return (T)0;// MatrixT();// MatrixT::Zero();// Calculate_Tensor(idx, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr);
			};
			thrust::transform(
				idxfirst,
				idxlast,
				a.begin(),
				g_f
			);

		}
	};
	/// We need to launch the kernel function as much as possible,
	/// because launching the function need time
	template<int d, int side = HOST>
	class Particles {
		/// Note the procedure:
		///		M-->V by divide rho
		///		V-->h by SPH 
		/// In SPH, V is like m and h is like rho
		Typedef_VectorDii(d);
		/// Constant
	public:
		real t_dot = (real).2;										////threshold for the dot product between two normals
		real t_r = (real)0.1;											////local tangential radius, the supporting radius to search for the tangential neighbors, initialized based on _dx
		real v_r = (real)0.1;											////volumetric radius, the supporting radius to search for the volumetric neighbors, initialized based on _dx
		//SPH* v_kernel;												////volumetric SPH kernel
		SPH* t_kernel;												////tangential SPH kernel
		real rho;
		/// Attribute
	public:
		Array<VectorD, side> x;										///Need init
		Array<VectorD, side> v;										///Need init
		Array<VectorD, side> f;										///internal result
		Array<real, side> m;										///Need init
		Array<real, side> h;										///Need init
		Array<MatrixD, side> e;										///Need init
		Array<MatrixT, side> g;										///init by method
		Array<real, side> vol;										///Need init
		Array<real, side> ah; // advected height					///init by h
		Array<real, side> gm; // surface tension coefficient		///internal result
		Array<real, side> GM;  // surfactant concentration			///Need init
		Array<real, side> p; // pressure							///internal result
		Array<real, side> vo; // vorticity							///Need init
		Array<real, side> s;
		/// some params
		real gravity = 9.81; // the gravitational acceleration
		real gamma_0 = 7.275e-2;
		real gamma_a = 8.3144598 * 298.15; // used in update surface tension coeef
		real alpha_h = 1e1; //unknown, used in updating pressure
		real alpha_k = 1e1; //unknown, used in updating pressure
		real alpha_d = 1e1; //unknown, used in updating pressure
		real alpha_c = 1e1; //unknown, used in updating concentration
		real h_0 = 6e-7; // rest thickness of the film 
		real p_0 = 10132.5; // the standard atmospheric pressure
		real mu = 8.9e-4; // used when calculate viscosity force
		real Vol_0 = 1e1; // the volume at the beginning
		real Vol_b = 1e1;
		Array<Array<int, side>> nbs;
		// here is the nbs transfered to GPU
		Array<int*,side> nbs_ptr;
		Array<int,side> nbs_num;
		Neighbor<d> nb;
		IO<d,side> io;
		/// Function for procedure
	public:
		void __host__ initParameter(std::unordered_map<std::string,real> assi_map,std::string outputFile="D:\\ACG\\output.txt") {
			std::vector<std::string> name{ "gravity","gamma_0","gamma_a","alpha_h","alpha_k",\
				"alpha_d","alpha_c","h_0","p_0","mu","Vol_0", "Vol_b","t_dot","t_r","v_r"};
			std::vector<real*> value{ &gravity,&gamma_0,&gamma_a,&alpha_h,&alpha_k,\
				& alpha_d, &alpha_c, &h_0, &p_0, &mu, &Vol_0, &Vol_b,& t_dot,&t_r,&v_r };
			for (int i = 0; i < name.size(); i++) {
				if (assi_map.find(name[i]) != assi_map.end()) {
					std::cout << "Init the const parameter:" << name[i] << std::endl;
					(* value[i]) = assi_map[name[i]];
				}
			}
			if constexpr (side == DEVICE) {
				//checkCuda(cudaMalloc((void**)&v_kernel, sizeof(SPH)));
				checkCuda(cudaMalloc((void**)&t_kernel, sizeof(SPH)),"Malloc for t_kernel in CUDA failed");
				//checkCuda(cudaMemcpy(v_kernel, new SPH(v_r), sizeof(SPH), cudaMemcpyHostToDevice));
				checkCuda(cudaMemcpy(t_kernel, new SPH(t_r), sizeof(SPH), cudaMemcpyHostToDevice), "Transfer for t_kernel in CUDA failed");
			}
			else {
				//v_kernel = new SPH(v_r);
				t_kernel = new SPH(t_r);
			}
			io.Init_Write(outputFile);
		}
		void __host__ initAttribute(Array<VectorD> x, Array<VectorD> v, Array<real> m, Array<real> h, Array<VectorD> normals,
			Array<real> vol, Array<real> GM, Array<real> vo,VectorD MaxPosition, VectorD MinPosition
		) {
			this->x = x;
			this->v = v;
			//for (int i = 0; i < this->x.size(); i++) {
			//	VectorD pos = this->x[i];
				//printf("---idx:%d %f,%f,%f\n", i, pos[0], pos[1], pos[2]);
			//}	
			this->h = h;
			this->ah = h;
			this->GM = GM;
			this->vo = vo;
			/// get the frame from the normal
			Array<MatrixD> local_e(normals.size());
			for (int i = 0; i < normals.size(); i++) {
				local_e[i] = Frame_From_Normal(normals[i]);
			}
			this->e = local_e;
			/// then calculate the nbs
			/// TOFIX: neighbor will use the frame last time to calculate tangential neighbor
			InitSizeArray(this->f, x)
			InitSizeArray(this->g, x)
			InitSizeArray(this->gm, x)
			InitSizeArray(this->p, x)
			InitSizeArray(this->nbs, x)
			InitSizeArray(this->vol, x)
			InitSizeArray(this->m, x)
			InitSizeArray(this->s, x)
				printf("update NB\n");
			updateNb(MaxPosition, MinPosition);
			printf("update NB end\n");
			// here need to init m and v
			checkUpdateThisReal(updateVol(),vol,"Init Vol")
			checkUpdateThisReal(updateM(), m, "Init Mass")
			//this->m = m;
			updateS();
			Vol_0=Vol_b = calculateVolume();
			//printf("The init Vol %f---------------------------------------------\n", Vol_0);
			updateG();
		}
		void __host__ update(real dt) {
			thrust::fill(f.begin(), f.end(), VectorD::Zero());
			
			//checkUpdateReal(updateH(), h, "test ori_h  original",true);
			/*checkUpdateReal(updateLowercaseGamma(), gm, "test gamma", false);
			checkUpdateReal(updatePressure(), p, "test pressure", true);
			checkUpdateReal(updateAdvectedHeight(dt), ah, "test ah",true);
			checkUpdateReal(updateVorticity(dt), vo, "test Vo", false);
			checkUpdateReal(updateConcentration(dt), GM, "test Con",false);
			checkUpdateVector(updateExternalForce(), f, "test externa force",false);
			checkUpdateVector2(updateVorticityForce(), f, "test VorticityForce",false);
			checkUpdateVector2(updatePressureForce(), f, "test PressureForce",false);
			checkUpdateVector2(updateMarangoniForce(), f, "test Maran Force", false);
			checkUpdateVector2(updateCapillaryForces(), f, "test CapillaryForces", false);
			checkUpdateVector2(updateViscosityForces(), f, "test ViscosityForces", false);
			checkUpdateVector(updateVelocity(dt), v, "test velocity", true);
			checkUpdateVector(updatePosition(dt), x, "test Position", false);
			updateNb();
			checkUpdateMatrix(updateFrame(),e,"test frame");*/
			updateS();
			//updateG();
			checkUpdateReal(updateH(),h, "test ori_h",false);
			Vol_b = calculateVolume();

			//printf("This is the ner Vol_b %f\n", Vol_b);

			

		}
		real __host__  calculateVolume() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + x.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			int dd = d;
			//printf("%d %d %d %d", vol.size(), h.size(), x.size(), e.size());
			Array<real, side> temp(x.size());
				thrust::transform(
					idxfirst,
					idxlast,
					temp.begin(),
					[vol_ptr, x_ptr, e_ptr, h_ptr, dd] __device__ __host__(const int idx)->real {
					return (1.0 / dd) * (vol_ptr[idx] / h_ptr[idx]) * x_ptr[idx].dot(e_ptr[idx].col(d - 1));
				}
				);
			return thrust::reduce(temp.begin(), temp.end());
		}
		void __host__ output_triangle(const Array<Eigen::Matrix<int, 3, 1>>& triangles) {
			io.outputTriangles(triangles);
		}
		void __host__ output() {
			io.outputOneFrame(Frame<d, side>(x, e, h));
		}
		void __host__ updateNb(VectorD MaxPosition, VectorD  MinPosition) {
			/// Here we need calculate nbs in CPU, then send it to GPU
			Array<MatrixD, HOST> e_host = e;
			Array<VectorD, HOST> x_host = x;
			nb.construct(x_host, t_r, MaxPosition, MinPosition);
			Array<int*> nbs_ptr_local(x_host.size());
			Array<int> nbs_num_local(x_host.size());
			for (int i = 0; i < x_host.size(); i++) {
				//if (i % 100 == 0) printf("find nb:%d\n", i);
				Array<int> tem;
				findTangNeighbor(i, e_host, x_host, tem);
				//for (int j = 0; j < tem.size(); j++)
				//printf("%d ", tem[j]);
				nbs[i] = tem;
				nbs_ptr_local[i] = thrust::raw_pointer_cast(&nbs[i][0]);
				nbs_num_local[i] = nbs[i].size();
				//printf("%d %d:\n", i, nbs[i].size());
				//for (int j = 0; j < nbs[i].size(); j++) {
				////	int x = nbs[i][j];
				//	printf("%d ", x);
				//}printf("\n");
			}
			nbs_ptr = nbs_ptr_local;
			nbs_num = nbs_num_local;
		}
		void __host__ updateNb() {
			/// Here we need calculate nbs in CPU, then send it to GPU
			Array<MatrixD, HOST> e_host = e;
			Array<VectorD, HOST> x_host = x;
			nb.update(x_host);
			Array<int*> nbs_ptr_local(x_host.size());
			Array<int> nbs_num_local(x_host.size());
			
			for (int i = 0; i < x_host.size(); i++) {
				//if (i % 100 == 0) printf("find nb:%d\n", i);
				Array<int> tem;
				findTangNeighbor(i, e_host, x_host, tem);
				nbs[i] = tem;
				nbs_ptr_local[i] = thrust::raw_pointer_cast(&nbs[i][0]);
				nbs_num_local[i] = nbs[i].size();
			}
			nbs_ptr = nbs_ptr_local;
			nbs_num = nbs_num_local;
		}
		void __host__ updateLowercaseGamma() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* GM_ptr = thrust::raw_pointer_cast(&GM[0]);
			/// why + ?
			real gamma_local_0 = gamma_0, gamma_local_a = gamma_a;
			thrust::transform(
				idxfirst,
				idxlast,
				gm.begin(),
				[GM_ptr, gamma_local_0, gamma_local_a] __device__ __host__(const int idx)->real {
					return gamma_local_0 - gamma_local_a * GM_ptr[idx];
				}
			);
		}
		void __host__ updatePressure() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			real* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			/// Two curvature is calculated?
			/// -1?
			real h_local_0 = h_0;
			real alpha_local_k = alpha_k;
			real alpha_local_d = alpha_d;
			real alpha_local_h = alpha_h;
			SPH* t_kernel_local = t_kernel;
			thrust::transform(
				idxfirst,
				idxlast,
				p.begin(),
				[h_local_0,  alpha_local_k, alpha_local_d, gm_ptr, x_ptr, e_ptr, 
				g_ptr, Vol_ptr, v_ptr, h_ptr, nbs_ptr_ptr, nbs_num_ptr, 
				this, alpha_local_h, t_kernel_local] __device__ __host__(const int idx)->real {
				real k = calculateCurvature(idx, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr, nbs_ptr_ptr, nbs_num_ptr, t_kernel_local);
				//printf("curvature:%f\t", k);
				return alpha_local_h * (h_ptr[idx] / h_local_0 - 1) + alpha_local_k * gm_ptr[idx] *k+
					alpha_local_d * calculateDiv(idx, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr, v_ptr,nbs_ptr_ptr, nbs_num_ptr, t_kernel_local);
				}
			);
		}
		void __host__ updateAdvectedHeight(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			real* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* ah_ptr = thrust::raw_pointer_cast(&ah[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			thrust::transform(
				idxfirst,
				idxlast,
				ah.begin(),
				[dt, x_ptr, e_ptr, g_ptr, Vol_ptr, v_ptr, h_ptr, ah_ptr, t_kernel_local, nbs_ptr_ptr, nbs_num_ptr, this] __device__ __host__(const int idx)->real {
				return ah_ptr[idx] - ah_ptr[idx] * 
					calculateDiv(idx, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr, v_ptr, nbs_ptr_ptr, nbs_num_ptr, t_kernel_local) * dt;
			}
			);
		}
		void __host__ updateVorticity(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			real* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* vo_ptr = thrust::raw_pointer_cast(&vo[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			int** nbs_ptr_ptr= thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr= thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			real alpha_c_local = alpha_c;
			auto vo_ij = [vo_ptr] __device__ __host__(const int i, const int j)->real {
				return vo_ptr[j] - vo_ptr[i];
			};
			thrust::transform(
				idxfirst,
				idxlast,
				vo.begin(),
				[dt, vo_ptr, alpha_c_local, x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr, nbs_ptr_ptr,
					nbs_num_ptr, t_kernel_local, vo_ij,this] __device__ __host__(const int idx)->real {
						real newVol= LapTangReal(idx, vo_ij,
							t_kernel_local, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr
						);
						return alpha_c_local * dt * newVol+ vo_ptr[idx];
				}
			);
		}
		void __host__ updateConcentration(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			real* Vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* GM_ptr = thrust::raw_pointer_cast(&GM[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			auto GM_ij = [GM_ptr] __device__ __host__(const int i, const int j)->real {
				return GM_ptr[j] - GM_ptr[i];
			};
			real alpha_c_local = alpha_c;
			thrust::transform(
				idxfirst,
				idxlast,
				GM.begin(),
				[dt, GM_ptr, GM_ij, alpha_c_local, x_ptr, e_ptr, g_ptr, Vol_ptr,
					h_ptr, nbs_ptr_ptr, nbs_num_ptr, t_kernel_local,this] __device__ __host__(const int idx)->real {
						return GM_ptr[idx]+alpha_c_local * dt * LapTangReal(idx,GM_ij,
								t_kernel_local, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr);
				}
			);
		}
		void __host__ updateExternalForce() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			real p_0_local = p_0;
			real Vol_0_local = Vol_0;
			real Vol_b_local = Vol_b;
			VectorD gravity_local=VectorD::Zero();
			real* s_ptr = thrust::raw_pointer_cast(&s[0]);;
			gravity_local[d-1]=-gravity;
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[s_ptr,f_ptr, e_ptr, h_ptr, p_0_local, Vol_0_local, Vol_b_local,
					m_ptr, gravity_local, this] __device__ __host__(const int idx)->VectorD {
				
						real p_b_local = (Vol_0_local / Vol_b_local) * p_0_local; //printf("what?:%.12f %.12f %.12f %.12f %.12f--%.12f\n", Vol_0_local, Vol_b_local, p_0_local, p_b_local, s_ptr[idx], (p_b_local - p_0_local) * s_ptr[idx]);
						VectorD newForce = f_ptr[idx] + m_ptr[idx] * gravity_local +(p_b_local - p_0_local) * s_ptr[idx] * (e_ptr[idx]).col(d - 1)/1000;
						// +(p_0_local - p_b_local) / (2 * h_ptr[idx]) * (e_ptr[idx]).col(d - 1);
						return newForce;
				}
			);
		}
		void __host__ updateVorticityForce() {
			// TODO: waiting for interface
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vo_ptr= thrust::raw_pointer_cast(&vo[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[x_ptr, f_ptr, nbs_num_ptr, nbs_ptr_ptr, e_ptr, vo_ptr, this] __device__ __host__(const int idx)->VectorD {
					for (int k = 0; k < nbs_num_ptr[idx]; k++) {
						int j = nbs_ptr_ptr[idx][k];
						VectorD rij = x_ptr[idx] - x_ptr[j];
						VectorD rt_ij = planeVector(projectPlane(rij,e_ptr[idx]), e_ptr[idx]);
						VectorD newForce= f_ptr[idx] - rt_ij.cross(vo_ptr[j] * e_ptr[j].col(d - 1));
						return newForce;
					}
				}
			);
		}
		void __host__ updatePressureForce() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* p_ptr = thrust::raw_pointer_cast(&p[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			auto p_f= [p_ptr] __device__ __host__(const int i)->real {
				return p_ptr[i];
			};
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[p_f,x_ptr, f_ptr, vol_ptr, h_ptr, nbs_num_ptr, 
					nbs_ptr_ptr, e_ptr, g_ptr, t_kernel_local, this] __device__ __host__(const int idx)->VectorD {
				VectorD newForce=f_ptr[idx] + 2 * vol_ptr[idx] *
					GradTang_Symmetric(idx,p_f,
						t_kernel_local, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr);
				return newForce;
			}
			);
		}
		void __host__ updateMarangoniForce() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			auto gm_f = [gm_ptr] __device__ __host__(const int i)->real {
				return gm_ptr[i];
			};
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[gm_f,x_ptr, e_ptr, g_ptr, f_ptr, vol_ptr, h_ptr, 
					nbs_num_ptr, nbs_ptr_ptr, t_kernel_local,this] __device__ __host__(const int idx)->VectorD {
				VectorD newForce= f_ptr[idx] + (vol_ptr[idx] / h_ptr[idx]) *
					GradTang_Difference(
						idx, gm_f, t_kernel_local, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr
					);
				return newForce;
				}
			);
		}
		void __host__ updateCapillaryForces() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			real* gm_ptr = thrust::raw_pointer_cast(&gm[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			auto xt_f = [x_ptr, e_ptr] __device__ __host__(const int i, const int j)->real {
				real value= -(x_ptr[i] - x_ptr[j]).dot(e_ptr[i].col(d - 1));
				return value;
			};
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[xt_f,x_ptr, e_ptr, g_ptr, f_ptr, vol_ptr, h_ptr, gm_ptr, 
					nbs_num_ptr, nbs_ptr_ptr, t_kernel_local, this] __device__ __host__(const int idx)->VectorD {
				real lap= LapTangReal(idx, xt_f, t_kernel_local, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr
				);
				//printf("This capilary:%.12f %.12f %.12f %.12f\n", lap, vol_ptr[idx], gm_ptr[idx],h_ptr[idx]);
				VectorD newForce = f_ptr[idx] + (vol_ptr[idx] * gm_ptr[idx] / h_ptr[idx]) * e_ptr[idx].col(d - 1) * lap;
					
				return newForce;
				}
			);
		}
		void __host__ updateViscosityForces() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* t_kernel_local = t_kernel;
			real local_mu = mu;
			auto uij_f = [ v_ptr, e_ptr] __device__ __host__(const int i, const int j)->VectorD {
				VectorD u_ij = v_ptr[j] - v_ptr[i];
				VectorD returnV= u_ij - u_ij.dot(e_ptr[i].col(d - 1)) * e_ptr[i].col(d - 1);
				return returnV;
			};
			thrust::transform(
				idxfirst,
				idxlast,
				f.begin(),
				[local_mu,uij_f,x_ptr, e_ptr, g_ptr, f_ptr, vol_ptr, h_ptr,
						nbs_num_ptr, nbs_ptr_ptr, t_kernel_local,this] __device__ __host__(const int idx)->VectorD {
							VectorD newForce=f_ptr[idx] + (vol_ptr[idx] * local_mu) *
								LapTangVec(idx,uij_f,t_kernel_local, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, h_ptr);
							return newForce;
				}
			);
		}
		void __host__ updateVelocity(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* f_ptr = thrust::raw_pointer_cast(&f[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			real* m_ptr = thrust::raw_pointer_cast(&m[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				v.begin(),
				[f_ptr, v_ptr, m_ptr, dt] __device__ __host__(const int idx)->VectorD {
				return v_ptr[idx] + (f_ptr[idx] / m_ptr[idx]) * dt;
			}
			);
		}
		void __host__ updatePosition(real dt) {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + gm.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			VectorD* v_ptr = thrust::raw_pointer_cast(&v[0]);
			thrust::transform(
				idxfirst,
				idxlast,
				x.begin(),
				[x_ptr, v_ptr, dt] __device__ __host__(const int idx)->VectorD {
				return x_ptr[idx] + v_ptr[idx] * dt;
			}
			);
		}
		void __host__ updateFrame() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + e.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			real v_r_local = v_r;
			thrust::transform(
				idxfirst,
				idxlast,
				e.begin(),
				[nbs_ptr_ptr, nbs_num_ptr, x_ptr, e_ptr, v_r_local, this] __device__ __host__(const int idx)->MatrixD {
				return update_Frame(idx, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr,v_r_local);
			}
			);
		}
		void __host__ updateG() {
			//printf("now begin in G\n");
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + h.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			auto g_f = [nbs_num_ptr, nbs_ptr_ptr, x_ptr, e_ptr,this] __device__ __host__(const int idx)->MatrixT {
				return Calculate_Tensor(idx, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr);
			};
			//try{
				thrust::transform(
					idxfirst,
					idxlast,
					g.begin(),
					g_f	
				);
			//}
			//catch (thrust::system_error& e){
			//	std::cerr << "CUDA error after cudaSetDevice--------------------: " << e.what() << std::endl;
			//}
		}
		void __host__ updateH() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + h.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			real* s_ptr = thrust::raw_pointer_cast(&s[0]);
			SPH* local_sph = t_kernel;
			int kernel_select = 1; /// here need to be modified!
			thrust::transform(
				idxfirst,
				idxlast,
				h.begin(),
				[s_ptr,nbs_ptr_ptr, nbs_num_ptr, x_ptr, e_ptr, g_ptr,
					local_sph, kernel_select, vol_ptr, this] __device__ __host__(const int idx)->real {
						return RealSPH_H(idx, local_sph, kernel_select, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr, s_ptr);
			}
			);
		}
		void __host__ updateVol() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + vol.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* local_sph = t_kernel;
			real* h_ptr= thrust::raw_pointer_cast(&h[0]);
			int kernel_select = 1; /// here need to be modified!
			thrust::transform(
				idxfirst,
				idxlast,
				vol.begin(),
				[nbs_ptr_ptr, nbs_num_ptr, x_ptr, e_ptr, g_ptr,
				local_sph, kernel_select, vol_ptr,h_ptr, this] __device__ __host__(const int idx)->real {
				return h_ptr[idx]*RealSPH_S(idx, local_sph, kernel_select, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr);
			}
			);
		}
		void __host__ updateS() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + vol.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* local_sph = t_kernel;
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			int kernel_select = 1; /// here need to be modified!
			thrust::transform(
				idxfirst,
				idxlast,
				s.begin(),
				[nbs_ptr_ptr, nbs_num_ptr, x_ptr, e_ptr, g_ptr,
				local_sph, kernel_select, vol_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return RealSPH_S(idx, local_sph, kernel_select, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, vol_ptr);
			}
			);
		}
		void __host__ updateM() {
			thrust::counting_iterator<int> idxfirst(0);
			thrust::counting_iterator<int> idxlast = idxfirst + m.size();
			VectorD* x_ptr = thrust::raw_pointer_cast(&x[0]);
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			real* vol_ptr = thrust::raw_pointer_cast(&vol[0]);
			MatrixT* g_ptr = thrust::raw_pointer_cast(&g[0]);
			int** nbs_ptr_ptr = thrust::raw_pointer_cast(&nbs_ptr[0]);
			int* nbs_num_ptr = thrust::raw_pointer_cast(&nbs_num[0]);
			SPH* local_sph = t_kernel;
			real* h_ptr = thrust::raw_pointer_cast(&h[0]);
			int kernel_select = 1; /// here need to be modified!
			thrust::transform(
				idxfirst,
				idxlast,
				m.begin(),
				[nbs_ptr_ptr, nbs_num_ptr, x_ptr, e_ptr, g_ptr,
				local_sph, kernel_select, vol_ptr, h_ptr, this] __device__ __host__(const int idx)->real {
				return vol_ptr[idx]*1e3;
			}
			);
		}
		static  real __host__ __device__ calculateCurvature(int idx, VectorD* x_ptr, MatrixD* e_ptr, MatrixT* g_ptr, real* Vol_ptr,
			real* h_ptr, int** nbs_ptr_ptr, int* nbs_num_ptr, SPH* t_kernel) {
			return LapTangReal(
				idx,
				[h_ptr] __device__ __host__(const int i, const int j)->real {
				return h_ptr[j] - h_ptr[i];
			},
				t_kernel, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx], x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr
				);
		}
		// here the v_ptr is pointer
		static  real __host__ __device__ calculateDiv(int idx, VectorD* x_ptr, MatrixD* e_ptr, MatrixT* g_ptr, real* Vol_ptr,
			real* h_ptr, VectorD* v_ptr, int** nbs_ptr_ptr, int* nbs_num_ptr, SPH* t_kernel) {
			//TOFIX:
			return DivTang(
				idx,
				[e_ptr, v_ptr, x_ptr] __device__ __host__(const int i, const int j)->VectorD {
				VectorD r_ij = x_ptr[i] - x_ptr[j];
				VectorD ei[2], ej[2];//
				ej[0] = ei[0] = r_ij.normalized();
				ei[1] = (e_ptr[i].col(d - 1)).cross(ei[0]);
				ej[1] = (e_ptr[j].col(d - 1)).cross(ej[0]);
				VectorD t = VectorD::Zero();
				for (int k = 0; k < d - 1; k++) {
					t += (ej[k].dot(v_ptr[j]) - ei[k].dot(v_ptr[j])) * ei[k];
				}
				return t;
			}, t_kernel, 0, nbs_num_ptr[idx], nbs_ptr_ptr[idx],
				x_ptr, e_ptr, g_ptr, Vol_ptr, h_ptr
				);
		}
			/// Main function for SPH
	public:
		template<typename F>
		static  real __device__ __host__ RealSPH(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h, real* s_ptr) {
			real phi_real = 0;
			real norm0 = 0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				//if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				norm0 += s_ptr[j] * sph->Weight<d - 1>(coord.norm(), kernelID);
			}
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				//if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				phi_real += Vol[j] / h[j] * phi[j] * sph->Weight<d - 1>(coord.norm(), kernelID);
			}
			return phi_real;// / norm0;
		}
		static  real __device__ __host__ RealSPH_H(int i, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol,real* s_ptr) {
			real phi_real = 0;
			real norm0 = 0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				//if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				norm0 += s_ptr[j] * sph->Weight<d - 1>(coord.norm(), kernelID);
			}
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				//if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				real r = (e[j].col(d - 1)).dot((e[i].col(d - 1)));
				//printf("%f,", r);
				phi_real += r*Vol[j] * sph->Weight<d - 1>(coord.norm(), kernelID);
			}
			return phi_real;// / norm0;
		}
		static  real __device__ __host__ RealSPH_S(int i, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol) {
			real phi_real = 0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				//if (i == j) continue;
				VectorD r_ij = x[i] - x[j];
				VectorT coord = projectPlane(r_ij, e[i]);
				real r = (e[j].col(d - 1)).dot((e[i].col(d - 1)));
				phi_real += r*sph->Weight<d - 1>(coord.norm(), kernelID);
				//printf("%f ", phi_real);
			}//printf("\n");
			return 1/phi_real;
		}
		template<typename F>
		static  VectorD __device__ __host__ GradTang_Symmetric(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			VectorD grad_phi = VectorD::Zero();
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorD kernel_grad = Kernel_Grad_Tangential(i, j, sph, kernelID, x, e, g);
				grad_phi += Vol[j] * h[i] * kernel_grad * (phi(i) / h[i] / h[i] + phi(j) / h[j] / h[j]);
			}
			return grad_phi;
		}
		template<typename F>
		static  VectorD __device__ __host__  GradTang_Difference(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			VectorD grad_phi = VectorD::Zero();
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorD kernel_grad = Kernel_Grad_Tangential(i, j, sph, kernelID, x, e, g);
				grad_phi += Vol[j] / h[j] * kernel_grad * (phi(j) - phi(i));
			}
			return grad_phi;
		}
		template<typename F>
		static  real __device__ __host__  DivTang(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			real div_phi = (real)0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				VectorD kernel_grad = Kernel_Grad_Tangential(i, j, sph, kernelID, x, e, g);
				div_phi += Vol[j] / h[j] * kernel_grad.dot(phi(i, j));
			}
			printf("This is the div_phi:%.12f\n", div_phi);
			return div_phi;
		}
		template<typename F>
		static  VectorD __device__ __host__ LapTangVec(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			VectorD lap_phi = VectorD::Zero();
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				real kernel_lap = Kernel_Lap_Tangential(i, j, sph, kernelID, x, e, g);
				lap_phi += Vol[j] / h[j] * phi(i, j) * kernel_lap;
			}
			return lap_phi;
		}
		template<typename F>
		static  real __device__ __host__ LapTangReal(int i, const F& phi, SPH* sph, int kernelID, int nbs_num,
			int* nbs, VectorD* x, MatrixD* e, MatrixT* g, real* Vol, real* h) {
			real lap_phi = (real)0;
			for (int k = 0; k < nbs_num; k++) {
				int j = nbs[k];
				if (i == j) continue;
				real kernel_lap = Kernel_Lap_Tangential(i, j, sph, kernelID, x, e, g);
				lap_phi += Vol[j] / h[j] * phi(i, j) * kernel_lap;
			}
			return lap_phi;
		}
		static VectorD __device__ __host__ Kernel_Grad_Tangential(int i, int j, SPH* sph, int kernelID, VectorD* x, MatrixD* e, MatrixT* g) {
			VectorD r_ij = x[i] - x[j];
			VectorT coord = projectPlane(r_ij, e[i]);
			return planeVector(g[i].inverse() * sph->Grad<d - 1>(coord, kernelID), e[i]);
		}
		static real __device__ __host__ Kernel_Lap_Tangential(int i, int j, SPH* sph, int kernelID, VectorD* x, MatrixD* e, MatrixT* g) {
			VectorD r_ij = x[i] - x[j];
			VectorT coord = projectPlane(r_ij, e[i]);
			VectorD grad_kerenel = planeVector((g[i].inverse())* sph->Grad<d - 1>(coord, kernelID),e[i]);
			return 2 * grad_kerenel.norm() / coord.norm();
		}
		
		/// Helper function
	public:
		/// 1. project to local frame
		static real __host__  __device__ dot(const VectorD& v1, const  VectorD& v2) {
			real ans = 0;
			ans += v1[0] * v2[0];
			ans += v1[1] * v2[1];
			if constexpr(d==3) ans += v1[2] * v2[2];
			return ans;
		}
		static VectorD __host__  __device__  mul(const MatrixD& v1, const  VectorD& v2) {
			VectorD ans;
			ans[0] = v1(0, 0) * v2[0] + v1(0, 1) * v2[1] + v1(0, 2) * v2[2];
			ans[1] = v1(1, 0) * v2[0] + v1(1, 1) * v2[1] + v1(1, 2) * v2[2];
			if constexpr (d == 3) ans[2] = v1(2, 0) * v2[0] + v1(2, 1) * v2[1] + v1(2, 2) * v2[2];
			return ans;
		}
		static VectorD __host__ __device__ col(MatrixD e, int i) {
			VectorD ans;
			ans[0] = e(0, i);
			ans[1] = e(1, i);
			if constexpr (d==3) ans[2] = e(2, i);
			return ans;
		}
		static VectorT __host__ __device__ projectPlane(const VectorD& u, const MatrixD& e) {
			VectorT t;
			for (int i = 0; i < d - 1; i++) {
				t[i] =  dot(u, (col(e, i)));
			}
			return t;
		}
		static VectorD __host__ __device__ planeVector(const VectorT& u, const MatrixD& e) {
			VectorD t= VectorD::Zero();
			for (int i = 0; i < d - 1; i++) {
				t += u[i]*e.col(i);
			}
			return t;
		}
		/// 2. check tang --- only host
		static bool __host__ isTangNeighbor(const VectorD& pos, const MatrixD& e_local, const int idx,VectorD* x_host,MatrixD* e_host,real t_r,real t_dot )
		{
			////check angle
			VectorD n = e_local.col(d - 1);
			VectorD n_p = e_host[idx].col(d - 1);
			real angle = n.dot(n_p);
			if (angle < t_dot) return false;	////skip the points with large angles
			////check distance
			VectorD u = x_host[idx] - pos;
			VectorT t = projectPlane(u, e_local);
			return t.norm() < t_r;
		}
		/// 3. define tang neigbor --- only host
		void __host__ findTangNeighbor(int i,Array<MatrixD>& e_host,Array<VectorD>& x_host, Array<int>& nbs)
		{
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e_host[0]);
			VectorD* x_ptr = thrust::raw_pointer_cast(&x_host[0]);
			VectorD pos = x_host[i];
			nb.searchNeibor(pos, nbs);// ,
			//	[x_ptr, e_ptr,i,pos,this]__host__ (int j) {
			//		return isTangNeighbor(pos, e_ptr[i], j, x_ptr, e_ptr, t_r, t_dot);
			//	}
			//);
		}
		/// 4. function to find normal by PCA
		static  VectorD __host__ __device__ Normal(int i, MatrixD* e){ return e[i].col(d - 1); }
		void __host__ init_Normal_All(const Array<MatrixD, HOST>& e) {
			this->e = e;
		}
		static  MatrixD __host__ __device__ init_Frame(int idx, int nbs_num, int* nbs, VectorD* x, MatrixD* e,real v_r) {
			/// find the neighbors must be called before this.
			VectorD x_tidle = VectorD::Zero();
			real w_total = 0;
			MatrixD Cp = MatrixD::Zero();
			//printf("---------------- %d\n", nbs_num);
			for (int i = 0; i < nbs_num; i++) {
				//if (nbs[i] == idx) continue;
				real d = (x[nbs[i]] - x[idx]).norm();
				//printf("%d: %f %f", nbs_num, d, v_r);
				real wij = WPCA(d,v_r);
				x_tidle += WPCA(d, v_r) * x[nbs[i]];
				w_total += wij;
			}
			if (w_total == 0) { printf("Init normal error! invalid w_total"); exit(1); }
			x_tidle /= w_total;
			for (int i = 0; i < nbs_num; i++) {
				//if (nbs[i] == idx) continue;
				real d = (x[nbs[i]] - x[idx]).norm();
				real wij = WPCA(d,v_r);
				Cp += wij * (x[nbs[i]] - x_tidle) * (x[nbs[i]] - x_tidle).transpose();
			}Cp /= w_total;
			/// Note that: the normal cannot change dramastically!
			/// the direction for new normal must be close to the old one
			VectorD normal = Min_Eigenvector<d>(Cp);
			
			if ((e[idx].col(d - 1)).dot(normal) < 0) normal = -normal;
			//printf("%d:This is the normal:(%f,%f,%f) This is the position:(%f,%f,%f)\n", idx, normal[0], normal[1], normal[2], x[idx][0], x[idx][1], x[idx][2]);
			return Frame_From_Normal(normal);
		}
		static  MatrixD __host__ __device__ Frame_From_Normal(VectorD normal) {
			MatrixD new_e;
			if constexpr (d == 2) {
				new_e.col(0) = Orthogonal_Vector<d>(normal).normalized();
				new_e.col(1) = normal.normalized();
				return new_e;
			}
			else if constexpr (d == 3) {
				new_e.col(0) = Orthogonal_Vector<d>(normal).normalized();
				new_e.col(2) = normal.normalized();
				new_e.col(1) = (new_e.col(0)).cross(new_e.col(2));
				return new_e;
			}
		}
		static  MatrixD __host__ __device__ update_Frame(int i, int nbs_num, int* nbs, VectorD* x, MatrixD* e,real v_r) {
			/// TODO, which may not be implemented and be substitute by init_normal
			return init_Frame(i, nbs_num, nbs, x, e, v_r);
		}
		static  real __host__ __device__ WPCA(const real r,const real v_r) 
		{
			if (r < v_r)return (real)1 - pow(r / v_r, 3);
			else return (real)0;
		}
		/// 5. approxmate local shape  and calculate metrix tensor
		template<typename F>
		static  Vectord<3 * (d - 1)> __host__ __device__ Fit_Local_MLS(int idx, F& phi, int nbs_num, int* nbs, VectorD* x, MatrixD* e) {
			real* data = new real[(d - 1) * nbs_num];
			real* f = new real[nbs_num];
			for (int i = 0; i < nbs_num; i++) {
				int nb = nbs[i];
				VectorT tang = projectPlane(x[nb] - x[idx], e[idx]);
					if constexpr (d == 2) {
						data[i] = tang[0];
					}
					else if constexpr (d == 3) {
						data[2 * i] = tang[0];
						data[2 * i + 1] = tang[1];
					}
					f[i] = phi(nb);
			}
			Vectord<3 * (d - 1)> ans= MLS_Fit<d,side>(data, f, nbs_num, VectorT::Zero());
			delete[] data;
			delete[] f;
			return ans;
			//return VectorX(5);
		}
		static  Vectord<3 * (d - 1)> __host__ __device__ Fit_Local_Shape(int idx, int nbs_num, int* nbs, VectorD* x, MatrixD* e) {
			auto phi = [e, x, idx]__device__ __host__ (int j) {
				//return (e[j].col(d - 1)).dot(x[j] - x[idx]);
				return (e[idx].col(d - 1)).dot(x[j] - x[idx]);
			};
			//printf("in fit shaoe idx:%d %f %f %f\n", idx, x[idx][0], x[idx][1], x[idx][2]);
			return Fit_Local_MLS(idx, phi, nbs_num, nbs, x, e);

		}
		static  MatrixT __host__ __device__ Calculate_Tensor(int idx, int nbs_num, int* nbs, VectorD* x, MatrixD* e) {
			//printf("idx:%d %f %f %f\n", idx, x[idx][0], x[idx][1], x[idx][2]);
			Vectord<3 * (d - 1)> c=Fit_Local_Shape(idx, nbs_num, nbs, x, e);
			//printf("here is the out ans %f %f %f %f %f %f\n", c[0], c[1], c[2], c[3], c[4], c[5]);
			MatrixT metrix_tensor;
			if constexpr (d == 2) metrix_tensor << 1 + c[1] * c[1];
			else if constexpr (d == 3) {
				metrix_tensor << 1 + c[1] * c[1], c[2] * c[1], c[2] * c[1], 1 + c[2] * c[2];
			}
			//printf("finish tensor\n");
			return metrix_tensor;
			
		}
		/// 5. Vol
		//real Vol(const int idx) const { return M(idx) / pho; }
		// Helpter function in calculate the process

		/*
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
		Array<int> __host__ findTangNeighbor(const VectorD& pos, const MatrixD& e) const
		{
			Array<int> nbs;
			MatrixD* e_ptr = thrust::raw_pointer_cast(&e[0]);
			nb.searchNeibor(pos, nbs, 
				[]() {}
			);
			return nbs;
		}
		
		*/
	};
}
#endif