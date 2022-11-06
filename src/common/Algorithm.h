#ifndef ALGORITHM_H
#define ALGORITHM_H
#include"Common.h"
#include<unordered_map>
namespace ACG {
	/// Aux function
	#define pi_math  ((real)3.14159265358979)
	real __host__ __device__ SlowPower2(const real& a) { return a * a; }
	real __host__ __device__ SlowPower3(const real& a) { return a * a * a; }
	real __host__ __device__  SlowPower4(const real& a) { real b = a * a; return b * b; }
	template<int d>
	Eigen::Matrix<real, d, 1> __host__ __device__ Unit(int i) {
		using VectorD = Eigen::Matrix<real, d, 1>;
		if constexpr (d == 3) {
			if (i == 0) return VectorD(1, 0, 0);
			else if (i == 1) return VectorD(0, 1, 0);
			else VectorD(0, 0, 1);
		}
		else if constexpr (d == 2) {
			if (i == 0) return VectorD(1, 0);
			else if (i == 1) return VectorD(0, 1);
		}
	}
	template<int d>
	int  __host__ __device__ Min_Index(Eigen::Matrix<real, d, 1> v) {
		if constexpr (d == 2) return abs(v[0]) < abs(v[1]) ? 0 : 1;
		else if constexpr (d == 3) {
			return abs(v[0]) < abs(v[1]) ? (abs(v[0]) < abs(v[2]) ? 0 : 2) : (abs(v[1]) < abs(v[2]) ? 1 : 2);
		}
	}
	template<int d>
	Eigen::Matrix<real, d, 1> __host__ __device__ Min_Eigenvector(const Eigen::Matrix<real, d, d>& v)
	{
		using MatrixD = Eigen::Matrix<real, d, d>;
		using VectorD = Eigen::Matrix<real, d, 1>;
		Eigen::SelfAdjointEigenSolver<MatrixD> eig(v);
		int index = 0; VectorD eigenValues = eig.eigenvalues();
		for (int i = 1; i < d; i++) {
			if (abs(eigenValues[index]) > abs(eigenValues[i])) index = i;
		}
		return eig.eigenvectors().col(index);
	}
	template<int d>
	Eigen::Matrix<real, d, 1> __host__ __device__ Orthogonal_Vector(Eigen::Matrix<real, d, 1> v) {
		using VectorD = Eigen::Matrix<real, d, 1>;
		if constexpr (d == 2) return VectorD(v[1], -v[0]);
		else {
			int index = Min_Index<d>(v);
			return (index == 0) ?
				(VectorD(0, -v[2], v[1])) :
				((index == 1) ? VectorD(v[2], 0, -v[0]) : VectorD(-v[1], v[0], 0));
		}
	}

/// Grid
	template<int d>
	class Grid {
	public:
		Typedef_VectorDii(d);
		VectorD MinPosition;
		VectorD MaxPosition;
		real dx;
		int cellNumber;
		Grid() {}
		Grid(real dx, const VectorD& MaxPosition, const VectorD& MinPosition) :
			MinPosition(MinPosition), MaxPosition(MaxPosition), dx(dx)
		{
			cellNumber = CellNumber();
		}
		Grid(Grid& grid) {
			this->MinPosition = grid->MinPosition;
			this->MaxPosition = grid->MaxPosition;
			this->dx = grid->dx;
			this->cellNumber = grid->cellNumber;
		}
		VectorDi getCell(VectorD pos) {
			return ((pos - MinPosition) / dx).template cast<int>();
		}
		int getCellId(VectorDi cell) {
			VectorDi tem = CellNumberDim();
			if constexpr (d == 2) return tem[0] * cell[1] + cell[0];
			else if constexpr (d == 3) return tem[1] * tem[0] * cell[2] + tem[0] * cell[1] + cell[0];
		}
		VectorDi getCell(int cellId) {
			VectorDi tem = CellNumberDim();
			if constexpr (d == 3) {
				tem[2] = cellId / (tem[1] * tem[0]);
				cellId = cellId % (tem[1] * tem[0]);
			}
			tem[1] = cellId / tem[0];
			tem[0] = cellId % tem[0];
			return tem;
		}
		int CellNumber() {
			VectorDi tem = CellNumberDim();
			if constexpr (d == 2) return tem[0] * tem[1];
			else if constexpr (d == 3) return tem[0] * tem[1] * tem[2];
		}
		VectorDi CellNumberDim() {
			/// Note: all the size will increase one!
			return ((MaxPosition - MinPosition) / dx).template cast<int>() + VectorDi::Ones();
		}
		bool ValidCell(VectorDi cell) {
			VectorDi tem = CellNumberDim();
			if (cell[0] < 0 || cell[1] < 0) return false;
			if (cell[0] >= tem[0] || cell[1] >= tem[1]) return false;
			if constexpr (d == 3) {
				if (cell[2] < 0 || cell[2] >= tem[2]) return false;
			}
			return true;
		}
		bool getNextCell(VectorDi cell, int index, VectorDi& ans) {
			static VectorDi add_term_2D[]{
			VectorDi(-1,-1),VectorDi(0,-1),VectorDi(1,-1),
			VectorDi(-1,0),VectorDi(0,0),VectorDi(1,0),
			VectorDi(-1,1),VectorDi(0,1),VectorDi(1,1) };
			static VectorDi add_term_3D[]{
			VectorDi(-1,-1,-1),VectorDi(0,-1,-1),VectorDi(1,-1,-1),
			VectorDi(-1,0,-1),VectorDi(0,0,-1), VectorDi(1,0,-1),
			VectorDi(-1,1,-1),VectorDi(0,1,-1),VectorDi(1,1,-1),
			VectorDi(-1,-1,0),VectorDi(0,-1,0),VectorDi(1,-1,0),
			VectorDi(-1,0,0),VectorDi(0,0,0),VectorDi(1,0,0),
			VectorDi(-1,1,0),VectorDi(0,1,0),VectorDi(1,1,0),
			VectorDi(-1,-1,1),VectorDi(0,-1,1),VectorDi(1,-1,1),
			VectorDi(-1,0,1),VectorDi(0,0,1), VectorDi(1,0,1),
			VectorDi(-1,1,1),VectorDi(0,1,1),VectorDi(1,1,1) };
			if constexpr (d == 2) {
				ans = add_term_2D[index] + cell;
			}
			else if constexpr (d == 3) {
				ans = add_term_3D[index] + cell;
			}
			return ValidCell(ans);
		}
	};
/// Class to find the neighbor of points and the interfaces are as follow,
/// 1. construct(points): construct spatial hashing structure for points
/// 2. construct(points): update spatial hashing structure for points, update may be different from construct and more fast
/// 3.searchNeibor(pos, ans): find the neightor points for pos,note that , point may be not in the initial points for construct 
	template<int d>
	class Neighbor {
		Typedef_VectorDii(d);
	public:
		Grid<d> grid;
		std::unordered_map<int, Array<int>> hash_table;
		Array<VectorD> points;
	public:
		void construct(Array<VectorD>& points, real dx) {
			this->points = points;
			VectorD MaxPosition = getMaxPosition();
			VectorD MinPosition = getMinPosition();
			std::cout << MaxPosition << std::endl;
			VectorD d = (MaxPosition - MinPosition);
			MaxPosition = d * 2 + MinPosition;
			MinPosition = MinPosition - d;
			construct(dx, MaxPosition, MinPosition);
		}
		void construct(Array<VectorD>& points, real dx, const VectorD& MaxPosition, const VectorD& MinPosition) {

			this->points = points;
			construct(dx, MaxPosition, MinPosition);
		}
		void construct(real dx, const VectorD& MaxPosition, const VectorD& MinPosition) {
			hash_table.clear();
			grid = Grid<d>(dx, MaxPosition, MinPosition);
			/// TODO: here construct the hash table, you must use the grid class
			for (int i = 0; i < points.size(); i++) {
				//if (i % 100 == 0) printf("spatial update:%d\n", i);
				VectorDi cell = grid.getCell(points[i]);
				int cellID = grid.getCellId(cell);
				if (hash_table.find(cellID) == hash_table.end()) {
					hash_table.insert(std::pair<int, Array<int>>(cellID, Array<int>()));
				}
				hash_table[cellID].push_back(i);
			}
		}
		void update(Array<VectorD>& points) {
			update(points, grid.MaxPosition, grid.MinPosition);
		}
		void update(Array<VectorD>& points, const VectorD& MaxPosition, const VectorD& MinPosition) {
			/// TODO: here need to update the hash table
			construct(points, grid.dx, MaxPosition, MinPosition);
		}
		template<class F>
		void searchNeibor(VectorD pos, Array<int>& ans, F& f) {
			/// TODO: here need to get the array of index for neighbor points
			/// Please consider how to convert std::vector to host vector
			ans.clear();
			// change: Fix the parameter of getCell from points[i] to pos
			VectorDi cell = grid.getCell(pos);
			int range = d == 2 ? 9 : 27;
			for (int i = 0; i < range; i++) {
				VectorDi adjCell;
				if (grid.getNextCell(cell, i, adjCell)) {
					int cellId = grid.getCellId(adjCell);
					if (hash_table.find(cellId) != hash_table.end()) {
						for (int j = 0; j < hash_table[cellId].size(); j++) {
							int point_index = hash_table[cellId][j];
							if ((points[point_index] - pos).norm() <= grid.dx && f(point_index)) ans.push_back(point_index);
						}
					}
				}
			}
			//printf("%d\n", ans.size());
		}
		void searchNeibor(VectorD pos, Array<int>& ans) {
			/// TODO: here need to get the array of index for neighbor points
			/// Please consider how to convert std::vector to host vector
			searchNeibor(pos, ans, [](int j)->bool {return true; });
		}
		int closestPoint(VectorD pos, Array<int>& ans, real& ans_dis) {
			/// TODO: here need to get hte closest point index
			ans.clear();
			// change position[i] to pos
			VectorDi cell = grid.getCell(pos);
			int range = d == 2 ? 8 : 26;
			int ans_index = -1;
			for (int i = 0; i < range; i++) {
				VectorDi adjCell;
				if (grid.getNextCell(cell, i, adjCell)) {
					int cellId = grid.getCellId(adjCell);
					if (hash_table.find(cellId) != hash_table.end()) {
						for (int j = 0; j < hash_table[cellId].size(); j++) {
							int point_index = hash_table[cellId][j];
							if ((points[point_index] - pos).norm() <= grid.dx) ans.push_back(point_index);
							if (ans_index < 0) {
								ans_index = point_index;
								ans_dis = (points[point_index] - pos).norm();
							}
							else if ((points[point_index] - pos).norm() < (points[ans_index] - pos).norm()) {
								ans_index = point_index;
								ans_dis = (points[point_index] - pos).norm();
							}
						}
					}
				}
			}
			return ans_index;
		}
		int closestPoint(VectorD pos, real& ans_dis) {
			/// TODO: here need to get hte closest point index

			// change points[i] to pos
			VectorDi cell = grid.getCell(pos);
			int range = d == 2 ? 8 : 26;
			int ans_index = -1;
			for (int i = 0; i < range; i++) {
				VectorDi adjCell;
				if (grid.getNextCell(cell, i, adjCell)) {
					int cellId = grid.getCellId(adjCell);
					if (hash_table.find(cellId) != hash_table.end()) {
						for (int j = 0; j < hash_table[cellId].size(); j++) {
							int point_index = hash_table[cellId][j];
							if (ans_index < 0) {
								ans_index = point_index;
								ans_dis = (points[point_index] - pos).norm();
							}
							else if ((points[point_index] - pos).norm() < (points[ans_index] - pos).norm()) {
								ans_index = point_index;
								ans_dis = (points[point_index] - pos).norm();
							}
						}
					}
				}
			}
			return ans_index;
		}
	public:
		/*Structure defined by yourself*/
	protected:
		VectorD getMinPosition() {
			; /// TODO : return the lower bound for bounding box
			ArrayIter<VectorD> largest1 = thrust::min_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[0] < b[0]; });
			ArrayIter<VectorD> largest2 = thrust::min_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[1] < b[1]; });
			ArrayIter<VectorD> largest3;
			auto com3 = [](VectorD a, VectorD b)->bool {return a[2] < b[2]; };
			if constexpr (d == 2) {
				return VectorD((*largest1)[0], (*largest2)[1]);
			}
			else if constexpr (d == 3) {
				largest3 = thrust::min_element(points.begin(), points.end(), com3);
				return VectorD((*largest1)[0], (*largest2)[1], (*largest3)[2]);
			}
		}
		VectorD getMaxPosition() {
			/// TODO : return the upper bound for bounding box
			ArrayIter<VectorD> largest1 = thrust::max_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[0] < b[0]; });
			ArrayIter<VectorD> largest2 = thrust::max_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[1] < b[1]; });
			ArrayIter<VectorD> largest3; auto com3 = [](VectorD a, VectorD b)->bool {return a[2] < b[2]; };
			//change the iterator to value by adding *
			if constexpr (d == 2) {
				return VectorD((*largest1)[0], (*largest2)[1]);
			}
			else if constexpr (d == 3) {
				largest3 = thrust::max_element(points.begin(), points.end(), com3);
				return VectorD((*largest1)[0], (*largest2)[1], (*largest3)[2]);
			}
		}
	};

	/// Class to do the SPH
	class UnitKernel {
		Typedef_VectorDii(3);
	public:
		int type;
		VectorD alpha;//0,1,2, -> d=1,2,3
		real __device__ __host__  Weight(const int d, const real r)const { Assert(true, "base function of unitKernel could not be used"); }
		real __device__ __host__  Grad(const int d, const real r)const { Assert(true, "base function of unitKernel could not be used"); }
	};
	class UnitSPIKY : public UnitKernel {
		Typedef_VectorDii(3);
	public:
		__device__ __host__ UnitSPIKY() { alpha = VectorD(2.0, 10.0 / pi_math, 15.0 / pi_math); type = 0; }
		real __device__ __host__ Weight(const int d, const real r)const { return r < 1 ? alpha[d - 1] * SlowPower3(1 - r) : 0; }
		real __device__ __host__ Grad(const int d, const real r)const { return r < 1 ? alpha[d - 1] * (-3) * SlowPower2(1 - r) : 0; }
	};
	class UnitCUBIC :public UnitKernel {
		Typedef_VectorDii(3);
	public:
		__device__ __host__ UnitCUBIC() { alpha = VectorD(4.0 / 3.0, 40.0 / (7.0 * pi_math), 8.0 / pi_math); type = 1;}
		real __device__ __host__ Weight(const int d, const real r)const {
			if (0 <= r && r < 0.5) return alpha[d - 1] * ((real)6 * r * r * r - (real)6 * r * r + 1);
			else if (0.5 <= r && r < 1) return alpha[d - 1] * 2 * SlowPower3(1 - r);
			else return 0;
		}
		real __device__ __host__ Grad(const int d, const real r)const {
			if (0 <= r && r < 0.5) return alpha[d - 1] * 6.0 * r * (3.0 * r - 2.0);
			else if (0.5 <= r && r < 1) return alpha[d - 1] * (-6.0) * SlowPower2(1.0 - r);
			else return 0;
		}
	};

	class SPH {
		using real = double;
		real h;
		real h_pows_inv[5];
	public:
		//0:SPIKY,1:Cubic,2: TODO quatratic
		UnitSPIKY unitSPIKY;
		UnitCUBIC unitCUBIC;
		__device__ __host__ SPH(real _h = 1.0) :h(_h) {
			h_pows_inv[0]=1;
			for (int i = 1; i < 5; i++) { h_pows_inv[i]=h_pows_inv[i - 1] / h; }
		}
		template<int d> real __device__ __host__ Weight(real r, int kernelID)const {
			if(kernelID==0)
				return unitSPIKY.Weight(d, fabs(r / h)) * h_pows_inv[d];
			else if (kernelID == 1)
				return unitCUBIC.Weight(d, fabs(r / h)) * h_pows_inv[d];
		}
		template<int d> real __device__ __host__ Grad_Norm(const real r, int kernelID) const {
			if (kernelID == 0)
				return unitSPIKY.Grad(d, fabs(r / h)) * h_pows_inv[d + 1];
			else if (kernelID == 1)
				return unitCUBIC.Grad(d, fabs(r / h)) * h_pows_inv[d + 1];
		}
		template<int d> 
		Vectord<d> __device__ __host__ Grad(const Vectord<d>& r, int kernelID)const {
			real r_len = r.norm();
			if (r_len == 0) return r;
			real grad_coeff = Grad_Norm<d>(r_len, kernelID) / r_len;
			return  grad_coeff * r;
		}
	};
}
#endif