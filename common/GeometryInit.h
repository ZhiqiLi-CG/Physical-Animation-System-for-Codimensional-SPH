#ifndef __Geometry_Init_h__
#define __Geometry_Init_h__
#include "Common.h"
#include <queue>
namespace ACG{
	/*
	class Triangle
	{
	public:
		Typedef_VectorDii(3);
		VectorD a, b, c;
		Triangle(real ax, real ay, real az, real bx, real by, real bz, real cx, real cy, real cz) :
			a(VectorD(ax, ay, az)), b(VectorD(bx, by, bz)), c(VectorD(cx, cy, cz)) {}
		Triangle() {}
		Triangle(VectorD a, VectorD b, VectorD c):a(a),b(b),c(c) {}
	};*/
	class InitSphere {
	public:
		Typedef_VectorDii(3);
		real radius;
		real dx;
		Array<VectorD> points;
		Array<VectorDi> triangles;
		InitSphere(real radius,real stop_dx):radius(radius) {
			dx = splitTriagle(radius, stop_dx, points, triangles);
		}
		void draw_points() {

		}
		void draw_lines() {

		}
		void draw_triangles() {

		}
		static VectorD midArc(VectorD a, VectorD b)
		{
			VectorD c(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
			return c/c.norm()*a.norm();
		}
		static real splitTriagle(real radius, real stop_dx,Array<VectorD>& points, Array<VectorDi>& triangles) {
			std::queue<VectorDi> triangl_list; points.clear(); triangles.clear();
			points.push_back(VectorD(radius, 0, 0));
			points.push_back(VectorD(0, radius, 0));
			points.push_back(VectorD(0, 0, radius));
			triangl_list.push(VectorDi(0,1,2));
			real real_dx;
			while (1) {
				VectorDi triangle = triangl_list.front();
				VectorD a = points[triangle[0]];
				VectorD b = points[triangle[1]];
				VectorD c = points[triangle[2]];
				if ((a-b).norm() < stop_dx) {
					real_dx = (a-b).norm();
					break;
				}
				VectorD ab = midArc(a, b);
				VectorD bc = midArc(b, c);
				VectorD ca = midArc(c, a);
				points.push_back(ab);
				points.push_back(bc);
				points.push_back(ca);
				int ab_index = points.size() - 3;
				int bc_index = points.size() - 2;
				int ca_index = points.size() - 1;
				triangl_list.push(VectorDi(triangle[0], triangle[1], ab_index));
				triangl_list.push(VectorDi(triangle[1], triangle[2], bc_index));
				triangl_list.push(VectorDi(triangle[2], triangle[0], ca_index));
				triangl_list.push(VectorDi(ab_index, bc_index, ca_index));
				triangl_list.pop();
			}
			while (!triangl_list.empty()) {
				triangles.push_back(triangl_list.front());
				triangl_list.pop();
			}
			return real_dx;
		}
	};
}
#endif