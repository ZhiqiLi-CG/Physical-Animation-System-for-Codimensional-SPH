#ifndef GEOMETRYINIT_H
#define GEOMETRYINIT_H
#include "Common.h"
#include <queue>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include "camera.h"
#include "constant.h"
#include<unordered_map>
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
		Array<real> Vol,h,m,vo, GM;
		Array<VectorD> v, normals;
		Array<VectorDi> triangles;
		VectorD maxPosition, minPosition;
		InitSphere() {}
		InitSphere(real radius,real stop_dx):radius(radius) {
			dx = splitTriagle(radius, stop_dx, points, triangles);
			printf("This is the dx:%f", dx);
			real V = 4 * radius * radius * pi_math;
			real S=V / points.size();
			Vol.reserve(points.size());
			h.reserve(points.size());
			for (int i = 0; i < points.size(); i++) {
				h.push_back(6e-7);
				//printf("%0.9lf+++\n", 6e-7);
				Vol.push_back(600.0f * 1e-9 * S);
			}
			for (int i = 0; i < points.size(); i++) {
				//printf("M:%f\n", Vol[i] * 1e3);
				m.push_back(Vol[i] * 1e3);
			}
			for (int i = 0; i < points.size(); i++) {
				v.push_back(VectorD::Zero());
			}
			for (int i = 0; i < points.size(); i++) {
				normals.push_back(points[i].normalized());
			}
			for (int i = 0; i < points.size(); i++) {
				vo.push_back(0);
			}
			for (int i = 0; i < points.size(); i++) {
				GM.push_back(1e-7);
			}
			ArrayIter<VectorD> largest1 = thrust::max_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[0] < b[0]; });
			ArrayIter<VectorD> largest2 = thrust::max_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[1] < b[1]; });
			ArrayIter<VectorD> largest3 = thrust::max_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[2] < b[2]; });
			maxPosition = VectorD(largest1[0][0], largest2[0][1], largest3[0][2]);
			largest1 = thrust::min_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[0] < b[0]; });
			largest2 = thrust::min_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[1] < b[1]; });
			largest3 = thrust::min_element(points.begin(), points.end(), [](VectorD a, VectorD b)->bool {return a[2] < b[2]; });
			minPosition = VectorD(largest1[0][0], largest2[0][1], largest3[0][2]);
			VectorD d = maxPosition - minPosition;
			maxPosition += 10 * d;
			minPosition -= 10 * d;
		}
		void augmentH(real aug,int side=0) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if (side == 0 && points[i][2] < 0)
					h[i] += aug;// *abs(points[i][2]);
				if (side == 1 && points[i][2] > 0)
					h[i] += aug;// *abs(points[i][2]);
			}
		}
		void augmentH2(real aug, int side = 0) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if (side == 0 && points[i][2] < 0)
					h[i] = aug;// *abs(points[i][2]);
				if (side == 1 && points[i][2] > 0)
					h[i] = aug;// *abs(points[i][2]);
			}
		}
		void sinH(real aug) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
					h[i] += aug*sin(points[i][0]+ points[i][1]+ points[i][2]);
			}
		}
		void sinH2(real aug,real range) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if(points[i][2]>-range&& points[i][2]<range)
					h[i] += aug * sin(10*(points[i][0] + points[i][1]));
			}
		}
		void augmentGM(real aug, int side = 0) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if (side == 0 && points[i][2] < 0)
					GM[i] += aug;
				if (side == 1 && points[i][2] > 0)
					GM[i] += aug;
			}
		}
		void augmentVorticity(real aug, real range) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if (points[i][2]<range && points[i][2] > -range) {
					vo[i] += aug;
				}
			}
		}
		void augmentVorticity(real aug, real range1, real range2) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if (points[i][2]<range1 && points[i][2] > range2) {
					vo[i] += aug*abs(points[i][2]);
				}
			}
		}
		void rotateV(real aug) {
			for (int i = 0; i < points.size(); i++) {
				v[i] = VectorD(-points[i][1], points[i][0], 0).normalized() * (points[i][1] * points[i][1] + points[i][0] * points[i][0]) * aug;
			}
		}
		void augmentV(real aug, int side = 0) { // 0:low, 1:high
			for (int i = 0; i < points.size(); i++) {
				if (side == 0 && points[i][2] < 0)
					v[i] +=VectorD(0,0, aug*abs(points[i][2])* 1e-3);
				if (side == 1 && points[i][2] > 0)
					v[i] += VectorD(0, 0, aug * abs(points[i][2])*1e-3);
				if (side == 2 && points[i][1] < 0)
					v[i] += VectorD(0,  aug * abs(points[i][2]) * 1e-3, 0);
				if (side == 3 && points[i][1] > 0)
					v[i] += VectorD(0, aug * abs(points[i][2]) * 1e-3,0);
			}
		}
		void draw_points() {
			for (int i = 0; i < points.size(); i++) {
				glColor3f(1, 0, 0);
				glBegin(GL_POINTS);
				glVertex3f(points[i][0], points[i][1], points[i][2]);
				glEnd();
			}
		}
		void draw_lines() {
			for (int i = 0; i < triangles.size(); i++) {
				glColor3f(1, 0, 0);
				glBegin(GL_LINES);
				glVertex3f(points[triangles[i][0]][0], points[triangles[i][0]][1], points[triangles[i][0]][2]);
				glVertex3f(points[triangles[i][1]][0], points[triangles[i][1]][1], points[triangles[i][1]][2]);
				glEnd();
				glBegin(GL_LINES);
				glVertex3f(points[triangles[i][2]][0], points[triangles[i][2]][1], points[triangles[i][2]][2]);
				glVertex3f(points[triangles[i][1]][0], points[triangles[i][1]][1], points[triangles[i][1]][2]);
				glEnd();
				glBegin(GL_LINES);
				glVertex3f(points[triangles[i][0]][0], points[triangles[i][0]][1], points[triangles[i][0]][2]);
				glVertex3f(points[triangles[i][2]][0], points[triangles[i][2]][1], points[triangles[i][2]][2]);
				glEnd();
			}
		}
		Array<VectorD> getSphereColor(Camera& camera) {
			Array<VectorD> colors;
			VectorD position = VectorD(camera.camera_x, camera.camera_y, camera.camera_z);
			real gamma = 1;
			for (int i = 0; i < triangles.size(); i++) {
				VectorD normal; 
				real color[3];
				getArtifactColor(position, points[triangles[i][0]], h[triangles[i][0]], gamma, color);
				colors.push_back(VectorD(color[0], color[1], color[2]));
				getArtifactColor(position, points[triangles[i][1]], h[triangles[i][1]], gamma, color);
				colors.push_back(VectorD(color[0], color[1], color[2]));
				getArtifactColor(position, points[triangles[i][2]], h[triangles[i][2]], gamma, color);
				colors.push_back(VectorD(color[0], color[1], color[2]));
			}
			return colors;
			//printf("%d %d %d\n", color[0], color[1], color[2]);

		}
		real ratio(VectorD v1, VectorD v2, VectorD u1, VectorD u2) {
			real r1 = (v1 - v2).norm() / (u1 - u2).norm();
			real r2 = (u1 - u2).norm() / (v1 - v2).norm();
			if (r1 >= 1) return r1;
			else return r2;
		}
		bool validTriangle(VectorD v1, VectorD v2, VectorD v3, VectorD u1, VectorD u2, VectorD u3) {
			real threshhold =4;
			if (ratio(v1, v2, u1, u2) >= threshhold || ratio(v3, v2, u3, u2) >= threshhold || ratio(v1, v3, u1, u3) >= threshhold)
				return false;
			return true;
		}
		void draw_triangles(Array<VectorD> color, Array<VectorD> initPoints) {
			//glColor3f(1.0, 0, 0);
			for (int i = 0; i < triangles.size(); i++) {
				if (validTriangle(points[triangles[i][0]], points[triangles[i][1]], points[triangles[i][2]], initPoints[triangles[i][0]], initPoints[triangles[i][1]], initPoints[triangles[i][2]])) {
					glBegin(GL_TRIANGLES);
					glColor3f(color[i * 3][0], color[i * 3][1], color[i * 3][2]);
					glVertex3f(points[triangles[i][0]][0], points[triangles[i][0]][1], points[triangles[i][0]][2]);
					glColor3f(color[i * 3 + 1][0], color[i * 3 + 1][1], color[i * 3 + 1][2]);
					glVertex3f(points[triangles[i][1]][0], points[triangles[i][1]][1], points[triangles[i][1]][2]);
					glColor3f(color[i * 3 + 2][0], color[i * 3 + 2][1], color[i * 3 + 2][2]);
					glVertex3f(points[triangles[i][2]][0], points[triangles[i][2]][1], points[triangles[i][2]][2]);
					glEnd();
				}
				else {
					glBegin(GL_POINTS);
					glColor3f(color[i * 3][0], color[i * 3][1], color[i * 3][2]);
					glVertex3f(points[triangles[i][0]][0], points[triangles[i][0]][1], points[triangles[i][0]][2]);
					glEnd();
					glBegin(GL_POINTS);
					glColor3f(color[i * 3 + 1][0], color[i * 3 + 1][1], color[i * 3 + 1][2]);
					glVertex3f(points[triangles[i][1]][0], points[triangles[i][1]][1], points[triangles[i][1]][2]);
					glEnd();
					glBegin(GL_POINTS);
					glColor3f(color[i * 3 + 2][0], color[i * 3 + 2][1], color[i * 3 + 2][2]);
					glVertex3f(points[triangles[i][2]][0], points[triangles[i][2]][1], points[triangles[i][2]][2]);
					glEnd();
				}
			}
		}
		static void getArtifactColor(VectorD camera_position, VectorD point, real h, real gamma, real color[3]) {
			VectorD normal = point; normal = normal.normalized();
			real cos_theta = (point - camera_position).normalized().dot(normal);
			//printf("-------%0.9lf %f\n", h,cos_theta);
			real thick = artifact_thick(h * 2, gamma, cos_theta);
			//printf("-------%0.9lf\n", thick);
			getColor(thick*1e9, color);

		}
		static VectorD midArc(VectorD a, VectorD b)
		{
			VectorD c(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
			return c/c.norm()*a.norm();
		}
		static int find_pair(std::unordered_map<long long, int>& pairs, int x, int y,int id) {
			long long pair;
			if (x >= y) std::swap(x, y);
			pair = ((((long long)1L) << 32) * ((long long)x)) + y;
			if (pairs.find(pair) != pairs.end()) return pairs[pair];
			else {
				pairs.insert(std::pair<long long, int>(pair, id));
				return id;
			}
		}
		static real splitTriagle(real radius, real stop_dx,Array<VectorD>& points, Array<VectorDi>& triangles) {
			std::queue<VectorDi> triangl_list; points.clear(); triangles.clear();
			points.push_back(VectorD(radius, 0, 0));
			points.push_back(VectorD(0, radius, 0));
			points.push_back(VectorD(0, 0, radius));
			points.push_back(VectorD(-radius, 0, 0));
			points.push_back(VectorD(0, -radius, 0));
			points.push_back(VectorD(0, 0, -radius));
			triangl_list.push(VectorDi(0,1,2));
			triangl_list.push(VectorDi(0, 4, 2));
			triangl_list.push(VectorDi(3, 4, 2));
			triangl_list.push(VectorDi(3, 2, 1));

			triangl_list.push(VectorDi(0, 1, 5));
			triangl_list.push(VectorDi(0, 4, 5));
			triangl_list.push(VectorDi(3, 4, 5));
			triangl_list.push(VectorDi(3, 5, 1));

			real real_dx;
			std::unordered_map<long long,int> pairs;
			while (1) {
				VectorDi triangle = triangl_list.front();
				VectorD a = points[triangle[0]];
				VectorD b = points[triangle[1]];
				VectorD c = points[triangle[2]];
				//printf("%f ", (a - b).norm());
				if ((a-b).norm() < stop_dx) {
					real_dx = (a-b).norm();
					break;
				}
				VectorD ab = midArc(a, b);
				int ab_index = find_pair(pairs,triangle[0], triangle[1], points.size());
				if(ab_index== points.size()){ points.push_back(ab); }
				VectorD bc = midArc(b, c);
				int bc_index = find_pair(pairs, triangle[1], triangle[2], points.size());
				if (bc_index == points.size()) { points.push_back(bc); }
				VectorD ca = midArc(c, a);
				int ca_index = find_pair(pairs, triangle[2], triangle[0], points.size());
				if (ca_index == points.size()) { points.push_back(ca); }
				triangl_list.push(VectorDi(triangle[0], ca_index, ab_index));
				triangl_list.push(VectorDi(triangle[1], ab_index, bc_index));
				triangl_list.push(VectorDi(triangle[2], bc_index, ca_index));
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