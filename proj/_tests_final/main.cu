#include <iostream>
#include <stdlib.h>
//#define GLUT_DISABLE_ATEXIT_HACK
//#include <windows.h>
//#include <GL/glew.h>
//#include <GL/gl.h>
//#include <GL/glut.h>
#include <iostream>
#include <cstdio>

#include "Common.h"
#include "Algorithm.h"
#include "GeometryInit.h"
#include "constant.h"
#include "Particles.h"
#include "IO.h"
#include "LA.h"
using namespace ACG;
int frames_number = 500;
#define __THRUST_HAS_CUDART__
void rotate1(InitSphere& sphere) {
    sphere.rotateV(15);
    sphere.augmentV(2000, 0);
    sphere.augmentV(-2000, 1);
}
int main()
{
    Inverse<6>(Eigen::Matrix<real, 6, 6>::Zero());
    
    Particles<3,HOST> particles;
 
    
   // for (int i = 0; i < sphere.points.size(); i++) {
       // printf("-->%f %f %f\n", sphere.points[i][0], sphere.points[i][1], sphere.points[i][2]);
   // }
    real init_dt = 0.001;
    real init_dx=0.03;
    std::string config_file = "D:\\ACG\\config.txt";
    std::ifstream in;
    in = std::ifstream(config_file.c_str());
    std::unordered_map<std::string, real> arg_map;
    while (!in.eof()) {
        std::string name;
        real num;
        in >> name >> num;
        printf("load:%s,%f\n", name.c_str(), num);
        arg_map.insert(std::pair<std::string, real>(name, num));
        if (name == "debug_update") {
            if (num < 0) debug_update = false;
            else debug_update = true;
        }
        if (name == "init_dx") {
            init_dx = num;
        }
        if (name == "init_dt") {
            init_dt = num;
        }
    }
    InitSphere sphere(1, init_dx);//0.03
   // sphere.augmentH(2e-7, 1);
    //sphere.sinH2(1e-7, 0.2);
   // sphere.augmentGM(5e-8, 1);
   // sphere.augmentVorticity(1e-4,0.1);
    //sphere.rotateV(15);
   // sphere.augmentV(2000, 0);
    //sphere.augmentV(-2000, 1);
    //sphere.augmentV(2000, 2);
    //sphere.sinH(2e-7);
    //sphere.augmentV(-2000, 1);
    sphere.augmentVorticity(1e-4, 0.02,-0.02);
    printf("end load config\n");
    printf("triangle size %d\n", sphere.triangles.size());
    printf("points size %d\n", sphere.points.size());
    particles.initParameter(arg_map);
    particles.initAttribute(
        sphere.points, sphere.v, sphere.m,
        sphere.h, sphere.normals, sphere.Vol,
        sphere.GM, sphere.vo, sphere.maxPosition,
        sphere.minPosition
    );
    

    particles.output_triangle(sphere.triangles);
    
    particles.output();
    real dt = init_dt;
    for(int i=0;i< frames_number;i++){
        printf("Update to frame %d\n", i + 1);
        particles.update(dt);
        particles.output();
    }
}
