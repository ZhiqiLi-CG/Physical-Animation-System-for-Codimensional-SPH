#include <iostream>


#ifndef __Main_cpp__
#define __Main_cpp__

#include <stdlib.h>
#define GLUT_DISABLE_ATEXIT_HACK
#include <windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include "camera.h"
#include "view3d.h"
#include "Common.h"
#include "Algorithm.h"
#include "GeometryInit.h"
#include "constant.h"
#include "IO.h"
using namespace ACG;
Array<Vectord<3>> initPoints;
template<int d>
class DisplayData {
public:
    Typedef_VectorDii(d);
    Array<Frame<d, HOST>> inputFrames;
    Array< Eigen::Matrix<int, 3, 1>> triangles;
    IO<3, HOST> io;
    void readData(std::string fileName, InitSphere& global_sphere) {
        io.Init_Read(fileName);
        io.readFile();
        inputFrames = io.inputFrames;
        triangles = io.triangles;
        global_sphere.triangles = triangles;
    }
    void updateFrame(int index, InitSphere& global_sphere) {
        global_sphere.points = inputFrames[index].points;
        global_sphere.h = inputFrames[index].h;
    }
};
InitSphere global_sphere;
DisplayData<3> displayData;

template<int d> void draw()
{
    Typedef_VectorDii(d);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
   // glutSolidTeapot(1);
    int frame_index = frame;
    Array<VectorD> color = global_sphere.getSphereColor(camera);
    global_sphere.draw_triangles(color, initPoints);
/*    for (int i = 0; i < displayData.triangles.size(); i++) {
        Assert(displayData.triangles[i].size() == 3, "test triangle data readed's size");
        glBegin(GL_TRIANGLES);
        for (int j = 0; j < 3; j++) {
            int point_index = displayData.triangles[i][j];
            VectorD point = displayData.points[frame_index][point_index];
            glVertex3f(-0.5, -0.5, 0.0);
        }
        glEnd();
    }
*/
    glFlush();
}
void KeyboardKeys(unsigned char key, int x, int y)
{
    if (key == 'w')
        camera.WalkStraight(camera.speed);

    if (key == 's')
        camera.WalkStraight(-camera.speed);

    if (key == 'a')
        camera.WalkTransverse(camera.speed);

    if (key == 'd')
        camera.WalkTransverse(-camera.speed);
    if (key == '[') {
        frame--;
        int index_frame = frame;
        printf("update to frame:%d\n", index_frame);
        displayData.updateFrame(index_frame, global_sphere);
        printf("end update to frame:%d\n", index_frame);
    }
    if (key == ']') {
        frame++;
        int index_frame=frame;
        printf("update to frame:%d\n", index_frame);
        displayData.updateFrame(index_frame,global_sphere);
        printf("end update to frame:%d\n", index_frame);
    }
    glutPostRedisplay();
}
template<int d> void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    init();
    draw<d>();
}

void main(int argc, char* argv[])
{
    scanf("%d", &frame_number);
    std::string fileName = "D:\\ACG\\output.txt";
    printf("Wait, read the data,....\n");
    displayData.readData(fileName, global_sphere);
    displayData.updateFrame(frame, global_sphere);
    initPoints = global_sphere.points;
    printf("read finished, now begin to draw!\n");

    printf("Wait, init the data,....\n");
    //InitSphere sphere(1,0.1);
    //global_sphere = sphere;
    printf("read finished, now begin to draw!\n");

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutCreateWindow("bubble");
    glutDisplayFunc(display<3>);
    glutSpecialFunc(SpecialKeys);
    glutKeyboardFunc(KeyboardKeys);
    glutMainLoop();
}

#endif