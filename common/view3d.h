#ifndef __view_3d_h__
#define __view_3d_h__

#include <stdlib.h>
#define GLUT_DISABLE_ATEXIT_HACK
#include <windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include "camera.h"
#include <atomic>
std::atomic <int>   frame(0);
int window_width = 640;
int window_height = 480;
void draw();
Camera camera;
    void setProjection()
    {
        if (0 == window_height)
        {
            window_height = 1;
        }
        double ratio = 1.0 * window_width / window_height;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, ratio, 0.01, 100);
        glViewport(0, 0, window_width, window_height);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(camera.camera_x, camera.camera_y, camera.camera_z, camera.lookat_x, camera.lookat_y, camera.lookat_z, 0.0f, 1.0f, 0.0f);
    }

    void setLight()
    {
        GLfloat ambient[] = { 1.0,1.0,1.0,1.0 };
        GLfloat diffuse[] = { 1.0,1.0,1.0,1.0 };
        GLfloat position[] = { 10.0,10.0,0.0,1.0 };

        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
        glLightfv(GL_LIGHT0, GL_POSITION, position);

        glEnable(GL_LIGHT0);
        glEnable(GL_LIGHTING);
    }

    void init(void)
    {
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glDepthFunc(GL_LESS);
        glEnable(GL_DEPTH_TEST);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        setProjection();
        setLight();
    }

    void SpecialKeys(int key, int x, int y)
    {
        if (key == GLUT_KEY_UP)
            camera.PitchCamera(camera.angle);

        if (key == GLUT_KEY_DOWN)
            camera.PitchCamera(-camera.angle);

        if (key == GLUT_KEY_LEFT)
            camera.YawCamera(-camera.angle);

        if (key == GLUT_KEY_RIGHT)
            camera.YawCamera(camera.angle);

        glutPostRedisplay();
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
        }
        if (key == ']') {
            frame++;
        }
        glutPostRedisplay();
    }

#endif