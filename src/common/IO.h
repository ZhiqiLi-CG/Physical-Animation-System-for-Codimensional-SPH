#ifndef IO_H
#define IO_H
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <vector>
#include <list>
#include <queue>
#include <memory>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include "Common.h"
namespace ACG {
    template<int d, int originSide>
    class IO {
    public:
        Typedef_VectorDii(d);
        std::ifstream in;
        std::ofstream out;
        int num_frame = 0;
        Array<Frame<d, originSide>> inputFrames;
        Array< Eigen::Matrix<int, 3, 1>> triangles;
        void Init_Read(std::string read_file) {
            in = std::ifstream(read_file.c_str());
            if (!in.is_open())
                printf("the file:%s for reading is invalid!", read_file.c_str());
        }
        void Init_Write(std::string write_file) {
            out = std::ofstream(write_file.c_str(), std::ios::trunc);
            if (!out.is_open())
                printf("the file:%s for writing is invalid!", write_file.c_str());
        }
        void outputOneFrame(Frame<d, originSide> OneFrame) {
            printf("Output the %d frame and the points size is %d\n", num_frame, OneFrame.points.size());
            out << "frame " << num_frame++ << " " << OneFrame.points.size() << std::endl;
            for (int i = 0; i < OneFrame.points.size(); i++) {
                out << OneFrame.points[i](0) << " " << OneFrame.points[i](1);
                if constexpr (d == 3) out << " " << OneFrame.points[i](2);
                out << std::endl;
            }
            for (int i = 0; i < OneFrame.normal.size(); i++) {
                out << OneFrame.normal[i](0) << " " << OneFrame.normal[i](1);
                if constexpr (d == 3)  out << " " << OneFrame.normal[i](2);
                out << std::endl;
            }
            for (int i = 0; i < OneFrame.h.size(); i++) {
                out << OneFrame.h[i] << std::endl;
            }
        }
        void outputTriangles(Array<Eigen::Matrix<int, 3, 1>> trangle) {
            printf("Output the traigles and the triangls size is %d\n", trangle.size());
            out << trangle.size() << std::endl;
            for (int i = 0; i < trangle.size(); i++) {
                out << trangle[i][0] << " " << trangle[i][1] << " " << trangle[i][2] << std::endl;
            }
        }
        void readTriangles() {
            int num_traingle;
            in >> num_traingle;
            printf("Read the traigles and the triangls size is %d\n", num_traingle);
            for (int i = 0; i < num_traingle; i++) {
                Eigen::Matrix<int, 3, 1> triangle;
                in >> triangle[0] >> triangle[1] >> triangle[2];
                triangles.push_back(triangle);
            }
        }
        void readOneFrame() {
            std::string frame_str;
            int serial, num_frame;
            in >> frame_str >> serial >> num_frame;
            printf("Read %d frame and the points size is %d\n", serial, num_frame);
            //std::cout << frame_str << "????" << std::endl;
            Assert(frame_str == std::string("frame"), "wrong format for read file");
            Array<VectorD> points, normals;
            Array<real> hs;
            for (int i = 0; i < num_frame; i++) {
                VectorD newpoint;
                in >> newpoint[0] >> newpoint[1];
                if constexpr (d == 3) in >> newpoint[2];
                points.push_back(newpoint);
            }
            for (int i = 0; i < num_frame; i++) {
                VectorD normal;
                in >> normal[0] >> normal[1];
                if constexpr (d == 3) in >> normal[2];
                normals.push_back(normal);
            }
            for (int i = 0; i < num_frame; i++) {
                real h;
                in >> h;
                hs.push_back(h);
            }
            inputFrames.push_back(Frame<d, HOST>(points, normals, hs));
        }
        void readFile() {
            readTriangles();
            int i = 0;
            while (!in.eof()) {
                readOneFrame();
                i++;
                if (i == frame_number) break;

            }
        }
        ~IO() { in.close(); out.close(); }
    };
}
#endif