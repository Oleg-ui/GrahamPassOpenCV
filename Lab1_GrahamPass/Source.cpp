#include <iostream>

#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <random>
#include <vector>
#include <algorithm>

struct point {
  int x;
  int y;
};

std::vector<point> RandomVector(int size) {
  bool flag = false;
  std::vector<point> vec_tmp(size);
  std::mt19937 gen;

  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < size; i++) {
    vec_tmp[i].x = gen() % 1700 + 100;
    vec_tmp[i].y = gen() % 900 + 100;
  }
  while (!flag) {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (i == j) {
          continue;
        }
        if (vec_tmp[i].x == vec_tmp[j].x && vec_tmp[i].y == vec_tmp[j].y) {
          vec_tmp[i].x = gen() % 1700 + 100;
          vec_tmp[i].y = gen() % 900 + 100;
          i = 0;
          j = 0;
        }
      }
    }
    flag = true;
  }
  return vec_tmp;
}

double Rotation(point a, point b, point c) {
  return((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x));
}

void GrahamPass(std::vector<point>& basis_vector, int* count, std::vector<size_t>& result) {
  if (basis_vector.size() == 1)  return;

  std::vector<size_t> tmp_vector;

  size_t min = 0;

  for (int i = 1; i < basis_vector.size(); i++) {
    min = basis_vector.at(min).x > basis_vector.at(i).x || (basis_vector.at(min).x == basis_vector.at(i).x && basis_vector.at(min).y > basis_vector.at(i).y) ? i : min;
  }

  tmp_vector.push_back(min);

  for (int i = 0; i < basis_vector.size(); i++) {
    if (i == min) {
      continue;
    }
    tmp_vector.push_back(i);
  }

  for (int i = 2; i < basis_vector.size(); i++) {
    int j = i;
    while (j > 1 && Rotation(basis_vector.at(min), basis_vector.at(tmp_vector.at(j - 1)), basis_vector.at(tmp_vector.at(j))) < 0) {
      std::swap(tmp_vector.at(j), tmp_vector.at(j - 1));
      j--;
    }
  }

  std::vector<size_t> s;

  s.push_back(tmp_vector.at(0));
  s.push_back(tmp_vector.at(1));

  for (int i = 2; i < basis_vector.size(); i++) {
    while (Rotation(basis_vector.at(s.at(s.size() - 2)), basis_vector.at(s.at(s.size() - 1)), basis_vector.at(tmp_vector.at(i))) <= 0) {
      s.pop_back();
    }
    s.push_back(tmp_vector.at(i));
  }

  *count = s.size();
  for (int i = 0; i < s.size(); i++) {
    result.push_back(s[i]);
  }
}

int main() {
  setlocale(LC_ALL, "RUS");
  int c = 0;
  int count = 0;
  int size = 1924;
  std::vector<point> vec = RandomVector(size);

  std::vector<size_t> graham;
  GrahamPass(vec, &count, graham);

  cv::Mat image = cv::imread("White1920x1080.jpg", 0);
  for (int i = 0; i < size; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        image.at<uchar>(vec[i].y + j, vec[i].x + k) = 0;
      }
    }
    c++;
  }

  for (size_t i = 0; i < graham.size() - 1; i++) {
    line(image, cv::Point(vec[graham[i]].x, vec[graham[i]].y), cv::Point(vec[graham[i + 1]].x, vec[graham[i + 1]].y), cv::Scalar(0, 0, 255), 1);

  }
  line(image, cv::Point(vec[graham[0]].x, vec[graham[0]].y), cv::Point(vec[graham[graham.size() - 1]].x, vec[graham[graham.size() - 1]].y), cv::Scalar(0, 0, 255), 1);

  std::cout << "c = " << c << std::endl;

  cv::imshow("Original image", image);
  cv::waitKey(0);
}