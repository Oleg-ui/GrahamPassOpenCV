#include <iostream>

#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <random>
#include <vector>
#include <omp.h>

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

std::vector<size_t> GrahamPassSeq(const std::vector<point>& basis_vec, int* count) {
  std::vector<size_t> result;

  if (basis_vec.size() == 1)  return result;

  std::vector<int> vec_tmp;

  int min = 0;

  for (int i = 1; i < static_cast<int>(basis_vec.size()); i++) {
    min = basis_vec.at(min).x > basis_vec.at(i).x ||
      (basis_vec.at(min).x == basis_vec.at(i).x &&
        basis_vec.at(min).y > basis_vec.at(i).y) ? i : min;
  }

  vec_tmp.push_back(min);

  for (int i = 0; i < static_cast<int>(basis_vec.size()); i++) {
    if (i == min) {
      continue;
    }
    vec_tmp.push_back(i);
  }

  for (int i = 2; i < static_cast<int>(basis_vec.size()); i++) {
    int j = i;
    while (j > 1 && Rotation(basis_vec.at(min), basis_vec.at(vec_tmp.at(j - 1)),
      basis_vec.at(vec_tmp.at(j))) < 0) {
      std::swap(vec_tmp.at(j), vec_tmp.at(j - 1));
      j--;
    }
  }

  result.push_back(vec_tmp.at(0));
  result.push_back(vec_tmp.at(1));

  for (int i = 2; i < static_cast<int>(basis_vec.size()); i++) {
    while (Rotation(basis_vec.at(result.at(result.size() - 2)),
      basis_vec.at(result.at(result.size() - 1)),
      basis_vec.at(vec_tmp.at(i))) <= 0) {
      result.pop_back();
    }
    result.push_back(vec_tmp.at(i));
  }

  *count = static_cast<int>(result.size());

  return result;
}

std::vector<size_t> GrahamPassOmp(std::vector<point>& basis_vec,
  int* count) {
  std::vector<size_t> result;
  std::vector<point> result_point;

  int block = static_cast<int>(basis_vec.size()) / omp_get_max_threads();
  int remainder = static_cast<int>(basis_vec.size()) % omp_get_max_threads();

  if (4 * omp_get_max_threads() >= static_cast<int>(basis_vec.size())) {
    result = GrahamPassSeq(basis_vec, count);
    return result;
  }

  //=====================================================================================================================

  std::vector<int> vec_tmp;

  int min = 0;

  for (int i = 1; i < static_cast<int>(basis_vec.size()); i++) {
    min = basis_vec.at(min).x > basis_vec.at(i).x ||
      (basis_vec.at(min).x == basis_vec.at(i).x &&
        basis_vec.at(min).y > basis_vec.at(i).y) ? i : min;
  }

  vec_tmp.push_back(min);

  for (int i = 0; i < static_cast<int>(basis_vec.size()); i++) {
    if (i == min) {
      continue;
    }
    vec_tmp.push_back(i);
  }

  for (int i = 2; i < static_cast<int>(basis_vec.size()); i++) {
    int j = i;
    while (j > 1 && Rotation(basis_vec.at(min), basis_vec.at(vec_tmp.at(j - 1)),
      basis_vec.at(vec_tmp.at(j))) < 0) {
      std::swap(vec_tmp.at(j), vec_tmp.at(j - 1));
      j--;
    }
  }

  for (int i = 0; i < static_cast<int>(vec_tmp.size()); i++) {
    basis_vec[i] = basis_vec[vec_tmp[i]];
  }

  //=====================================================================================================================

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    std::vector<point> result_thread;
    std::vector<size_t> result_tmp;
    

#pragma omp critical
    {
      if (id == 0) {
        result_thread.reserve(block + remainder);
        result_thread.insert(result_thread.begin(), basis_vec.begin(), basis_vec.begin() + block + remainder);
      }
      else {
        result_thread.reserve(basis_vec.size() / omp_get_num_threads());
        result_thread.insert(result_thread.begin(), basis_vec.begin() + remainder + id * block, basis_vec.begin() + remainder + id * block + block);
      }
    }

#pragma omp barrier

    result_tmp = GrahamPassSeq(result_thread, count);

#pragma omp critical
    {
      for (size_t result_from_tmp : result_tmp) {
        result_point.push_back(result_thread[result_from_tmp]);
      }
    }
  }
  result = GrahamPassSeq(result_point, count);
  return result;
}

/*void GrahamPass(std::vector<point>& basis_vector, int* count, std::vector<size_t>& result) {
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
    while (s.size() > 1 && Rotation(basis_vector.at(s.at(s.size() - 2)), basis_vector.at(s.at(s.size() - 1)), basis_vector.at(tmp_vector.at(i))) <= 0 ||
      Rotation(basis_vector.at(s.front()), basis_vector.at(s.back()), basis_vector.at(tmp_vector.at(i))) <= 0) {
      s.pop_back();
    }
    s.push_back(tmp_vector.at(i));
  }

  *count = s.size();
  for (int i = 0; i < s.size(); i++) {
    result.push_back(s[i]);
  }
}*/

int main() {
  int c = 0;
  int count = 0;
  int size = 9000;
  std::vector<point> vec = RandomVector(size);
  //std::vector<point> vec = { {10, 10}, {10, 40}, {40, 10}, {40, 40}, {20, 20} };
  std::vector<size_t> graham;
  

  cv::Mat image = cv::imread("White1920x1080.jpg", 0);
  for (int i = 0; i < size; i++) {
    for (int j = -1; j < 2; j++) {
      for (int k = -1; k < 2; k++) {
        image.at<uchar>(vec[i].y + j, vec[i].x + k) = 0;
      }
    }
    c++;
  }
  graham = GrahamPassOmp(vec, &count);
  
  
  for (size_t i = 0; i < graham.size() - 1; i++) {
    line(image, cv::Point(vec[graham[i]].x, vec[graham[i]].y), cv::Point(vec[graham[i + 1]].x, vec[graham[i + 1]].y), cv::Scalar(0, 0, 255), 1);

  }
  line(image, cv::Point(vec[graham[0]].x, vec[graham[0]].y), cv::Point(vec[graham[graham.size() - 1]].x, vec[graham[graham.size() - 1]].y), cv::Scalar(0, 0, 255), 1);
  
  std::cout << "c = " << c << std::endl;
  std::cout << "graham size = " << graham.size() << std::endl;
  
  cv::imshow("Original image", image);
  cv::waitKey(0);
}