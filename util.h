#ifndef _UTIL_TYPE_H_
#define _UTIL_TYPE_H_

#include <cstdlib>
#include <vector>
#include <random>

#define EPS 1e-8
#define DEBUG 0

using namespace std;

typedef vector<vector<double> > DoubleVector2d;
typedef vector<DoubleVector2d> DoubleVector3d;
typedef vector<DoubleVector3d> DoubleVector4d;

struct Neuron {
    double u;
    double z;
    bool is_valid;
};

typedef vector<vector<struct Neuron> > NeuronVector2d;
typedef vector<NeuronVector2d> NeuronVector3d;
typedef vector<NeuronVector3d> NeuronVector4d;


struct Weight {
  double val; // weight
  double lazy_sub; // sum of dEn/dWij of the datasets in one minibatch
  double gsum;
  int count; // number of mini-batch dataset
};

typedef vector<vector<struct Weight> > WeightVector2d;
typedef vector<WeightVector2d> WeightVector3d;
typedef vector<WeightVector3d> WeightVector4d;

inline double GenRandom(double fmin, double fmax) {
  double f = (double)rand() / RAND_MAX;
  return fmin + f * (fmax - fmin);
}

#endif
