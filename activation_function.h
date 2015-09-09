#ifndef _ACTIVATION_FUNCTION_H_
#define _ACTIVATION_FUNCTION_H_

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <limits>
#include "util.h"
using namespace std;

class ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> const &neurons) {
        return -numeric_limits<double>::max();
    }

    virtual double CalculateDerivative(double u) {
        return -numeric_limits<double>::max();
    }
};

class LogisticSigmoid : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> const &neurons) {
        return 1/(1 + exp(-u));
    }

    virtual double CalculateDerivative(double u) {
        double fu = 1/(1 + exp(-u));
        return fu * (1 - fu);
    }
};

class TangentSigmoid : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> const &neurons) {
        return tanh(u);
    }

    virtual double CalculateDerivative(double u) {
        return (1 - tanh(u)*tanh(u));
    }
};

class RectifiedLinear : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> const &neurons) {
        return max(u, 0.0);
    }

    virtual double CalculateDerivative(double u) {
        if (u >= 0) return 1.0;
        else return 0.0;
    }
};

class Identity : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> const &neurons) {
        return u;
    }

    virtual double CalculateDerivative(double u) {
        return 1.0;
    }
};

class Softmax : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> const &neurons) {
        double denominator = 0;

	if( u > 30 ){
	  for( int i = 0; i < neurons.size(); i++ )
	    if( u+1 < neurons[i].u ) return 0;
	  return 1;
	}

        for (int i=0; i<neurons.size(); i++) {
            denominator += exp(neurons[i].u);
        }
        assert(denominator != 0.0);
        return (double)(exp(u)/denominator);
    }

    virtual double CalculateDerivative(double u) {
        assert(0);
        return -numeric_limits<double>::max();
    }
};

#endif
