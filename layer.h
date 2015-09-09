#ifndef _LAYER_H_
#define _LAYER_H_

#include <cassert>
#include <vector>
#include "activation_function.h"
#include "util.h"
using namespace std;

class Layer {
public:
 Layer() : f_(NULL), dropout_rate_(1.0) {}
 Layer( ActivationFunction *f, double dropout_rate ) 
            : f_(f)
	    , dropout_rate_(dropout_rate){
        assert(0.0 <= dropout_rate && dropout_rate <= 1.0);
    }

    virtual ~Layer() {}
    virtual void CheckInputUnits(vector<struct Neuron> const &units) { assert(0); }
    virtual void ArrangeOutputUnits(vector<struct Neuron> &units) { assert(0); }
    virtual void ConnectNeurons(vector<struct Neuron> const &input, vector<struct Neuron> const &output) { assert(0); }
    virtual void ChooseDropoutUnits(vector<struct Neuron> &input);
    virtual void CalculateOutputUnits(vector<struct Neuron> &units) { assert(0); }
    virtual void Propagate(vector<struct Neuron> const &input, vector<struct Neuron> &output) { assert(0); }
    virtual void BackPropagate(vector<struct Neuron> const &input, vector<double> const &next_delta, ActivationFunction *f, vector<double> &delta) { assert(0); }
    virtual void UpdateLazySubtrahend(vector<struct Neuron> const &input, const vector<double> &next_delta) { assert(0); }
    virtual void ApplyLazySubtrahend() { assert(0); }

    ActivationFunction *f_;
    double dropout_rate_;    
private:

};

#endif
