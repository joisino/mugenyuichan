#include "layer.h"
#include "util.h"


void Layer::ChooseDropoutUnits(vector<struct Neuron> &input) {
  for( int i = 0; i < input.size(); i++ )
    input[i].is_valid = dropout_rate_ >= GenRandom( 0.0 , 1.0 );
}
