#include "fully_connected_layer.h"
#include "iostream"
#include <iomanip>

FullyConnectedLayer::FullyConnectedLayer(int num_input, int num_output, ActivationFunction *f, double learning_rate, double momentum, double dropout_rate) 
        : neuron_connected_(false),
          num_input_(num_input),
          num_output_(num_output),
          learning_rate_(learning_rate),
	  momentum_(momentum),
          Layer(f, dropout_rate) {  
  bufsize_ = 1024;
  bufp_ = 0;
  linkbuf_.resize( num_output , vector<double>( bufsize_ , 0.0 ) );
  sum_ = sqsum_ = ave_ = sig_ = vector<double>( num_output , 0.0 );
	  }


void FullyConnectedLayer::CheckInputUnits(vector<struct Neuron> const &units) {
    assert(units.size() == num_input_);
}

void FullyConnectedLayer::ArrangeOutputUnits(vector<struct Neuron> &units) {
    units.resize(num_output_);
}

void FullyConnectedLayer::ConnectNeurons(
        vector<struct Neuron> const &input, 
        vector<struct Neuron> const &output) {
    assert(!neuron_connected_);
    assert(input.size() == num_input_);
    assert(output.size() == num_output_);
    
    biases_.resize(num_output_);
    for (int i=0; i<num_output_; i++) {
        struct Weight w;

        w.val = 0.0;//GenRandom(0.0, 0.1);
        w.lazy_sub = 0.0;
        w.count = 0;
	w.gsum = 0.0;
        biases_[i] = w;
    }

    double lim = 1.0 / num_input_; //1.0 / sqrt( num_input_ );
    weights_.resize(num_input_);
    for (int i=0; i<num_input_; i++) {
        weights_[i].resize(num_output_);
        for (int j=0; j<num_output_; j++) {
            struct Weight w;

            w.val = GenRandom(0.0, lim);
            w.lazy_sub= 0.0;
            w.count = 0;
	    w.gsum = 0.0;
            weights_[i][j] = w;
        }
    }

    neuron_connected_ = true;
}

void FullyConnectedLayer::CalculateOutputUnits(vector<struct Neuron> &units) {
    assert(units.size() == num_output_);

    double outmax = -1000;
    double outmin = 1000;

    for (int i=0; i<num_output_; i++) {
        units[i].z = f_->Calculate(units[i].u, units);

	sum_[i]   += units[i].z - linkbuf_[i][bufp_];
	sqsum_[i] += units[i].z*units[i].z - linkbuf_[i][bufp_]*linkbuf_[i][bufp_];
	linkbuf_[i][bufp_] = units[i].z;
	ave_[i] = sum_[i] / bufsize_;
	sig_[i] = sqsum_[i] / bufsize_ - ave_[i] * ave_[i];

	outmax = max( outmax , units[i].z );
	outmin = min( outmin , units[i].z );
    }

    bufp_ = (bufp_+1)%bufsize_;

    //cout << "fullyout : " << setprecision(9) << outmax << " " << outmin << endl;
    
#if DEBUG
    printf( "fullyout : %lf %lf\n" , outmax , outmin );
#endif
}

void FullyConnectedLayer::Propagate(
        vector<struct Neuron> const &input, 
        vector<struct Neuron> &output) {
    assert(input.size() == num_input_);
    assert(output.size() == num_output_);

    double outmax = -1000;
    double outmin = 1000;
    
    for (int i=0; i<num_output_; i++) {
        output[i].u = 0.0;
    }

    for (int i=0; i<num_input_; i++) {
        for (int j=0; j<num_output_; j++) {
            struct Weight w = weights_[i][j];
            output[j].u += w.val * input[i].z;
        }
    }


    assert(biases_.size() == num_output_);
    for (int i=0; i<num_output_; i++) {
        output[i].u += biases_[i].val;

	outmax = max( outmax , output[i].u );
	outmin = min( outmin , output[i].u );	
    }

    cout << "fully : " << setprecision(9) << outmax << " " << outmin << endl;
	
    
#if DEBUG
    printf( "fully : %lf %lf\n" , outmax , outmin );
#endif
    
}

void FullyConnectedLayer::BackPropagate(
        vector<struct Neuron> const &input,
        vector<double> const &next_delta,
        ActivationFunction *f,
        vector<double> &delta) {
    assert(input.size() == num_input_);
    assert(next_delta.size() == num_output_);
    assert(weights_.size() == num_input_);

    double deltamax = -1000;
    double deltamin = 1000;
    
    delta.resize(num_input_);
    for (int i=0; i<num_input_; i++) {
        delta[i] = 0.0;

        assert(weights_[i].size() == num_output_);
        for (int j=0; j<num_output_; j++) {
            double w = weights_[i][j].val;

            delta[i] += 
                next_delta[j] * w * f->CalculateDerivative(input[i].u);

	    deltamax = max( deltamax , delta[i] );
	    deltamin = min( deltamin , delta[i] );	    
        }
    }

#if DEBUG
    printf( "fullydelta : %lf %lf\n" , deltamax , deltamin );
#endif
}

void FullyConnectedLayer::UpdateLazySubtrahend(
        vector<struct Neuron> const &input,
        vector<double> const &next_delta) {
    assert(input.size() == num_input_);
    assert(next_delta.size() == num_output_);

    assert(weights_.size() == num_input_);
    for (int i=0; i<num_input_; i++) {
        assert(weights_[i].size() == num_output_);
        for (int j=0; j<num_output_; j++) {
            Weight &w = weights_[i][j];
            w.lazy_sub += next_delta[j] * input[i].z;
            w.count++;
        }
    }

    assert(biases_.size() == num_output_);
    for (int i=0; i<num_output_; i++) {
        Weight &w = biases_[i];
        w.lazy_sub += next_delta[i];
        w.count++;
    }
}

void FullyConnectedLayer::ApplyLazySubtrahend() {
    assert(weights_.size() == num_input_);
    for (int i=0; i<num_input_; i++) {
        assert(weights_[i].size() == num_output_);
        for (int j=0; j<num_output_; j++) {
            Weight &w = weights_[i][j];

            assert(w.count > 0);


	       double prevdelta = - w.lazy_sub * learning_rate_ / w.count + momentum_ * w.gsum;
	       w.val += prevdelta;
	       w.gsum = prevdelta;

	       /*
	    w.gsum += (w.lazy_sub / w.count) * (w.lazy_sub / w.count);
	    w.val -= learning_rate_ / sqrt( w.gsum ) * w.lazy_sub / w.count;
	       */

	    //printf( "%lf\n" , learning_rate_ / sqrt( w.gsum ) * w.lazy_sub / w.count );

	    w.lazy_sub = 0.0;
            w.count = 0;
        }
    }

    //printf( "kokomadefull/////////////////////////////////////\n" );

    assert(biases_.size() == num_output_);

    for (int i=0; i<num_output_; i++) {
        Weight &w = biases_[i];

        assert(w.count > 0);

	   double prevdelta = - w.lazy_sub * learning_rate_ / w.count + momentum_ * w.gsum;
	   w.val += prevdelta;
	   w.gsum = prevdelta;
	   /*
	w.gsum += (w.lazy_sub / w.count) * (w.lazy_sub / w.count);
	w.val -= learning_rate_ / sqrt( w.gsum ) * w.lazy_sub / w.count;
	   */

	w.lazy_sub = 0.0;
        w.count = 0;
    }

    double w_n = num_input_*num_output_;
    double lim = 1.0 / num_input_;
    double limsum = lim*w_n;
    double sumw = 0.0;

    for (int i=0; i<num_input_; i++)
        for (int j=0; j<num_output_; j++)
	  sumw += weights_[i][j].val;
    sumw = fabs( sumw );
    
    if( sumw > limsum ){
      for (int i=0; i<num_input_; i++)
        for (int j=0; j<num_output_; j++)
	  weights_[i][j].val *= limsum / sumw;
    }


}

void FullyConnectedLayer::Write() {
  FILE *fp = fopen( "dat" , "a" );
  
  for( int i = 0; i < weights_.size(); i++ ){
    for( int j = 0; j < weights_[i].size(); j++ ){
      fprintf( fp , "%lf " , weights_[i][j] );
    }
    fprintf( fp , "\n" );
  }

  for( int i = 0; i < biases_.size(); i++ )
    fprintf( fp , "%lf " , biases_[i] );
  fprintf( fp , "\n" );

  for( int i = 0; i < ave_.size(); i++ )
    fprintf( fp , "%lf %lf\n" , ave_[i] , sig_[i] );
  
  fclose( fp );
}
