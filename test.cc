#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <ctime>
#include <random>
#include <iomanip>
#include <iostream>
#include "util.h"
#include "neural_net.h"
#include "fully_connected_layer.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;


int rev( int x ){
  int res = 0;

  for( int i = 0; i < 4; i++ )
    res += ( (x>>(i*8)) & 255 ) << (24-i*8);

  return res;
}


double limg[60000][28*28+1];
int llabel[60000];
double timg[60000][28*28+1];
int tlabel[60000];

void TestMNISTAutoencoder(){
  mt19937 mt( time( NULL ) );
  
  int magic_number;
  int N, H, W;
  
  vector<double> in, out;
  DoubleVector2d ins, outs;

  unsigned char pixels[28*28];
  char outputfilename[100];

  
  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;
  Identity id;

  int hidden_neuron_n = 24;

  FullyConnectedLayer *full1 = new FullyConnectedLayer(28*28, hidden_neuron_n, &rel, 0.01, 0.9 , 0.9);
  FullyConnectedLayer *full2 = new FullyConnectedLayer(hidden_neuron_n, 28*28, &rel, 0.01, 0.9 , 0.5);
  
  srand(time(NULL));
  net.SetInputSize(28*28);
  net.AppendLayer( full1 );
  net.AppendLayer( full2 );  
  net.ConnectLayers();


  FILE *fp;

  fp = fopen( "train-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);

  fread( &H , sizeof( H ) , 1 , fp );
  H = rev(H);
  
  fread( &W , sizeof( W ) , 1 , fp );
  W = rev(W);

  for( int k = 0; k < N; k++ ){
    if( k % (N/10) == (N/10)-1) cerr << k << " / " << N << endl;
    for( int i = 0; i < H; i++ ){
      for( int j = 0; j < W; j++ ){
	unsigned char tmp;
	fread( &tmp , sizeof( tmp ) , 1 , fp );
	limg[k][i*W+j] = (double)tmp / 256.0;
      }
    }
  }

  fclose( fp );

  fp = fopen( "train-labels-idx1-ubyte" , "rb" );

  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);

  for( int k = 0; k < N; k++ ){
    unsigned char tmp;
    fread( &tmp , sizeof( tmp ) , 1 , fp );
    llabel[k] = int(tmp);
  }

  fclose( fp );

  
  int bloop = 0;
  while( ++bloop ){
    cerr << bloop << endl;
    for( int loop = 0; loop < 100; loop++ ){
      in.clear();
      out.clear();

      int img_n = mt()%N;
      while( llabel[img_n] != 4 ) img_n = mt()%N;


      for( int i = 0; i < H; i++ ){
	for( int j = 0; j < W; j++ ){
	  in.push_back( limg[img_n][i*W+j] - 0.7 + GenRandom( -0.05 , 0.05 ) );
	  out.push_back( limg[img_n][i*W+j] );	  
	}
      }
    
      ins.clear();
      ins.push_back( in );
      outs.clear();
      outs.push_back( out );
    
      net.TrainNetwork(ins, outs);
    }

    in.clear();
    out.clear(); out.resize(28*28);

    int img_n = mt()%N;
    while( llabel[img_n] != 4 ) img_n = mt()%N;    

    for( int i = 0; i < H; i++ )
      for( int j = 0; j < W; j++ )
	in.push_back( limg[img_n][i*W+j] - 0.7 );

    net.PropagateLayers( in , out );

    for( int i = 0; i < H*W; i++ )
      pixels[i] = (unsigned char)( min( 0.9999 , out[i] ) * 256.0 );

    sprintf( outputfilename , "output/img_%d_%d.png" , bloop , llabel[img_n] );

    stbi_write_png( outputfilename, 28, 28, 1, pixels, 28);


    if( bloop % 10 == 0 ){
      vector<struct Neuron> input_neurons(hidden_neuron_n);
      vector<struct Neuron> output_neurons(28*28);

      for( int i = 0; i < hidden_neuron_n; i++ ){
	double ave = full1->ave_[i];
	double sig = full1->sig_[i];
	normal_distribution<> norm( ave , sig );
	input_neurons[i].z = norm(mt);
      }

      full2->Propagate( input_neurons , output_neurons );
      full2->CalculateOutputUnits( output_neurons );

      for( int i = 0; i < H*W; i++ )
	pixels[i] = (unsigned char)( min( 0.9999 , output_neurons[i].z ) * 256.0 );
      
      sprintf( outputfilename , "generated/img_%d.png" , bloop/10 );

      stbi_write_png( outputfilename, 28, 28, 1, pixels, 28);
      
    }
    
  }

}


void TestYuiAutoencoder(){
  mt19937 mt( time( NULL ) );
  normal_distribution<> noise( 0.0 , 0.05 );
  
  int N = 1060;
  char filename[256];
  
  vector<double> in, out;
  DoubleVector2d ins, outs;

  int width, height, bpp;
  unsigned char pixels[96*96*3];
  char outputfilename[256];

  unsigned char* lpixels;
  
  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;
  Identity id;

  int hidden_neuron_n = 128;
  int input_n = 96*96*3;

  FullyConnectedLayer *full1 = new FullyConnectedLayer(input_n, hidden_neuron_n, &sigmoid, 0.01, 0.9 , 1.0);
  FullyConnectedLayer *full2 = new FullyConnectedLayer(hidden_neuron_n, input_n, &sigmoid, 0.01, 0.9 , 1.0);
  
  srand(time(NULL));
  net.SetInputSize( input_n );
  net.AppendLayer( full1 );
  net.AppendLayer( full2 );  
  net.ConnectLayers();

  FILE *fp = fopen( "aclog" , "w" );
  fclose( fp );

  int bloop = 0;
  while( ++bloop ){
    cerr << bloop << endl;
    for( int loop = 0; loop < 100; loop++ ){
      in.clear();
      out.clear();

      int img_n = mt()%N;
      sprintf( filename , "selected96/img_%d.png" , img_n );

      lpixels = stbi_load( filename , &width , &height , &bpp , 0 );


      for( int i = 0; i < input_n; i++ ){
	in.push_back( (double)lpixels[i] / 256.0 + noise(mt) );
	out.push_back( (double)lpixels[i] / 256.0 );	    
      }

      stbi_image_free (lpixels);      
    
      ins.clear();
      ins.push_back( in );
      outs.clear();
      outs.push_back( out );
    
      net.TrainNetwork(ins, outs);
    }


    in.clear();
    out.clear(); out.resize( input_n );

    int img_n = mt()%N;
    sprintf( filename , "selected96/img_%d.png" , img_n );

    lpixels = stbi_load( filename , &width , &height , &bpp , 0 );

    for( int i = 0; i < input_n; i++ )
      in.push_back( (double)lpixels[i] / 256.0 );

    stbi_image_free (lpixels);      

    net.PropagateLayers( in , out );

    double er = 0.0;
    for( int i = 0; i < input_n; i++ )
      er += fabs( out[i] - in[i] );

    cerr << er << endl;

    fp = fopen( "aclog" , "a" );
    fprintf( fp , "%lf\n" , er );
    fclose( fp );
    

    for( int i = 0; i < input_n; i++ )
      pixels[i] = (unsigned char)( min( 0.9999 , out[i] ) * 256.0 );

    sprintf( outputfilename , "output/img_%d.png" , bloop  );
    stbi_write_png( outputfilename, 96, 96, 3, pixels, 96*bpp);


    if( bloop % 1 == 0 ){
      for( int loop = 0; loop < 5; loop++ ){
	vector<struct Neuron> input_neurons(hidden_neuron_n);
	vector<struct Neuron> output_neurons(input_n);

	for( int i = 0; i < hidden_neuron_n; i++ ){
	  double ave = full1->ave_[i];
	  double sig = full1->sig_[i];
	  normal_distribution<> norm( 0 , 0.5 );
	  input_neurons[i].z = norm(mt);
	}

	full2->Propagate( input_neurons , output_neurons );
	full2->CalculateOutputUnits( output_neurons );

	for( int i = 0; i < input_n; i++ )
	  pixels[i] = (unsigned char)( min( 0.9999 , output_neurons[i].z ) * 256.0 );
      
	sprintf( outputfilename , "generated/img_%d_%d.png" , bloop, loop );

	stbi_write_png( outputfilename, 96, 96, 3, pixels, 96*bpp);
      }
      
    }

    if( bloop % 10 == 0 ){
      fp = fopen( "dat" , "w" );
      fclose( fp );
      full1->Write();
      full2->Write();
    }
    
  }

}

int main(){
  TestYuiAutoencoder();
  //TestMNISTAutoencoder();
}
