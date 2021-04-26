/* Defines a generalised feedforward neural network*/

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <iostream>
#include <bits/stdc++.h>
#include <math.h>
#include <Eigen/Dense>
#include <fstream>
#include <random>
#include <cmath>
#include <sstream>
#include <string>
using namespace std;
using namespace Eigen;

struct Batch{
	/* 
	 A Batch struct to store the batches of training data 
	*/
	MatrixXf inputs;
	VectorXf targets;
};


MatrixXf eye(int num){
	/*
	 Equivalent to numpy's eye function.
	 Creates an identity matrix of size num x num units.
	*/
	MatrixXf result(num,num);
	for(int i=0;i<num;i++){
		for(int j=0;j<num;j++){
			if(i==j) result(i,j) = 1;
			else result(i,j) = 0;
		}
	}
	return result;
}

MatrixXf random_normal(int input_units,int output_units){
	/*
	 Creates a matrix of size input_units x output_units with randomly selected elements from normal distribution.
	*/
	double scale = sqrt(2.0/(0.0+input_units+output_units));
	normal_distribution<double> distribution(0.0,scale);
	default_random_engine generator;
	MatrixXf result(input_units,output_units);
	for(int i=0;i<input_units;i++){
		for(int j=0;j<output_units;j++){
			result(i,j) = distribution(generator);
		}
	}
	return result;
}

MatrixXf zeros(int batchsize,int output_units){
	/*
	 Creates a zero matrix of size batchsize x output_units.
	*/
	MatrixXf result(batchsize,output_units);
	for(int i=0;i<batchsize;i++){
		for(int j=0;j<output_units;j++){
			result(i,j) = 0.0;
		}
	}
	return result;
}

MatrixXf mean(MatrixXf matrix){
	/*
	 Calculates columnwise mean of matrix.
	*/
	MatrixXf result(matrix.rows(),matrix.cols());
	for(int i=0;i<matrix.cols();i++){
		double sum = 0;
		for(int j=0;j<matrix.rows();j++){
			sum += matrix(j,i);
		}
		sum /= matrix.rows();
		for(int j=0;j<matrix.rows();j++){
			result(j,i) = sum;
		}
	}
	return result;
}

MatrixXf maximum(double num, MatrixXf input){
	/*
	 Calculates maximum of num and input element and returns the matrix.
	*/
	MatrixXf result(input.rows(),input.cols());
	for(int i=0;i<input.rows();i++){
		for(int j=0;j<input.cols();j++){
			double input_num = input(i,j);
			result(i,j) = max(num,input_num);
		}
	}
	return result;
}

MatrixXf compute_relu_grad(MatrixXf input){
	/*
	 Calculates gradiation of RELU for matrix input.
	*/
	MatrixXf result(input.rows(),input.cols());
	for(int i=0;i<input.rows();i++){
		for(int j=0;j<input.cols();j++){
			double temp = input(i,j);
			result(i,j) = temp>0?1:0;
		}
	}
	return result;
}

MatrixXf zeroes_like(MatrixXf input,VectorXf ref){
	/*
	 Similar to numpy zeroes like function.
	*/
	MatrixXf result(input.rows(),input.cols());
	for(int i=0;i<input.rows();i++){
		for(int j=0;j<input.cols();j++){
			result(i,j) = 0;
		}
		result(i,ref(i)) = -1;
	}
	return result;
}

MatrixXf elementwise(MatrixXf a, MatrixXf b){
	/*
	 Computes elementwise multiplication of matrices.
	*/
	MatrixXf result(a.rows(),a.cols());
	for(int i=0;i<a.rows();i++){
		for(int j=0;j<a.cols();j++){
			result(i,j) = a(i,j)*b(i,j);
		}
	}
	return result;
}

MatrixXf extend(MatrixXf a,int size){
	/*
	Extends a matrix to given size
	*/
	MatrixXf result(size,a.cols());
	for(int i=0;i<size;i++){
		for(int j=0;j<a.cols();j++){
			result(i,j) = a(0,j);
		}
	}
	return result;
}

MatrixXf compute_sigmoid (MatrixXf input){
	/*
	 Computes sigmoid of the matrix.
	*/
	MatrixXf result(input.rows(),input.cols());
		for(int i=0;i<input.rows();i++){
			for(int j=0;j<input.cols();j++){
				double temp = input(i,j);
				result(i,j) = (1.0 / (1.0 + exp(-1.0 * temp)));
				}
			}
		return result;
}

VectorXf softmax_crossentropy_with_logits(MatrixXf logits, VectorXf reference_answers){
	/*
	 Computes softmax crossentropy.
	*/
	VectorXf xentropy(logits.rows());
	for(int i=0;i<logits.rows();i++){
		double result = 0;
		for (int j=0;j<logits.cols();j++){
			result += exp(logits(i,j));
		}
		result = log(result);
		xentropy(i) = -1.0*logits(i,reference_answers(i)) + result; 
	}
	return xentropy;
}

MatrixXf grad_softmax_crossentropy_with_logits(MatrixXf logits, VectorXf reference_answers){
	/*
	 Computes gradient of softmax crossentropy.
	*/
	MatrixXf ones_for_answers = zeroes_like(logits,reference_answers);
	MatrixXf softmax(logits.rows(),logits.cols());

	for(int i=0;i<softmax.rows();i++){
		double result = 0;
		for(int j=0;j<softmax.cols();j++){
			softmax(i,j) = exp(logits(i,j));
			result += softmax(i,j);
		}
		for(int j=0;j<softmax.cols();j++){
			softmax(i,j) /= result;
		}
	}

	return (ones_for_answers + softmax)/logits.rows(); 
}

VectorXf row_max(MatrixXf logits){
	/*
	 Calculates row maximum vector for the matrix.
	*/
	VectorXf result(logits.rows());
	for(int i = 0; i<logits.rows();i++){
		int max = 0;
		for(int j = 0;j < logits.cols();j++){
			if(logits(i,j)>max) max = j;
		}
		result(i) = max;
	}
	return result;
}

vector<int> random_permutation(int n){
	/*
	 Generates random permutation of integers from 0 to n.
	*/
	vector<int> result;
	for(int i=0;i<n;i++) result.push_back(i);
	random_shuffle(result.begin(),result.end());
	return result;
}

class Layer{
	/*
	 Layer object.
	*/
	public:
		virtual MatrixXf forward(MatrixXf input) {
			return input;
		}
		virtual MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			int num_units = input.cols();
			MatrixXf d_layer_d_input = eye(num_units);
			return grad_output*d_layer_d_input;
		}
};

class DenseLayer: public Layer{
	/*
	 DenseLayer object.
	*/
	public:
		double learning_rate;
		MatrixXf weights;
		MatrixXf biases;
	public:
		
		DenseLayer(int input_units,int output_units, double rate=0.1){
			learning_rate = rate;
			weights = random_normal(input_units,output_units);
			biases = zeros(1,output_units);
		}

		MatrixXf forward(MatrixXf input){
			biases = extend(biases,input.rows());
			return input*weights + biases;
		}
		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			MatrixXf grad_input = grad_output * weights.transpose();
			MatrixXf grad_weights = input.transpose() * grad_output;
			MatrixXf grad_biases = mean(grad_output)*input.rows();
			weights = weights - learning_rate*grad_weights;
			biases = biases - learning_rate*grad_biases;
			return grad_input;
		}

};

class ReLU: public Layer{
	/*
	 RELU Layer object.
	*/
	public:

		MatrixXf forward(MatrixXf input){
			MatrixXf relu_forward = maximum(0.0,input);
			return relu_forward;
		}

		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			MatrixXf relu_grad = compute_relu_grad(input);
			return elementwise(grad_output,relu_grad);
		}
};

class Sigmoid: public Layer{
	/*
	 Logistic Layer object.
	*/
	public:

		MatrixXf forward(MatrixXf input){
			MatrixXf sigmoid_forward=compute_sigmoid(input);
			return sigmoid_forward;
		}

		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			MatrixXf y = compute_sigmoid(input);
			MatrixXf result(y.rows(),y.cols());
			for(int i=0;i<y.rows();i++){
				for(int j=0;j<y.cols();j++){
					double temp = y(i,j);
					result(i,j) = temp*(1.0-temp);
				}
			}
			return elementwise(grad_output,result);
		}
};

class Tanh: public Layer{
	/*
	 Tanh Layer object.
	*/
	public:
		MatrixXf forward(MatrixXf input){
			MatrixXf result(input.rows(),input.cols());
			for(int i=0;i<input.rows();i++){
				for(int j=0;j<input.cols();j++){
					double temp = input(i,j);
					result(i,j) = tanh(temp);
				}
			}
			return result;
		}

		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			MatrixXf result(input.rows(),input.cols());
			for(int i=0;i<input.rows();i++){
				for(int j=0;j<input.cols();j++){
					double temp = input(i,j);
					result(i,j) = 1-(tanh(temp)*tanh(temp));
				}
			}
			return elementwise(grad_output,result);
		}
};

vector<MatrixXf> forward(vector<Layer *> network, MatrixXf X){
	/*
	 Forward propagation of the neural network.
	*/
	vector<MatrixXf> activations;
	MatrixXf input = X;
	for(int i=0;i<network.size();i++){
		activations.push_back(network[i]->forward(input));
		input = activations[activations.size()-1];
	}
	return activations;
}

VectorXf predict(vector<Layer *> network, MatrixXf X){
	/*
	 Predicts the given model output on basis of given input.
	*/
	vector<MatrixXf> result = forward(network,X);
	MatrixXf logits = result[result.size()-1];
	return row_max(logits);
}

double train(vector<Layer *> network, MatrixXf X, VectorXf y){
	/*
	 Trains a given network with given training data.
	*/
	vector<MatrixXf> layer_activations = forward(network,X);
	vector<MatrixXf> layer_inputs;
	layer_inputs.assign(layer_activations.begin(),layer_activations.end());
	layer_inputs.insert(layer_inputs.begin(),X);
	MatrixXf logits = layer_activations[layer_activations.size()-1];
	VectorXf loss = softmax_crossentropy_with_logits(logits,y);
	MatrixXf loss_grad = grad_softmax_crossentropy_with_logits(logits,y);
	for(int i=network.size()-1;i>=0;i--){
		MatrixXf loss_grad1 = network[i]->backward(layer_inputs[i],loss_grad);
		loss_grad = loss_grad1;
	}
	double sum = 0;
	for(int i=0;i<loss.rows();i++){
		sum += loss(i);
	}
	return sum/loss.rows();

}

vector<Batch> iterate_minibatches(MatrixXf inputs, VectorXf targets, int batchsize){
	/*
	 Divides given data into minibatches of given batchsize.
	*/
	vector<int> indices = random_permutation(inputs.rows());
	int idx = 0;
	vector<Batch> result;
	while(idx<inputs.rows()){
		struct Batch batch;
		MatrixXf input = MatrixXf::Zero(batchsize,inputs.cols());
		VectorXf target = VectorXf::Zero(batchsize); 
		int j;
		for(j=idx;j<idx+batchsize && j <inputs.rows();j++){
			input.row(j-idx) = inputs.row(indices[j]);
			target.row(j-idx) = targets.row(indices[j]);
		}
		if(j==idx+batchsize) idx+=batchsize;
		else idx = inputs.rows();
		batch.inputs = input;
		batch.targets = target;
		result.push_back(batch);
	}
	return result;
}

vector<Layer *> build_network(int inputsize, int outputsize, int hiddenlayercount, vector<int> topology, int activationmode){
	// Builds the multilayer perceptron.
	vector<Layer *> network;
	// Define network
	network.push_back(new DenseLayer(inputsize,topology[0]));
	if(activationmode==1) network.push_back(new ReLU());
	else if(activationmode==2) network.push_back(new Sigmoid());
	else network.push_back(new Tanh());
	network.push_back(new Tanh());
	for(int i=0;i<hiddenlayercount-1;i++){
		network.push_back(new DenseLayer(topology[i],topology[i+1]));
		if(activationmode==1) network.push_back(new ReLU());
		else if(activationmode==2) network.push_back(new Sigmoid());
		else network.push_back(new Tanh());
	}
	network.push_back(new DenseLayer(topology[hiddenlayercount-1],outputsize));
	return network;
}