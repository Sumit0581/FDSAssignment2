#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <iostream>
#include <bits/stdc++.h>
#include <math.h>
#include <Eigen/Dense>
#include <fstream>
#include<random>
// #include "matplotlibcpp.h"
#include <cmath>
#include <sstream>
#include <string>
using namespace std;
using namespace Eigen;

struct Batch{
	MatrixXf inputs;
	VectorXf targets;
};


MatrixXf eye(int num){
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
	double scale = sqrt(2.0/(0.0+input_units+output_units));
	normal_distribution<double> distribution(0.0,scale);
	default_random_engine generator;
	MatrixXf result(input_units,output_units);
	for(int i=0;i<input_units;i++){
		for(int j=0;j<output_units;j++){
			result(i,j) = distribution(generator);
		}
	}
	// cout << result << endl;
	return result;
}

MatrixXf zeros(int batchsize,int output_units){
	MatrixXf result(batchsize,output_units);
	for(int i=0;i<batchsize;i++){
		for(int j=0;j<output_units;j++){
			result(i,j) = 0.0;
		}
		
	}
	return result;
}

MatrixXf mean(MatrixXf matrix){
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
	MatrixXf result(a.rows(),a.cols());
	for(int i=0;i<a.rows();i++){
		for(int j=0;j<a.cols();j++){
			result(i,j) = a(i,j)*b(i,j);
		}
	}
	return result;
}

MatrixXf extend(MatrixXf a,int size){
	MatrixXf result(size,a.cols());
	for(int i=0;i<size;i++){
		for(int j=0;j<a.cols();j++){
			result(i,j) = a(0,j);
		}
	}
	return result;
}

class Layer{
	public:
		virtual MatrixXf forward(MatrixXf input) {
			return input;
		}
		virtual MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			// cout << "Backward OG entered" << endl;
			int num_units = input.cols();
			MatrixXf d_layer_d_input = eye(num_units);
			return grad_output*d_layer_d_input;
		}
};

class DenseLayer: public Layer{
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
			// cout << "Forward entered" << endl;
			// cout << "Input: " << input.rows() << " " << input.cols() << endl;
			// cout << "weight: " << weights.rows() << " " << weights.cols() << endl;
			// cout << "biases: " << biases.rows() << " " << biases.cols() << endl;
			return input*weights + biases;
			// cout << "Forward finished" << endl;
		}
		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			// cout << "Backward entered" << endl;
			MatrixXf grad_input = grad_output * weights.transpose();
			MatrixXf grad_weights = input.transpose() * grad_output;
			MatrixXf grad_biases = mean(grad_output)*input.rows();
			// cout << grad_weights.rows() << " " << grad_weights.cols() << endl;
			// cout << weights.rows() << " " << weights.cols() << endl;
			// cout << grad_biases.rows() << " " << grad_biases.cols() << endl;
			// cout << biases.rows() << " " << biases.cols() << endl;
			// cout << "Backward initialisation is correct" << endl;
			weights = weights - learning_rate*grad_weights;
			biases = biases - learning_rate*grad_biases;

			return grad_input;
		}

};

class ReLU: public Layer{
	public:

		MatrixXf forward(MatrixXf input){
			MatrixXf relu_forward = maximum(0.0,input);
			return relu_forward;
		}

		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			// cout << "Backward not OG entered" << endl;
			MatrixXf relu_grad = compute_relu_grad(input);
			// cout << relu_grad.rows() << " " << relu_grad.cols() << endl;
			// cout << grad_output.rows() << " " << grad_output.cols() << endl;
			return elementwise(grad_output,relu_grad);
		}
};
MatrixXf compute_sigmoid (MatrixXf input){
	MatrixXf result(input.rows(),input.cols());
		for(int i=0;i<input.rows();i++){
			for(int j=0;j<input.cols();j++){
				double temp = input(i,j);
				result(i,j) = (1 / (1 + exp(- temp)));
				}
			}
		return result;
}

class Sigmoid: public layer{

	public:
			
			MatrixXf forward(MatrixXf input){
				MatrixXf sigmoid_forward=compute_sigmoid(input);
				return sigmoid_forward;
			}
			MatrixXf backward(MatrixXf input, MatrixXf grad_output){
				MatrixXf y=compute_sigmoid(input);
				MatrixXf result(y.rows(),y.cols());
				for(int i=0;i<y.rows();i++){
					for(int j=0;j<y.cols();j++){
					double temp = y(i,j);
					result(i,j) = y*(1-y);
					}
					}
				return elementwise(grad_output,result);
			}
};
class my_tanh : public Layer{
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

VectorXf softmax_crossentropy_with_logits(MatrixXf logits, VectorXf reference_answers){
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

vector<MatrixXf> forward(vector<Layer *> network, MatrixXf X){
	vector<MatrixXf> activations;
	MatrixXf input = X;
	for(int i=0;i<network.size();i++){
		activations.push_back(network[i]->forward(input));
		input = activations[activations.size()-1];
	}
	return activations;
}

VectorXf row_max(MatrixXf logits){
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

VectorXf predict(vector<Layer *> network, MatrixXf X){
	vector<MatrixXf> result = forward(network,X);
	MatrixXf logits = result[result.size()-1];
	return row_max(logits);
}

double train(vector<Layer *> network, MatrixXf X, VectorXf y){
	// cout << "Training entered" << endl;
	vector<MatrixXf> layer_activations = forward(network,X);
	vector<MatrixXf> layer_inputs;
	// cout << "Training entered" << endl;
	layer_inputs.assign(layer_activations.begin(),layer_activations.end());
	layer_inputs.insert(layer_inputs.begin(),X);
	// cout << "Train initialisation  is correct" << endl;
	MatrixXf logits = layer_activations[layer_activations.size()-1];

	VectorXf loss = softmax_crossentropy_with_logits(logits,y);
	MatrixXf loss_grad = grad_softmax_crossentropy_with_logits(logits,y);

	// cout << "Loss calculation is correct" << endl;
	for(int i=network.size()-1;i>=0;i--){
		// cout << "Back propagation initialisation is correct" << endl;
		// cout << network[i]->learning_rate << endl;
		// cout << "loss_grad: "<< loss_grad.rows() << " " << loss_grad.cols() <<  endl;
		// cout << "loss_grad: "<< layer_inputs[i].rows() << " " << layer_inputs[i].cols() <<  endl;
		MatrixXf loss_grad1 = network[i]->backward(layer_inputs[i],loss_grad);
		loss_grad = loss_grad1;
	}
	// cout << "Back propagation is correct" << endl;
	double sum = 0;
	for(int i=0;i<loss.rows();i++){
		sum += loss(i);
	}
	return sum/loss.rows();

}

vector<int> random_permutation(int n){
	vector<int> result;
	for(int i=0;i<n;i++) result.push_back(i);
	random_shuffle(result.begin(),result.end());
	// for(int i=0;i<result.size();i++){
	// 	cout << result[i] << " ";
	// }
	// cout << endl;
	return result;
}

vector<Batch> iterate_minibatches(MatrixXf inputs, VectorXf targets, int batchsize){
	vector<int> indices = random_permutation(inputs.rows());
	// cout << "Random permutation is correct" << endl;
	int idx = 0;
	vector<Batch> result;
	// cout << "Iteration minibatches Initialisation is correct" << endl;
	while(idx<inputs.rows()){
		struct Batch batch;
		MatrixXf input = MatrixXf::Zero(batchsize,inputs.cols());
		VectorXf target = VectorXf::Zero(batchsize); 
		// cout << "Loop Initialisation is correct" << endl;
		int j;
		for(j=idx;j<idx+batchsize && j <inputs.rows();j++){
			input.row(j-idx) = inputs.row(indices[j]);
			// cout << "Input calculation is correct " << j << endl;
			target.row(j-idx) = targets.row(indices[j]);
		}
		// cout << j << " " << idx+batchsize-1 << endl;
		if(j==idx+batchsize) idx+=batchsize;
		else idx = inputs.rows();
		// cout << "Inner loop is correct" << endl;
		batch.inputs = input;
		batch.targets = target;
		result.push_back(batch);
	}
	return result;
}



int main(){
	srand(time(0));
	// Load MNIST data
	ifstream trainFile("mnist_train_min.csv");
	string line;
	vector<vector<string>> values;
	MatrixXf train_input(900,784);
	VectorXf train_output(900);
	int row = 0 ;
	while(getline(trainFile,line)){
		string line_value;
		vector<string> line_values;
		stringstream ss(line);
		int col = 0;
		if(row==0){
			row++;
			continue;
		}
		while(getline(ss,line_value,',')){
			// cout << line_value << endl;
			line_values.push_back(line_value);
			if(col==0) train_output(row-1) = stof(line_value);
			else train_input(row-1,col-1) = stoi(line_value)/255.0;
			col++;
		}
		row++;
		values.emplace_back(line_values);
	}

	ifstream testFile("mnist_test_min.csv");
	MatrixXf test_input(100,784);
	VectorXf test_output(100);
	row = 0 ;
	while(getline(testFile,line)){
		string line_value;
		vector<string> line_values;
		stringstream ss(line);
		int col = 0;
		if(row==0){
			row++;
			continue;
		}
		while(getline(ss,line_value,',')){
			// cout << line_value << endl;
			line_values.push_back(line_value);
			if(col==0) test_output(row-1) = stof(line_value);
			else test_input(row-1,col-1) = stoi(line_value)/255.0;
			col++;
		}
		row++;
		values.emplace_back(line_values);
	}

    std::cout << "Nbr of training images = " << train_input.rows() << std::endl;
    std::cout << "Nbr of training labels = " << train_output.rows() << std::endl;
    std::cout << "Nbr of test images = " << test_input.rows() << std::endl;
    std::cout << "Nbr of test labels = " << test_output.rows() << std::endl;
    // for(int i=0;i<train_input.rows();i++){
    // 	cout << i << " ";
    // 	for(int j=0;j<train_input.cols();j++){
    // 		cout << train_input(i,j) << " ";
    // 	}
    // 	cout << endl;
    // }
    // for(int i=0;i<train_output.rows();i++){
    // 	cout << i << " " << train_output(i);
    // 	cout << endl;
    // }

    // for(int i=0;i<test_input.rows();i++){
    // 	cout << i << " ";
    // 	for(int j=0;j<test_input.cols();j++){
    // 		cout << test_input(i,j) << " ";
    // 	}
    // 	cout << endl;
    // }
    // for(int i=0;i<test_output.rows();i++){
    // 	cout << i << " " << test_output(i);
    // 	cout << endl;
    // }
    vector<double> train_log;
    vector<double> val_log;
    vector<Layer *> network;
    // cout << "Initialisation is correct" <<endl;
    network.push_back(new DenseLayer(train_input.cols(),100));
    network.push_back(new ReLU());
    network.push_back(new DenseLayer(100,200));
    network.push_back(new ReLU());
    network.push_back(new DenseLayer(200,200));
    network.push_back(new ReLU());
    network.push_back(new DenseLayer(200,10));
    // cout << "Network population  is correct" <<endl;
    for(int epoch = 0; epoch<25;epoch++){
    	vector<Batch> batch = iterate_minibatches(train_input,train_output,32);
    	// for(int i=0;i<batch.size();i++){
    	// 	cout << "Batch " << i << " done"<<endl;
    	// 	for(int j=0;j<batch[i].targets.rows();j++){
    	// 		cout << batch[i].targets(j) <<  " ";
    	// 	}
    	// 	cout << endl;
    	// }

    	cout << "Batch size: " << batch.size() << endl;
    	for(int i=0;i<batch.size();i++){
    		// cout << "Batch " << i << " done"<<endl;
    		train(network,batch[i].inputs,batch[i].targets);
    	}
    	// cout << "Training is correct" << endl;
    	VectorXf train_pred = predict(network,train_input);
    	double train_accuracy =0;
    	for(int i=0;i<train_pred.rows();i++){
    		if(train_pred(i)==train_output(i)) train_accuracy+=1;
    	}
    	train_accuracy /= train_pred.rows();

    	VectorXf test_pred = predict(network,test_input);
    	double test_accuracy =0;
    	for(int i=0;i<test_pred.rows();i++){
    		// cout << test_pred(i) << " " << test_output(i) <<endl;
    		if(test_pred(i)==test_output(i)) test_accuracy+=1;
    	}

    	test_accuracy /= test_pred.rows();

    	cout << "Epoch: " << epoch << " Training accuracy: " << train_accuracy << " Validation accuracy: " << test_accuracy << endl;
    }


    return 0;
}
