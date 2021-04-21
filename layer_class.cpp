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
	normal_distribution<double>(0.0,scale);
	default_random_engine generator;
	MatrixXf result(input_units,output_units);
	for(int i=0;i<input_units;i++){
		for(int j=0;j<output_units;j++){
			result(i,j) = distribution(generator);
		}
	}
	return result;
}

VectorXf zeros(int output_units){
	VectorXf result(output_units);
	for(int i=0;i<output_units;i++){
		result(i) = 0.0;
	}
	return result;
}

class Layer{
	public:
		MatrixXf forward(MatrixXf input) {
			return input;
		}
		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			int num_units = input.cols();
			MatrixXf d_layer_d_input = eye(num_units);
			return grad_output*d_layer_d_input;
		}
};

VectorXf mean(MatrixXf matrix){
	VectorXf result(matrix.cols());
	for(int i=0;i<matrix.cols();i++){
		double sum = 0;
		for(int j=0;j<matrix.rows();j++){
			sum += matrix(j,i);
		}
		sum /= matrix.rows();
		result(i) = sum;
	}
	return result;
}

class Dense: public Layer{
	private:
		double learning_rate;
		MatrixXf weights;
		VectorXf biases;
	public:
		
		Dense(int input_units,int output_units, double rate=0.1){
			learning_rate = rate;
			weights = random_normal(input_units,output_units);
			biases = zeros(output_units);
		}

		MatrixXf forward(MatrixXf input){
			return input*weights + biases;
		}
		MatrixXf backward(MatrixXf input, MatrixXf grad_output){
			
			MatrixXf grad_input = grad_output * weights.transpose();
			MatrixXf grad_weights = input.transpose() * grad_output;
			MatrixXf grad_biases = mean(grad_output)*input.rows();

			int num_units = input.cols();
			MatrixXf d_layer_d_input = eye(num_units);
			return grad_output*d_layer_d_input;
		}

};

int main(){
	return 0;
}