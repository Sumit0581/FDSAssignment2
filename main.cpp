#include "multi_layer_perceptron.h"

int main(){
	cout << endl;
	cout << "*****************************************************************" <<endl;
	cout << "Welcome to the Neural Network C++ code" << endl;
	cout << "You need to give the following data as input." << endl;
	cout << "1. Training Dataset: csv file containing training data (eg: mnist_train_min.csv)." <<endl;
	cout << "2. Test Dataset: csv file containing training data (eg: mnist_test_min.csv)." <<endl;
	cout << "3. Input units: Number of input data neurons (eg: 784 for MNIST data)." <<endl;
	cout << "4. Output units: Number of output data neurons (eg: 10 for MNIST data)." <<endl;
	cout << "5. Hidden layers count: Number of hidden layers." <<endl;
	cout << "6. Hidden Layer Topology: Number of neurons in each hidden layer." <<endl;
	cout << "7. Activation function: 1 for RelU, 2 for Logistic, 3 for Tanh." <<endl;
	cout << "*****************************************************************" <<endl;
	cout << endl;

	srand(time(0));
	string train_file, test_file;
	int inputsize, outputsize, hiddenlayercount,activationmode;
	cout << "Enter the name of training dataset: (Press Enter for default dataset (MNIST))";
	getline(cin,train_file) ;
	if(train_file.empty()){
		train_file.assign("mnist_train_min.csv");
		cout << "Training dataset is " << train_file <<endl;
	} 
	cout << "Enter the name of testing dataset: (Press Enter for default dataset (MNIST))";
	getline(cin,test_file);
	if(test_file.empty()){
		test_file.assign("mnist_test_min.csv");
		cout << "Testing dataset is " << test_file <<endl;
	} 

	cout << "Enter the number of input units: ";
	cin >> inputsize;

	cout << "Enter the number of output units: ";
	cin >> outputsize;

	cout << "Enter the number of hidden layers: ";
	cin >> hiddenlayercount;

	vector<int> topology(hiddenlayercount);
	cout << "Enter number of neurons in each hidden layer (separated by space):" <<endl;
	for(int i=0;i<hiddenlayercount;i++){
		cin >> topology[i];
	}

	cout << "Enter the type of activation you want \n 1. RelU \n 2. Logistics Sigmoid \n 3. Tanh \n";
	cin >> activationmode;

	// Load training data
	ifstream trainFileRows(train_file);
	string lines;
	int rows=0;
	while(getline(trainFileRows,lines)){
		rows++;
	}

	ifstream trainFile(train_file);
	string line;
	vector<vector<string>> values;
	MatrixXf train_input(rows-1,inputsize);
	VectorXf train_output(rows-1);
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
			line_values.push_back(line_value);
			if(col==0) train_output(row-1) = stof(line_value);
			else train_input(row-1,col-1) = stoi(line_value)/255.0;
			col++;
		}
		row++;
		values.emplace_back(line_values);
	}

	// Load test data
	ifstream testFileRows(test_file);
	rows=0;
	while(getline(testFileRows,lines)){
		rows++;
	}

	ifstream testFile(test_file);
	MatrixXf test_input(rows-1,inputsize);
	VectorXf test_output(rows-1);
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

    cout << "Nbr of training images = " << train_input.rows() << endl;
    cout << "Nbr of training labels = " << train_output.rows() << endl;
    cout << "Nbr of test images = " << test_input.rows() << endl;
    cout << "Nbr of test labels = " << test_output.rows() << endl;
    
    vector<double> train_log;
    vector<double> val_log;
    vector<Layer *> network = build_network(inputsize,outputsize,hiddenlayercount,topology,activationmode);
    
    // Training model
    for(int epoch = 0; epoch<25;epoch++){
    	vector<Batch> batch = iterate_minibatches(train_input,train_output,32);
    	for(int i=0;i<batch.size();i++){
    		train(network,batch[i].inputs,batch[i].targets);
    	}
    	VectorXf train_pred = predict(network,train_input);
    	double train_accuracy =0;
    	for(int i=0;i<train_pred.rows();i++){
    		if(train_pred(i)==train_output(i)) train_accuracy+=1;
    	}
    	train_accuracy /= train_pred.rows();
    	VectorXf test_pred = predict(network,test_input);
    	double test_accuracy =0;
    	for(int i=0;i<test_pred.rows();i++){
    		if(test_pred(i)==test_output(i)) test_accuracy+=1;
    	}
    	test_accuracy /= test_pred.rows();
    	cout << "Epoch: " << epoch+1 << " Training accuracy: " << train_accuracy << " Validation accuracy: " << test_accuracy << endl;
    }

    return 0;
}
