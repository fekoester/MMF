#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <armadillo>

//Struct holding all parameters for the reservoir
struct Par{
	double a = 0.5;
	double b = 0.4;
	double theta = 0.5;
	unsigned int N = 100;
	unsigned int tauInTheta = 141;
	double regularisationParameter = 1E-6;
	bool regularisationParameterRelative = true;
	std::vector<double> maskWeights;
	double relativeThreshold = 1E-6;
	std::string filenameToWrite = "mmf_output.txt";
	std::ofstream fileToWrite;

};

//Get Psi function
//Takes the "EW a" and the virtual node distance "theta"
double getP(Par &parameters){
	return exp(parameters.a*parameters.theta);
}

//Get the "i,j" term in the parsalcs triangle for "modifiedState"
//takes the "i-th", "j-th" values, the "EW a" and the "feedback b" and the "functions psi" and "m"
double getPascalTerm(Par &parameters,double psi, double m, unsigned long long int i, unsigned long long j){
	return pow(-1,j)*boost::math::binomial_coefficient<double>(i,j)*pow(psi,i-j)*pow(parameters.b/parameters.a,j)*pow(m,j);
}

//Get the "i,j" term in the binomial distrubtion for "modifiedState"
//takes the "i-th", "j-th" values, the "Ew a" and the "feedback b" and the "functions psi" and "m"
double getBinomialTerm(Par &parameters,double psi, double m, unsigned long long int i, unsigned long long j){
	return pow(-parameters.b/parameters.a,j)*(boost::math::pdf(boost::math::binomial(i,m),j));
}

//Returns the modifoed state matrix,
//Takes the "Parameters" of the reservoir and the probability "p" 
arma::Mat<double> getModifiedState(Par &parameters, double p){
	unsigned int maxNumberRows = 5000;
	arma::Mat<double> modifiedState(parameters.N,125000);
	int highest_l = 0;
	modifiedState.zeros();
	double threshold = getPascalTerm(parameters, p, 1-p, 0, 0)*parameters.relativeThreshold;
	double term = 0.0;
	for (unsigned int i = 0; i < maxNumberRows; i++){
		unsigned int checkForLayerBreak = 0;
		for (unsigned int j = 0; j < i+1; j++){
			//term = getPascalTerm(parameters, psi, m, i, j);
			term = getBinomialTerm(parameters, p, 1-p, i, j);
			if (term < threshold){
				if (int((parameters.N+i+j*parameters.tauInTheta)/parameters.N + 0.5) > highest_l){
					highest_l = int((parameters.N+i+j*parameters.tauInTheta)/parameters.N + 0.5);
				}

				checkForLayerBreak += 1;
			}

			if (checkForLayerBreak == i+1){
				modifiedState.reshape(parameters.N,highest_l);

				return modifiedState;
			}

			for (unsigned int n = 0; n < parameters.N; n++){
				//std::cout << n << " " << int((n+i+j*parameters.tauInTheta)/parameters.N + 0.5) << std::endl;
				modifiedState(n,int((n+i+j*parameters.tauInTheta)/parameters.N + 0.5)) += term*parameters.maskWeights[(n+i+j*parameters.tauInTheta)%parameters.N];

			}
		}
	}

	return modifiedState;
}


arma::Mat<double> getMatrix(Par &parameters, arma::Mat<double> &modifiedState){
	return modifiedState*modifiedState.t();
}

arma::Col<double> getVector(Par &parameters, arma::Mat<double> &modifiedState, unsigned int l){
	return modifiedState.col(l);
}

arma::Col<double> getCapacity(arma::Mat<double> &matrix, arma::Mat<double> &modifiedState, unsigned int l){
	return modifiedState.col(l).t()*matrix*modifiedState.col(l);
}

//Pareses the arguments to the variables given to the program
Par parsArguments(int argc, char** argv){

	std::mt19937 generator1(0);
	std::uniform_real_distribution<double> distribution(0.0,1.0);

	Par parameters;

	parameters.maskWeights = std::vector<double>(parameters.N);
	for (unsigned int j = 0; j< parameters.N ; j++){
		parameters.maskWeights[j] = distribution(generator1);
	}


	for (int i = 0; i < argc; i++){
		if (std::string(argv[i]) == "-a"){
			parameters.a = atof(argv[i+1]);
			std::cout << "Set a to " << parameters.a << std::endl;
		}

		if (std::string(argv[i]) == "-b"){
			parameters.b = atof(argv[i+1]);
			std::cout << "Set b to " << parameters.b << std::endl;
		}

		if (std::string(argv[i]) == "-theta"){
			parameters.theta = atof(argv[i+1]);
			std::cout << "Set theta to " << parameters.theta << std::endl;
		}

		if (std::string(argv[i]) == "-N"){
			parameters.N = atoi(argv[i+1]);
			std::cout << "Set N to " << parameters.N << std::endl;
		}

		if (std::string(argv[i]) == "-tauInTheta"){
			parameters.tauInTheta = atoi(argv[i+1]);
			std::cout << "Set tauInTheta to " << parameters.tauInTheta << std::endl;
		}

		if (std::string(argv[i]) == "-relativeThreshold"){
			parameters.relativeThreshold = atof(argv[i+1]);
			std::cout << "Set relativeThreshold to " << parameters.relativeThreshold << std::endl;
		}

		if (std::string(argv[i]) == "-regularisationParameter"){
			parameters.regularisationParameter = atof(argv[i+1]);
			std::cout << "Set regularisationParameter to " << parameters.regularisationParameter << std::endl;
		}

		if (std::string(argv[i]) == "-regularisationParameterRelative"){
			parameters.regularisationParameterRelative = atoi(argv[i+1]);
			std::cout << "Set regularisationParameterRelative to " << parameters.regularisationParameterRelative << std::endl;
		}

		if (std::string(argv[i]) == "-seed"){
			std::mt19937 generator2(atoi(argv[i+1]));
			parameters.maskWeights = std::vector<double>(parameters.N);
			for (unsigned int j = 0; j< parameters.N ; j++){
				parameters.maskWeights[j] = distribution(generator2);
			}

			std::cout << "Set seed to " << atof(argv[i+1]) << std::endl;

		}	
		
		if (std::string(argv[i]) == "-filename"){
			parameters.filenameToWrite = std::string(argv[i+1]);
			std::cout << "Set filename to " << parameters.filenameToWrite << std::endl;
		}

	}
	
	std::cout << "\n\n\n";

	//parameters.filenameToWrite = "./data_" + std::to_string(atof(parameters.a)) + "_" + std::to_string(atof(parameters.b)) + "_" std::to_string(atof(parameters.theta)) + "_" + std::to_string(atof(parameters.N)) + "_" + std::to_string(atof(parameters.relativeThreshold)) + ".txt";
	parameters.fileToWrite.open(parameters.filenameToWrite);

	return parameters;
}

int main(int argc, char** argv)
{

	auto start = std::chrono::steady_clock::now();
	Par parameters = parsArguments(argc, argv);
	double p = getP(parameters);

	arma::Mat<double> modifiedState = getModifiedState(parameters, p);
	arma::Mat<double> matrix = getMatrix(parameters, modifiedState);
	if (parameters.regularisationParameterRelative){
		parameters.regularisationParameter = matrix.max()*parameters.regularisationParameter;
	}
	matrix = arma::pinv(matrix + parameters.regularisationParameter*arma::eye<arma::mat>(parameters.N, parameters.N));

	std::cout << "Steps" << "\t" << "Capacity\n";
	parameters.fileToWrite << "Steps" << "\t" << "Capacity\n";
	for (unsigned int l = 0; l < modifiedState.n_cols; l++){
		std::cout << std::setprecision(14) << l << " " << getCapacity(matrix,modifiedState , l);
		parameters.fileToWrite << l << " " << getCapacity(matrix,modifiedState ,l);
	}

	auto end = std::chrono::steady_clock::now();

	if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() > 1000){
		std::cout << "Elapesed time : " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << std::endl;
	}
	else{
		std::cout << "Elapesed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	}


	return 0;
}
