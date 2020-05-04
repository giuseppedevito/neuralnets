// Copyright (C) 2020 Giuseppe De Vito
// 
// Author(s):
// Giuseppe De Vito <giuseppedv@gmail.com>
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
// License (COPYING.txt) for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

/**
 * Rete neurale Basic (2 combinazioni) Giuseppe De Vito
 * 
 */

#include <bits/stdc++.h>
#include <stdio.h>  /* printf, scanf, puts, NULL */
#include <stdlib.h> /* srand, rand */
#include <time.h>

static const int numInputs = 2;
static const int numBias = 1;
static const int numTrainingData = 2;
double learningRate = 0.1;
double trainingData[numTrainingData][numInputs] = {
    {0.0, 0.1},
    {0.1, 0.0}
};
double trainingOutputs[numTrainingData] = {
    0.0, 1.0
};
double weights[numInputs];
double bias[numBias];
int currentTrainingDataIndex = 0;

double sigmoid(double x);
double dSigmoid(double sig_x);
void training(int training_num);
void initWeights();
void initBias();
void initAll();
double propagate();
void learn(double propagated_output);
void printWeights();
void printAll();

int main(int argc, char ** argv){
    if(argc < 2){
        printf("Usage: Perceptron <training itarations num>\n");
        return 1;
    }
    initAll();
    training(atoi(argv[1]));
    printAll();
    return 0;
}

void training(int training_num){
    double propagation_output = 0.0;
    for(int i = 0; i < training_num; i++){
        currentTrainingDataIndex = i % numTrainingData;
        propagation_output = propagate();
        learn(propagation_output);
    }
}

double propagate(){
    float signal = 0.0;
    float activation = 0.0;
    for(int i = 0; i < numInputs; i++){
        signal += weights[i] * trainingData[currentTrainingDataIndex][i];
    }
    signal += bias[0];
    activation = sigmoid(signal);
    return activation;
}

void learn(double propagated_output){
    double k = propagated_output;
    //double sig_k = sigmoid(k);
    double d_sig_k = dSigmoid(k);
    double t = trainingOutputs[currentTrainingDataIndex] - k;
    double dEdt = -2 * t;
    double dEdk = dEdt * d_sig_k;
    double delta_bias = dEdk * learningRate;
    for(int i = 0; i < numInputs; i++){
        double delta_w = dEdk * trainingData[currentTrainingDataIndex][i] * learningRate;
        weights[i] = weights[i] - delta_w;
    }
    bias[0] = bias[0] - delta_bias;
}

void initWeights(){
    for(int i = 0; i < numInputs; i++){
        weights[i] = ((double)(std::rand() % 1000)) / 1000.0;
    }
}
void initBias(){
    for(int i = 0; i < numBias; i++){
        bias[i] = ((double)(std::rand() % 1000)) / 1000.0;
    }
}
void printWeights(){
    for(int i = 0; i < numInputs; i++){
        printf("w%d: %.05lf\n", i, weights[i]);
    }
}
void printAll(){
    printWeights();
    printf("Bias: %.05lf\n", bias[0]);
}
void initAll(){
    std::srand(1000);//init the random generator with the same seed
    initBias();
    initWeights();
}

double sigmoid(double x)
{
    return (double)(1.00 / (1.00 + std::exp(-x)));
}
double dSigmoid(double sig_x)
{
    return (double)((sig_x * (1.00 - sig_x)));
}
