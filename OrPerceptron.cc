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
 * Rete neurale Or Inclusivo Giuseppe De Vito
 * 
 */

#include <bits/stdc++.h>
#include <stdio.h>  /* printf, scanf, puts, NULL */
#include <stdlib.h> /* srand, rand */
#include <time.h>

static const int numInputs = 2;
static const int numBias = 1;
static const int numTrainingData = 4;
double learningRate = 0.1;
double trainingData[numTrainingData][numInputs] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0},
};
double trainingOutputs[numTrainingData] = {
    0.0, 1.0, 1.0, 1.0
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

/**
 * IMPORTANT: if k is the output of an Output Node and p is the
 * expected value from the specific training set, we are doing the
 * following error:
 * E = p - k
 * Now, we have to adjust weights to reduce such an error. In the
 * neural network litterature, Mean Squared Error (MSE) is used to delete
 * the signum from the computation:
 * Em = (p - k)^2/N where N is the number of samples (1 in our case).
 * So let consider E = (p - k)^2
 * If we want to update a weight (for example w1) to reduce the error,
 * we can use the gradient descendant tecnique:
 * w1 = w1 + Delta(w1), and for Delta we use a little fraction of the
 * derivate of the MSE Error:
 * Delta(w1) = - lr * dE/dw1, where lr is the learning rate, 0.1 in our
 * case, the signum (-) is added since the derivate of E is negative
 * if we have to increase w1, positive if we have to reduce it.
 * Hence we have to calculate dEdw1 using chain rules (funzioni composte, in italiano)
 * dEdw1 = dEdk * dkdw1 = -2(p - k) * dkdw1
 * But k = sigmoid(t), with t = w1*x1 + w2*x2 + w3*x3 + b, with xi=output of i-th hidden layer.
 * So dkdw1 = dkdt * dtdw1 = sigmoid(t)*(1 - sigmoid(t)) * dtdw1,
 * and dtdw1 = x1.
 * Definitively we have:
 * dEdw1 = dEdk * dkdt * dtdw1 = -2(p - k) * sigmoid(t)*(1 - sigmoid(t)) * x1
 * For the bias we have 
 * dEdb = dEdk * dkdt * dtdb = -2(p - k) * sigmoid(t)*(1 - sigmoid(t))
 */
void learn(double propagated_output){
    double k = propagated_output;
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
