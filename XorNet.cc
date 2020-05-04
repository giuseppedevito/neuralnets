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
 * Rete neurale Xor a due livelli Giuseppe De Vito
 * 
 */

#include <bits/stdc++.h>
#include <stdio.h>  /* printf, scanf, puts, NULL */
#include <stdlib.h> /* srand, rand */
#include <time.h>

static const int numInputs = 2;
static const int numHiddens = 3;
static const int numOutputs = 1;
static const int numBias = 1;
static const int numTrainingData = 4;
double learningRate = 0.1;
double hidddenInputs[numTrainingData][numInputs] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0},
};
double outputInputs[numHiddens][numOutputs] = {
    {0.0},
    {0.0},
    {0.0}
};
double outputOutputs[numOutputs] = {
    0.0
};
double trainingOutputs[numTrainingData][numOutputs] = {
    {0.0}, {1.0}, {1.0}, {0.0}
};
double hiddenWeights[numInputs][numHiddens] = {
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0}
};
double hiddenWeightsDeltas[numInputs][numHiddens] = {
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0}
};
double outputWeights[numHiddens][numOutputs] = {
    {0.0},
    {0.0},
    {0.0}
};
double outputBackPropagation[numHiddens][numOutputs] = {
    {0.0},
    {0.0},
    {0.0}
};
double outputWeightsDeltas[numHiddens][numOutputs] = {
    {0.0},
    {0.0},
    {0.0}
};
double outputBias[numBias] = {
    1.0
};
double outputBiasDeltas[numBias] = {
    1.0
};
double hiddenBias[numHiddens] = {
    1.0, 1.0, 1.0
};
double hiddenBiasDeltas[numHiddens] = {
    1.0, 1.0, 1.0
};
int currentTrainingDataIndex = 0;

double sigmoid(double x);
double dSigmoid(double sig_x);
void training(int training_num);
void initHiddenWeights();
void initOutputWeights();
void initWeights();
void initHiddenBiases();
void initOutputBiases();
void initBiases();
void initAll();
void propagate();
void learn();
void printHiddenWeights();
void printOutputWeights();
void printHiddenBiases();
void printOutputBiases();
void printWeights();
void printBiases();
void printAll();
void computeOutputDeltas();
void computeHiddenDeltas();
void updateOutputWeights();
void updateOutputBias();
void updateHiddenWeights();
void updateHiddenBias();
void computeDeltas();
void updateWeightsAndBias();
void training(int training_num);
void test();

int main(int argc, char ** argv){
    if(argc < 2){
        printf("Usage: XorNet <training itarations num>\n");
        return 1;
    }
    initAll();
    training(atoi(argv[1]));
    printAll();
    test();
    return 0;
}

void test(){
    for(int i = 0; i < numTrainingData; i++){
        for(int j = 0; j < numInputs; j++){
            double xj = hidddenInputs[i][j];
            printf("Test Inputs: x_%d: %.01lf\n", j, xj);
        }
        currentTrainingDataIndex = i;
        propagate();
        for(int j = 0; j < numOutputs; j++){
            printf("Results: %.05lf\n", outputOutputs[j]);
        }
    }
}

void training(int training_num){
    for(int i = 0; i < training_num; i++){
        currentTrainingDataIndex = i % numTrainingData;
        propagate();
        learn();
    }
}

void propagate(){
    //1. Start from the Hidden Nodes receiving Network Inputs
    double * inputs = hidddenInputs[currentTrainingDataIndex];
    for(int j = 0; j < numHiddens; j++){
        double signal = 0.0;
        double activation = 0.0;
        for(int i = 0; i < numInputs; i++){
            signal += hiddenWeights[i][j] * inputs[i];//cartesian product
        }
        signal += hiddenBias[j];//add the bias
        activation = sigmoid(signal);//let the sigmoid activate (or not) the neuron
        for(int k = 0; k < numOutputs; k++){
            outputInputs[j][k] = activation;
        }
    }
    //2. Terminate with the output nodes
    for(int i = 0; i < numOutputs; i++){
        double signal = 0.0;
        double activation = 0.0;
        for(int j = 0; j < numHiddens; j++){
            signal += outputWeights[j][i] * outputInputs[j][i];
        }
        signal += outputBias[i];
        activation = sigmoid(signal);
        outputOutputs[i] = activation;
    }
}

void learn(){

    /**
     * 1. Start the backpropagation phase starting from the output nodes,
     * calculating weights and biases deltas for them.
     * After, compute weights and biases deltas for hidden nodes.
     */ 
    computeDeltas();
    
    /**
     * 2. At the end of backpropagation we have to update all weights and
     * biases.
     */
    updateWeightsAndBias();
    
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
void computeOutputDeltas(){
    for(int i = 0; i < numOutputs; i++){
        double p = trainingOutputs[currentTrainingDataIndex][i];
        double k = outputOutputs[i];//output of the output node
        double sig_t = k;//to be used in the next instruction
        double d_sig_t = dSigmoid(sig_t);//derivate of the sigmoid
        double dEdb = -2 * (p - k) * d_sig_t;
        for(int j = 0; j < numHiddens; j++){
            double xj = outputInputs[j][i];//output of the j-th hidden node
            double dEdwj = dEdb * xj;
            outputWeightsDeltas[j][i] = dEdwj;
            outputBackPropagation[j][i] = dEdb;//we will use this for the hidden weights deltas
        }
        outputBiasDeltas[i] = dEdb;
    }
}

/**
 * Here we have to calculate the deltas for the input weights of the hidden nodes.
 * To calculate such values we  have to use backpropagation.
 * Let's start with the Error we have calculated in the computeOutputDeltas function.
 * E = (p - ki)^2, where ki is the output of ith output node.
 * We know that ki = sigmoid(t), where t = x1*w1i + x2*w2i + x3*w3i
 * Let's focus on a single item, for example x1. It is the value that comes from
 * hidden node 1; we have calculated it in the forward propagation step.
 * But x1 depends on the inputs and inputs weights of hidden nodes:
 * x1 = sigmoid(s), with s = xh1*wh11 + xh2*wh21 + bh1
 * The same way x2 = sigmoid(m), with m = xh1*wh12 + xh2*wh22 + bh2
 * x3 = sigmoid(g), with g = xh1*wh13 + xh2*wh23 + bh2
 * So: t = sigmoid(s)*w1i + sigmoid(m)*w2i + sigmoid(g)*w3i
 * Now, we have to calculate hidden weights deltas, so, considering wh11,
 * wh11 = wh11 + Delta(wh11) and Delta(wh11) = -lr * dEdwh11
 * Applying chain rules
 * dEdwh11 = dEdk * dkdt * dtdx1 * dx1ds * dsdwh11 = 
 * = -2(p - k) * sigmoid(t)*(1 - sigmoid(t)) * w1i * sigmoid(s)*(1 - sigmoid(s)) * xh1
 * Note that the first part, (-2(p - k) * sigmoid(t)*(1 - sigmoid(t))), has been 
 * calculated previously in the outputs weights deltas.
 * So we can fix bProp = -2(p - k) * sigmoid(t)*(1 - sigmoid(t))
 * and finally dEdwh11 = bProp * w1i * sigmoid(s)*(1 - sigmoid(s)) * xh1
 */
void computeHiddenDeltas(){
    for(int i = 0; i < numOutputs; i++){
        for(int j = 0; j < numHiddens; j++){
            double sig_s = outputInputs[j][i];
            double d_sig_s = dSigmoid(sig_s);
            double bProp = outputBackPropagation[j][i];
            for(int n = 0; n < numInputs; n++){
                double w_ji = outputWeights[j][i];//our w1i in the function doc
                double x_hn = hidddenInputs[currentTrainingDataIndex][n];
                hiddenWeightsDeltas[n][j] = bProp * w_ji * d_sig_s * x_hn;
            }
            hiddenBiasDeltas[j] = bProp * d_sig_s;
        }
    }
}

void updateOutputWeights(){
    for(int i = 0; i < numOutputs; i++){
        for(int j = 0; j < numHiddens; j++){
            outputWeights[j][i] -= learningRate * outputWeightsDeltas[j][i];
        }
    }
}

void updateOutputBias(){
    for(int i = 0; i < numOutputs; i++){
        outputBias[i] -= learningRate * outputBiasDeltas[i];
    }
}

void updateHiddenWeights(){
    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j < numHiddens; j++){
            hiddenWeights[i][j] -= learningRate * hiddenWeightsDeltas[i][j];
        }
    }
}

void updateHiddenBias(){
    for(int i = 0; i < numHiddens; i++){
        hiddenBias[i] -= learningRate * hiddenBiasDeltas[i];
    }
}

void computeDeltas(){
    computeOutputDeltas();
    computeHiddenDeltas();
}

void updateWeightsAndBias(){
    updateOutputWeights();
    updateOutputBias();
    updateHiddenWeights();
    updateHiddenBias();
}

void initHiddenWeights(){
    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j < numHiddens; j++){
            hiddenWeights[i][j] = ((double)(std::rand() % 1000)) / 1000.0;
        }
    }
}
void initOutputWeights(){
    for(int i = 0; i < numHiddens; i++){
        for(int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = ((double)(std::rand() % 1000)) / 1000.0;
        }
    }
}

void initWeights(){
    initHiddenWeights();
    initOutputWeights();
}

void initHiddenBiases(){
    for(int i = 0; i < numHiddens; i++){
        hiddenBias[i] = ((double)(std::rand() % 1000)) / 1000.0;
    }
}

void initOutputBiases(){
    for(int i = 0; i < numOutputs; i++){
        outputBias[i] = ((double)(std::rand() % 1000)) / 1000.0;
    }
}

void initBiases(){
    initHiddenBiases();
    initOutputBiases();
}

void initAll(){
    std::srand(1000);
    initBiases();
    initWeights();
}

void printHiddenWeights(){
    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j < numHiddens; j++){
            printf("Wh%d_%d: %.05lf\n", i, j, hiddenWeights[i][j]);
        }
    }
}
void printOutputWeights(){
    for(int i = 0; i < numHiddens; i++){
        for(int j = 0; j < numOutputs; j++){
            printf("Wo%d_%d: %.05lf\n", i, j, outputWeights[i][j]);
        }
    }
}
void printHiddenBiases(){
    for(int i = 0; i < numHiddens; i++){
        printf("Bh%d: %.05lf\n", i, hiddenBias[i]);
    }
}
void printOutputBiases(){
    for(int i = 0; i < numOutputs; i++){
        printf("Bo%d: %.05lf\n", i, outputBias[i]);
    }
}

void printWeights(){
    printHiddenWeights();
    printOutputWeights();
}

void printBiases(){
    printHiddenBiases();
    printOutputBiases();
}

void printAll(){
    printWeights();
    printBiases();
}

double sigmoid(double x)
{
    return (double)(1.00 / (1.00 + std::exp(-x)));
}
double dSigmoid(double sig_x)
{
    return (double)((sig_x * (1.00 - sig_x)));
}
