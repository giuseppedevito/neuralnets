# neuralnets
## Talking about Neural Networks Basis

## Introduction

## NeuralNet.cc

Beginning steps with Arduino to x86 porting; just to understanding the original code.  
### Don't use it.

## Perceptron
Basic single layer Neural Network with Sigmoid activation function, two input codes ((1,0) and (0,1)) and just the regression line x-y=0.

Compilation

g++ -o Perceptron Perceptron.cc

Execution

./Perceptron <number of training cycles>
  
Example

./Perceptron 1000

## OrPerceptron

The same as Perceptron but it find the regression line for the Or boolean operator.

Compilation

g++ -o OrPerceptron OrPerceptron.cc

Execution

./OrPerceptron <number of training cycles>
  
Example

./OrPerceptron 10000

## Xor Network

The double layer neural network used to find the plane that divide the xor solutions.

Number of Inputs: 2

Number of hidden nodes: 3

Number of outputs: 1

### Pay Attention: the number of training cycles must be high (I used 1000000)

Compilation

g++ -o XorNet XorNet.cc

Execution

./XorNet <number of training cycles>
  
Example

./XorNet 1000000


