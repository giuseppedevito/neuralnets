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

#include <bits/stdc++.h>
#include <stdio.h>  /* printf, scanf, puts, NULL */
#include <stdlib.h> /* srand, rand */
#include <time.h>

//nodi d'ingresso
static const int numInputs = 2;
//neuroni hidden layer
static const int numHiddenNodes = 2;
//numero neuroni d'uscita
static const int numOutputs = 1;

//gli array con i valori di uscita e i pesi
double hiddenOutputs[numHiddenNodes];
double hiddenBias[numHiddenNodes];
double hiddenWeights[numInputs][numHiddenNodes];
double outputOutputs[numOutputs];
double outputBias[numOutputs];
double outputWeights[numHiddenNodes][numOutputs];

//gli array per i delta necessari all'aggiornamento dei pesi
double deltaHiddenWeights[numInputs][numHiddenNodes];
double deltaHiddenBias[numHiddenNodes];
double deltaOutputWeights[numHiddenNodes][numOutputs];
double deltaOutputBias[numOutputs];

//learning rate (per non applicare le correzioni ai pesi, "intere" - darebbe instabilità)
double lr = 0.1;

//cosa vogliamo fare apprendere alla rete?
static const int numTrainingSets = 4000;
//static const long learning_cycles = 100000;
double training_inputs[4][numInputs] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 1.0f}};
double training_outputs[4][numOutputs] = {
    {0.0f},
    {1.0f},
    {1.0f},
    {0.0f}};

double sigmoid(double x);
double dSigmoid(double x);
void setup(int learning_cycles);
void generateInputs();
void loop();
void propagate(int i);
void learn(int i);
void printHiddenWeights();
void printOutputWeights();
void printWeights();
void initAll();
void printInputs(int i);
void generateTraining();
void initBiases();
void initWeights();

int main(int argc, char ** argv){
    if(argc < 2){
        printf("Usage: NeuralNet <learnign cycles>\n");
        return 1;
    }
    setup(atoi(argv[1]));
    return 0;
}

void generateTraining(){
    for(int i = 0; i < numTrainingSets; i++){
        double in1 = (double)(std::rand() % 1000)/1000.0;
        double in2 = (double)(std::rand() % 1000)/1000.0;
        bool in1Is0 = (in1 < 0.5);
        bool in2Is0 = (in2 < 0.5);
        bool in1is1 = !in1Is0;
        bool in2is1 = !in2Is0;
        training_inputs[i][0] = in1;
        training_inputs[i][1] = in2;
        if((in1Is0 && in2Is0) || (in1is1 && in2is1)){
            training_outputs[i][0] = 0.0;
        }else{
            training_outputs[i][0] = 1.0;
        }
    }
}

//la signoide e la sua derivata
double sigmoid(double x)
{
    return (double)(1.00 / (1.00 + std::exp(-x)));
}
double dSigmoid(double sig_x)
{
    return (double)((sig_x * (1.00 - sig_x)));
}

void setup(int learning_cycles)
{
    //Serial.begin(9600);
    //pinMode(2, INPUT);
    /**
   * fissiamo il seme in modo da ottenere sempre gli stessi
   * numeri casuali.
   */
    std::srand(1000);
    //init pesi
    initAll();
    //init trainings
    //generateTraining();
    printf("Siate pazienti... ci vuole molto!\n");
    //Fase di ISTRUZIONE della RETE
    for (long n = 0; n < learning_cycles; n++)
    {
        //sorteggio un set di apprendimento
        //int idSet = std::rand() % 4;
        for(int idSet = 0; idSet < 4; idSet++){
            //lo propago
            propagate(idSet);
            //backpropagation
            learn(idSet);
        }
    }
    printf("Ready!\n");
}

void generateInputs()
{
    /**
     * Generate random inputs from 0.0 to 1.0 
     */
    double a = (std::rand() % 1000) / 1000.0;
    double b = (std::rand() % 1000) / 1000.0;
    //training_inputs[0][0] = a; NOOOOOOO
    //training_inputs[0][1] = b;
    printf("IN:\t%.03f\t%.03f\n", a, b);
}

void loop()
{
    /*if (digitalRead(2)){
    
    
        
    
    propagate(0);
    for (int j=0; j<numOutputs; j++) {
      Serial.print("Y = ");
      Serial.println(outputLayer[j]);
      int pwm = outputLayer[j] * 255;
      analogWrite(11, pwm);
    }
    delay(1000);
  }*/
}

void initBiases(){
    for (int j = 0; j < numHiddenNodes; j++)
    {
        hiddenBias[j] = ((double)(std::rand() % 1000)) / 1000.0;
    }
    for(int j = 0; j < numOutputs; j++){
        outputBias[j] = ((double)(std::rand() % 1000)) / 1000.0;
    }
}

void initWeights(){
    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = ((double)(std::rand() % 1000)) / 1000.0;
        }
    }
    for(int i = 0; i < numHiddenNodes; i++){
        for(int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = ((double)(std::rand() % 1000)) / 1000.0;
        } 
    }
}

//inizializza i pesi e i bias con numeri casuali (tra 0 e 1)
void initAll()
{
    initBiases();
    initWeights();
}

void printInputs(int i){
    for(int j = 0; j < numInputs; j++){
        printf("Ingresso %d: %.03lf\n", j, training_inputs[i][j]);
    }
    printf("###################\n");
}

void propagate(int i)
{
    printInputs(i);

    // scelto il pattern i-esimo, lo propago
    // Per tutti i nodi Hidden:
    for (int j = 0; j < numHiddenNodes; j++)
    {
        double activation = hiddenBias[j];
        for (int k = 0; k < numInputs; k++)
        {
            //sommatoria degli ingressi * i pesi
            activation += training_inputs[i][k] * hiddenWeights[k][j];
        }
        //applico la sigmoide
        hiddenOutputs[j] = sigmoid(activation);
    }

    /*for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("Uscita Nodo Hidden %d: %.03lf\n", j, hiddenOutputs[j]);
    }*/

    // per l'uscita (o le uscite)
    for (int j = 0; j < numOutputs; j++)
    {
        double activation = outputBias[j];
        for (int k = 0; k < numHiddenNodes; k++)
        {
            activation += hiddenOutputs[k] * outputWeights[k][j];
        }
        outputOutputs[j] = sigmoid(activation);
    }

    for (int j = 0; j < numOutputs; j++)
    {
        printf("Uscita Rete %d: %.03lf\n", j, outputOutputs[j]);
        printf("Uscita Attesa %d: %.03lf\n", j, training_outputs[i][j]);
    }
    printf("===================\n");
}

double squaredError(double guessedVal, double expectedVal)
{
    return std::exp2((guessedVal - expectedVal));
}

double dSquaredError(double guessedVal, double expectedVal)
{
    return (expectedVal - guessedVal);
}

void learn(int i)
{
    // scelto il pattern i, applico la backpropagation

    // 1. parto dall'uscita e propago all'indietro l'errore
    // array con gli errori sulle uscite
    //double deltaOutput[numOutputs];//Aliverti
    for (int j = 0; j < numOutputs; j++)
    {
        //errore preso pari alla diff tra valore ottenuto nella propagazione e
        //valore desiderato
        /**
     * Aliverti
     */
        //double error = (training_outputs[i][j] - outputLayer[j]);
        /**
     * De Vito
     */
        double error = dSquaredError(outputOutputs[j], training_outputs[i][j]);
        //questo è il valore da usare per correggere i pesi.
        //l'errore portato all'ingresso del nodo di uscita
        /**
     * Aliverti
     */
        //deltaOutput[j] = error * dSigmoid(outputLayer[j]);
        /**
     * De Vito - 
     * Considerando xi=hiddenLayer[j]
     * Considerando wi=outputWeights[j][k]
     * Considerando y=sum(xi*wi)=outputLayer[j],
     * Considerando p=valore desiderato=training_outputs[i][j]
     * Considerando sig_k = uscita dalla funz sigmoide = outputLayer[j]
     * Considerando t = sig_k - p
     * Considerando la generazione del delta come
     * dEdwi = dEdt * dtdk * dkdwi = 2(sig_k - p)*sig_k*(1 - sig_k)*xi
     */
        error = error * dSigmoid(outputOutputs[j]); //only the last term is missing
        for (int k = 0; k < numHiddenNodes; k++)
        {
            deltaOutputWeights[k][j] = error * hiddenOutputs[k];
        }
        deltaOutputBias[j] = error;
    }

    /**
     * 2. Propago sui nodi hidden
     */
    for (int j = 0; j < numHiddenNodes; j++)
    {
        for(int p = 0; p < numOutputs; p++){
            double old_error = deltaOutputWeights[j][p];
            double error = old_error * dSigmoid(hiddenOutputs[j]) * outputWeights[j][p];
            for(int k = 0; k < numInputs; k++){
                deltaHiddenWeights[k][j] = error * training_inputs[i][k];
            }
            deltaHiddenBias[j] = old_error * dSigmoid(hiddenOutputs[j]);
        }
    }

    // propago sui nodi hidden
    // array con gli errori sui nodi hidden
    /*double deltaHidden[numHiddenNodes];
  for (int j=0; j<numHiddenNodes; j++) {
    double error = 0.0f;
    for (int k=0; k<numOutputs; k++) {
      //lo moltiplico per il peso per metterlo allo stesso livello
      //dell'uscita hidden
      error += deltaOutput[k] * outputWeights[j][k];
    }
    //qui riporto l'errore sul nodo hidden all'ingresso del nodo hidden
    deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
  }*/

    // 2. ora modifico i pesi
    // per le uscite - Aliverti
    /*for (int j = 0; j < numOutputs; j++)
    {
        outputLayerBias[j] += deltaOutput[j] * lr;
        for (int k = 0; k < numHiddenNodes; k++)
        {
            //il nuovo peso è pari al valore di uscita del nodo H moltiplicato per
            //l'errore portato all'ingresso del nodo di uscita
            outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
        }
    }*/
    //per le uscite - De Vito
    for (int j = 0; j < numOutputs; j++)
    {
        outputBias[j] += deltaOutputBias[j] * lr;
        for (int k = 0; k < numHiddenNodes; k++)
        {
            outputWeights[k][j] += deltaOutputWeights[k][j] * lr;
        }
    }
    // per i pesi H - Aliverti
    /*for (int j = 0; j < numHiddenNodes; j++)
    {
        hiddenLayerBias[j] += deltaHidden[j] * lr;
        for (int k = 0; k < numInputs; k++)
        {
            //il nuovo peso è pari al valore presentato all'ingresso moltiplicato per
            //l'errore H portato all'ingresso del nodo di ingresso
            hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
        }
    }*/
    // per i pesi H - De Vito
    for (int j = 0; j < numHiddenNodes; j++)
    {
        hiddenBias[j] += deltaHiddenBias[j] * lr;
        for (int k = 0; k < numInputs; k++)
        {
            hiddenWeights[k][j] += deltaHiddenWeights[k][j] * lr;
        }
    }
    printWeights();
}

void printHiddenWeights(){
    for (int j = 0; j < numHiddenNodes; j++)
    {
        for (int k = 0; k < numInputs; k++)
        {
            printf("Peso Hidden %d:%d: %.03lf\n", k, j, hiddenWeights[k][j]);
        }
    }
}

void printOutputWeights(){
    for (int j = 0; j < numOutputs; j++)
    {
        for (int k = 0; k < numHiddenNodes; k++)
        {
            printf("Peso Output %d:%d: %.03lf\n", k, j, outputWeights[k][j]);
        }
    }
}

void printWeights(){
    printHiddenWeights();
    printOutputWeights();
    printf("-----------\n");
}
