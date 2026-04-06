#include <iostream>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include "Tensor.h"
using namespace std;

int main() {
    srand(time(NULL));
    // 1: Crear un tensor de entrada de dimensiones 1000 × 20 ×20.
    Tensor entrada = Tensor::random({1000,20,20}, -1.0, 1.0);
    cout << "Pregunta 1 realizada" << endl;

    // 2: Transformarlo a 1000 × 400 usando view
    Tensor entrada_vector = entrada.view({1000,400});
    cout << "Pregunta 2 realizada" << endl;

    // 3: Multiplicarlo por una matriz 400 × 100
    Tensor W1 = Tensor::random({400,100}, -1.0, 1.0);
    Tensor capa1 = matmul(entrada_vector, W1);
    cout << "Pregunta 3 realizada" << endl;

    // 4: Sumar una matriz 1×100
    Tensor b1 = Tensor::random({1,100}, -1.0, 1.0);
    Tensor capa1_bias = capa1 + b1;
    cout << "Pregunta 4 realizada" << endl;

    // 5: Aplicar la funcion ReLU
    ReLU relu;
    Tensor capa1_relu = relu.apply(capa1_bias);
    cout << "Pregunta 5 realizada" << endl;

    // 6: Multiplicar por una matriz 100 × 10
    Tensor W2 = Tensor::random({100,10}, -1.0, 1.0);
    Tensor capa2 = matmul(capa1_relu, W2);
    cout << "Pregunta 6 realizada" << endl;

    // 7: Sumar una matriz 1×10.
    Tensor b2 = Tensor::random({1,10}, -1.0, 1.0);
    Tensor capa2_bias = capa2 + b2;
    cout << "Pregunta 7 realizada" << endl;

    // 8: Aplicar la funcion Sigmoid
    Sigmoid sig;
    Tensor salida = sig.apply(capa2_bias);
    cout << "Pregunta 8 realizada" << endl;
    cout << "Red neuronal ejecutada correctamente" << endl;

    /*
    //Prueba pregunta 3.2
    Tensor A = Tensor :: zeros ({2 , 3}) ;
    Tensor B = Tensor :: ones ({3 , 3}) ;
    Tensor C = Tensor :: random ({2 , 2} , 0.0 , 1.0) ;
    Tensor D = Tensor :: arange (0 , 6) ;
    Tensor E = Tensor::concat({A, B}, 0);
    */

    /*
    //Prueba pregunta 5.2
    Tensor A2 = Tensor::arange(-5, 5).view({2, 5});
    ReLU relu1;
    Tensor B2 = A2.apply(relu1);
    */

    /*
    //Prueba pregunta 7.1
    Tensor A3 = Tensor::arange(0, 12);
    Tensor B3 = A3.view({3, 4});
    */

    /*
    //Prueba pregunta 8
    Tensor A4 = Tensor::ones({2, 3});
    Tensor B4 = Tensor::zeros({2, 3});
    Tensor C4 = Tensor::concat({A4, B4}, 0);
    */
}