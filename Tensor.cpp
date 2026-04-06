#include "Tensor.h"

//Pregunta 3.1  - Constructor principal
Tensor::Tensor(const vector<size_t>& shape, const vector<double>& values) {
    dim = shape.size();  //Para definir la cantidad de dimensiones del tensor
    this->shape = shape; // Se almacena el shape recibido

    // Numero total de elementos del tensor
    size_t total = 1;
    for (size_t s : shape) {
        total *= s;
    }

    // Para validar la cantidad de datos
    if (total != values.size()) {
        throw invalid_argument("Cantidad de valores incorrecta");
    }

    matriz_1d = new double[total];         // Reserva de memoria dinamica contigua
    for (size_t i = 0; i < total; i++) {   // Copia de los valores
        matriz_1d[i] = values[i];
    }
}

// Pregunta 3.2 - Creacion de tensores pre-definidos
// Pregunta 3.2. Zeros: Para las matrices de ceros
Tensor Tensor::zeros(const vector<size_t>& shape) {
    // Colculo de total de elementos
    size_t total = 1;
    for (size_t s : shape) {
        total *= s;
    }

    // Inicializacion con ceros
    vector<double> values(total, 0.0);
    return Tensor(shape, values);
}

// Pregunta 3.2. Ones: Para la matrices de unos
Tensor Tensor::ones(const vector<size_t>& shape) {
    // Calculo del total de elementos
    size_t total = 1;
    for (size_t s : shape) {
        total *= s;
    }

    // Inicializacion con unos
    vector<double> values(total, 1.0);
    return Tensor(shape,values);
}

// Pregunta 3.2. Random: Para el random con valores de inicio a final
Tensor Tensor::random(const vector<size_t>& shape,double inicio, double final) {
    vector<double> values;

    // Calculo del total de elementos
    size_t total = 1;
    for (size_t r : shape) {
        total *= r;
    }

    // Generación de valors
    for (size_t i = 0; i < total; i++) {
        double r = inicio + (double)rand() / RAND_MAX * (final - inicio);
        values.push_back(r);
    }
    return Tensor(shape, values);
}

// Pregunta 3.2. Arange: Para el arange de Tensor
Tensor Tensor::arange(int inicio, int final) {
    vector<double> values;

    // Generacion de valores
    for (int i = inicio; i < final; i++) {
        values.push_back(i);
    }

    return Tensor({values.size()}, values);
}

// Pregunta 4 - Gestion de Memoria y Ciclo de Vida

// Pregunta 4. Constructor de Copia
Tensor::Tensor ( const Tensor & other ) {
    dim = other.dim;
    shape = other.shape;
    size_t total = 1;
    for (size_t s : shape) {total *= s;};
    matriz_1d = new double[total];
    for (size_t i = 0; i < total; i++) {
        matriz_1d[i] = other.matriz_1d[i];
    }
}

// Pregunta 4. Asignacion de Copia
Tensor & Tensor::operator =( const Tensor & other ) {
    if (this == &other) return *this;
    delete[] matriz_1d;                          // Libera memoria

    // Copia datos
    dim = other.dim;
    shape = other.shape;

    // Calcula tamaño
    size_t total = 1;
    for (size_t s : shape) {
        total *= s;
    }

    // Reserva y copia valores
    matriz_1d = new double[total];
    for (size_t i = 0; i < total; i++) {
        matriz_1d[i] = other.matriz_1d[i];
    }
    return *this;
}

// Constructor de movimiento
Tensor::Tensor ( Tensor && other ) noexcept {
    // Copia datos sin duplicarse la memoria
    dim = other.dim;
    shape = move(other.shape);
    matriz_1d = other.matriz_1d;

    other.matriz_1d = nullptr;
}

// Asignacion por movimiento
Tensor & Tensor::operator =( Tensor && other ) noexcept {
    if (this == &other) return *this;
    delete[] matriz_1d;               // Libera memoria

    //Toma datos de other
    dim = other.dim;
    shape = move(other.shape);
    matriz_1d = other.matriz_1d;

    //Vacia el objeto
    other.matriz_1d = nullptr;

    return *this;
}

// Destructor de la clase
Tensor::~Tensor() {
    delete[] matriz_1d;         //Libera la memoria
}

// Pregunta 5 - Polimorfismo y Transformaciones

// Pregunta 5.2 - Metodo de aplicacion en clase Tensor
//Pregunta 5.2 - Apply en calse Tensor
Tensor Tensor::apply(const TensorTransform & transform) const {
    return transform.apply(*this);
}

// Pregunta 5.2. Apply en calse ReLu
Tensor ReLU::apply(const Tensor& t) const {
    Tensor result = t; // copia

    // Calculo de elementos
    size_t total = 1;
    for (size_t s : t.shape)
        total *= s;

    // Se aplica max(0, x) a cada elemento
    for (size_t i = 0; i < total; i++)
        result.matriz_1d[i] = max(0.0, t.matriz_1d[i]);

    return result;
}

// Pregunta 5.2. Implementacion de Sigmoid
Tensor Sigmoid::apply(const Tensor& t) const {
    Tensor result = t;

    // Calculo de elementos
    size_t total = 1;
    for (size_t s : t.shape)
        total *= s;

    // Se aplica la funcion a cada elemento
    for (size_t i = 0; i < total; i++)
        result.matriz_1d[i] =
            1.0 / (1.0 + exp(-t.matriz_1d[i]));

    return result;
}

// Pregunta 6: Sobrecarga de operadores

// Pregunta 6. Suma
Tensor Tensor::operator+(const Tensor& other) const {
    // Caso 1: Mismas dimensiones. Por ejemplo (a: 2x3, b:2x3)
    if (shape == other.shape) {
        Tensor result(*this);
        // Calculo de elementos
        size_t total = 1;
        for (size_t s : shape) {
            total *= s;
        }
        // Sumar elementos de la misma posicion
        for (size_t i = 0; i < total; i++) {
            result.matriz_1d[i] += other.matriz_1d[i];
        }
        return result;
    }

    // Caso 2:  (N x M) + (1 x M)
    if (shape.size() == 2 &&
        other.shape.size() == 2 &&
        other.shape[0] == 1 &&
        shape[1] == other.shape[1]) {

        size_t filas = shape[0];
        size_t columnas = shape[1];
        Tensor result(*this);

        // Suma por filas
        for (size_t i = 0; i < filas; i++) {
            for (size_t j = 0; j < columnas; j++) {
                result.matriz_1d[i * columnas + j] +=
                    other.matriz_1d[j];}}

        return result;}

    // Si no es ninguno de los 2
    throw invalid_argument("Dimensiones incompatibles");
}

// Pregunta 6. Resta
Tensor Tensor::operator-(const Tensor& other) const {
    // Caso 1: Mismas dimensiones
    if (shape == other.shape) {
        Tensor result(*this);
        size_t total = 1;

        for (size_t s : shape) {
            total *= s;
        }
        for (size_t i = 0; i < total; i++) {
            result.matriz_1d[i] -= other.matriz_1d[i];
        }

        return result;
    }

    // Caso 2: (N x M) - (1 x M)
    if (shape.size() == 2 &&
        other.shape.size() == 2 &&
        other.shape[0] == 1 &&
        shape[1] == other.shape[1]) {

        size_t filas = shape[0];
        size_t columnas = shape[1];
        Tensor result(*this);

        for (size_t i = 0; i < filas; i++) {
            for (size_t j = 0; j < columnas; j++) {
                result.matriz_1d[i * columnas + j] -=
                    other.matriz_1d[j];
            }
        }

        return result;
        }

    // Si no es ninguno de los 2
    throw invalid_argument("Dimensiones incompatibles");
}

// Pregunta 6. Multiplicacion
Tensor Tensor::operator*(const Tensor& other) const {
    // Verifica que tengan el mismo tamaño
    if (shape != other.shape)
        throw invalid_argument("Dimensiones incompatibles");

    Tensor result(*this);

    // Calcula el numero de valores
    size_t total = 1;
    for (size_t s : shape)
        total *= s;

    // Multiplica valor por valor
    for (size_t i = 0; i < total; i++)
        result.matriz_1d[i] *= other.matriz_1d[i];

    return result;
}

// Multiplicación por un número
Tensor Tensor::operator*(double scalar) const {
    Tensor result(*this);

    // Calcula el numero de valores
    size_t total = 1;
    for (size_t s : shape)
        total *= s;

    // Multiplica cada valor por el escalar
    for (size_t i = 0; i < total; i++)
        result.matriz_1d[i] *= scalar;

    return result;
}

//Pregunta 7: Modificacion de dimensiones
//Pregunta 7.1: View

//Metodo definido
Tensor Tensor::view(const vector<size_t>& shapenuevo) const {

    //Verificar limites de dimensiones
    if (shapenuevo.size() > 3) {
        throw invalid_argument("Sobrepasa el maximo de 3 dimensiones");
        // No se puede ingresar cosas como view{(2,2,2,2)}
    }

    // Numero de elementos del tensor original
    size_t total_anterior = 1;
    for (size_t s : shape) {
        total_anterior *= s; //Recorre el shape y multiplica al total_anterior
    }

    //Numero total de elementos del nuevo tensor
    size_t total_nuevo = 1;
    for (size_t s : shapenuevo) {
        total_nuevo *= s; //Recorre el shape y multiplica al total_nuevo
    }

    // Verificando compatibilidad
    if (total_anterior != total_nuevo) {
        throw invalid_argument("Numero de elementos incompatible");
        // Unicamente si los resultados son iguales se continuara
    }

    // Nuevo tensor
    Tensor result(*this);

    // Se cambia la forma del nuevo
    result.shape = shapenuevo;
    result.dim = shapenuevo.size();

    return result; // Se retorna el nuevo tensor
}

//Pregunta 7.2 Unsqueeze

//Metodo definido
Tensor Tensor::unsqueeze(size_t dimposicion) const {

    //Verificar limites de dimensiones
    if (shape.size() >= 3) {
        throw invalid_argument("Dimensions incompatible");
        // No se puede ingresar un tensor con mas de 3 dimensiones
    }

    // Verifica la existencia de la posición
    if (dimposicion > shape.size()) {
        throw invalid_argument("Dimensions incompatible");
    }

    // Copia del shape actual
    vector<size_t> shapenuevo = shape;  //Para no actualizar el tensor original

    // Se inserta una dimension de tamano 1 en la posicion indicada
    shapenuevo.insert(shapenuevo.begin() + dimposicion, 1);

    // Buevo tensor basado en el actual
    Tensor result(*this); //Tesnor copia

    // Actualiza dimensiones
    result.shape = shapenuevo;
    result.dim = shapenuevo.size();

    return result; // Se retorna el nuevo tensor
}

//Pregunta 8 Concatenacion
Tensor Tensor::concat(const vector<Tensor>& tensores, size_t dimposicion) {

    // Debe haber al menos un tensor
    if (tensores.empty()) {
        throw invalid_argument("No hay tensores para concatenar");
    }

    // Shape base
    vector<size_t> shapenuevo = tensores[0].shape;
    // Verifica dimension valida
    if (dimposicion >= shapenuevo.size()) {
        throw invalid_argument("Dimension incompatible");
    }

    // Calcular tamaño total en la dimension de concatenacion
    size_t suma_dim = 0;
    for (const Tensor& t : tensores) {
        // Verificar dimensiones
        for (size_t i = 0; i < shapenuevo.size(); i++) {
            if (i != dimposicion && t.shape[i] != shapenuevo[i]) {
                throw invalid_argument("Dimensiones incompatibles");
            }
        }
        suma_dim += t.shape[dimposicion]; //Suma el tamaño de la dimension
    }
    shapenuevo[dimposicion] = suma_dim; //Actualiza la dimension

    // Calcular total elementos
    size_t total = 1;
    for (size_t s : shapenuevo) {
        total *= s;
    }
    Tensor resultado(shapenuevo, vector<double>(total)); //Tensor resultado

    // Posicion copia
    size_t posicion = 0;

    // Copiar datos de cada tensor
    for (const Tensor& t : tensores) {
        size_t total_tensor = 1;
        for (size_t s : t.shape) {
            total_tensor *= s;
        }
        for (size_t i = 0; i < total_tensor; i++) {   //Copiar los elementos
            resultado.matriz_1d[posicion++] = t.matriz_1d[i];
        }
    }
    return move(resultado); // retorna tensor usando move
}

//Pregunta 9 - Funciones amigas predeterminadas

//Dot
Tensor dot(const Tensor& a, const Tensor& b) {

    // Verificar que tengan el mismo shape
    if (a.shape != b.shape) {
        throw invalid_argument("Dimensiones incompatibles");
    }

    // Calcular el numero total de elementos
    size_t total = 1;
    for (size_t s : a.shape) {
        total *= s; // Multiplica las dimensiones
    }

    double suma = 0.0; // Variable donde se acumlarara la suma

    // Se multiplica elemeneto x elemento, segun el indice
    for (size_t i = 0; i < total; i++) {
        suma += a.matriz_1d[i] * b.matriz_1d[i];
    }

    // Crear tensor escalar
    Tensor resultado({1}, vector<double>{suma});

    return move(resultado); // Retorna tensor usando move
}

//Matmul
Tensor matmul(const Tensor& a, const Tensor& b) {

    // Verificar que tengan las dimensiones adecuadas. (tensores bidimensionales)
    if (a.shape.size() !=2 && b.shape.size() !=2) {
        throw invalid_argument("Solo estan permitidas matrices 2D");
    }

    // Comprobar las dimensiones
    size_t filasA = a.shape[0];
    size_t columnasA = a.shape[1];
    size_t filasB = b.shape[0];
    size_t columnasB = b.shape[1];
    if (columnasA!=filasB) {
        throw invalid_argument("Dimensions incompatible");
    }

    //Nuevas dimensiones
    vector<size_t> shapenuevo = {filasA, columnasB}; // Nuevo shape
    size_t total = filasA * columnasB; // total elementos del resultado

    // Crear tensor escalar
    Tensor resultado(shapenuevo, vector<double>(total, 0.0));

    //Bucle for
    for (size_t i = 0; i < filasA; i++) {
        for (size_t j = 0; j < columnasB; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < columnasA; k++) {
                sum += a.matriz_1d[i*columnasA+k] * b.matriz_1d[k*columnasB + j];
            }
            resultado.matriz_1d[i*columnasB + j] = sum;
        }
    }
    return move(resultado); // Retorna tensor usando move
}