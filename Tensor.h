#include <vector>
#include <stdexcept>
#include <cmath>
using namespace std;

class TensorTransform;

class Tensor {
public:
    // P2.
    double* matriz_1d = nullptr;
    vector<size_t> shape;
    int dim;

    // P3 - P4. Constructores
    Tensor(const vector<size_t>& shape, const vector<double>& values);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    // P3.2. Funciones estaticas
    static Tensor random(const vector<size_t>& shape, double inicio, double final);
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor arange(int inicio, int final);

    // P4. Destructor
    ~Tensor();

    // P4. Operadores asignacion
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // P6. Operadores
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    // P5. Transform
    Tensor apply(const TensorTransform& transform) const;

    // P7. Modificacion de dimensiones
    Tensor view(const vector<size_t>& shapenuevo) const;
    Tensor unsqueeze(size_t dimposicion) const;

    // P8. Concatenacion
    static Tensor concat(const vector<Tensor>& tensores, size_t dimposicion);

    // P9. Funciones amigas
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
};

// Pregunta 5:  Polimorfismo y Transformaciones
// Pregunta 5.1 - Interfaz de Transformacion
class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

//Pregunta 5.2. Implementacion de ReLu
class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

//Pregunta 5.2. Implementacion de Sigmoid
class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

//P9. Funciones amigas
Tensor dot(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);