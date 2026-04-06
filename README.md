# PROGAMACION_TAREA1 -Tensor++ - Programación III

##Integrantes
- Jose Manuel Timana Carmona
- Jouse Yeremi Huaman Huamani

## Descripción
El proyecto implementa una libreria de Tensores que simula funciones de otras librerias como Numpy. Permitinedo trabajar tensores de hasta 3 diemnsiones y algunas funciones con esta. Tales como: Opeaciones matematicas, transformaciones y la simlacion de una red neuronal.
---

## Características

### Clase Tensor
- La clase principal
- Permite crera tensores de 1D, 2D y 3D
- Utiliza memoria dinamica
- Validación de dimensiones y datos  

### Creación de tensores
- zeros() → crea tensores con ceros  
- ones() → crea tensores con unos  
- random() → genera valores aleatorios  
- arange() → genera valores secuenciales  

### Gestión de memoria
- Constructor de copia
- Asignación por copia  
- Constructor de movimiento  
- Asignación por movimiento  
- Destructor  

### Transformaciones
Son las funciones que se aplican a un tensor para generar uno nuevo con ciertas caractersiticas.
- Clase para realiza transformaciones: TensorTransform  
- Subclases de TensorTransform:
  - ReLU  
  - Sigmoid  
- Método apply(): Para aplicar la tronsformacion.

### Operadores
Para permitir operaciones entre 2 tensores se sobrecargaron operadores. Permitiendo sumar, restar, multiplicar, y multiplciar por un escalar a los tensores.
- Suma (+)  
- Resta (-)  
- Multiplicación elemento a elemento (*)  
- Multiplicación por escalar  

### Manipulación de dimensiones
- view(): Pars cambiar la forma del tensor 
- unsqueeze(): Para agregar una dimensión  

### Operaciones adicionales
- concat(): Sirve para concatenar tensores  
- dot(): Funcion amiga que sirve para calcular y saber el producto punto  
- matmul(): Funcion amiga que sirve para saber la multiplicación matricial  

---

## Aplicación: Red Neuronal

El programa implementa el siguiente flujo:

1. Crear un tensor de entrada de dimensiones 1000 x 20 x 20  
2. Transformarlo a 1000 × 400 usando view  
3. Multiplicación con matriz 400 x 100  
4. Suma una matriz 1 x 100  
5. Aplicar la funcion ReLU  
6. Multiplicación por una matriz 100 x 10  
7. Suma una matriz 1 x 10  
8. Aplicar la funcion Sigmoid  
---

## Estructura del proyecto

- main.cpp : ejecución del programa  
- Tensor.cpp : implementación de la clase  
- Tensor.h : declaración de la clase  

---
