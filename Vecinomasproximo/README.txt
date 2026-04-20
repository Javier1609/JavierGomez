PROYECTO UAX - EJERCICIO 2.2 - CALCULO DEL VECINO MAS PROXIMO

1. DESCRIPCION
Este proyecto implementa el segundo ejercicio del PDF:
- lectura del dataset California Housing
- uso exclusivo de longitude y latitude
- calculo de distancias euclidianas
- almacenamiento en matriz triangular superior
- vecino mas cercano de una muestra dada
- paralelizacion con OpenMP
- comparacion static vs dynamic
- conteo de vecinos dentro de un radio R
- version alternativa sin construir toda la matriz

2. COMPILACION
gcc -O2 -fopenmp -Wall -Wextra -Iinclude src/main.c src/dataset.c src/triangular.c src/algorithms.c src/benchmark.c -o vecino_mas_proximo -lm

3. EJECUCION
./vecino_mas_proximo housing.csv 10 5000 first 0.1 8 64 12345
./vecino_mas_proximo housing.csv 10 5000 random 0.1 8 64 12345

4. ARGUMENTOS
<csv> <indice_objetivo> <N> <modo_submuestreo> [radio] [hilos] [chunk] [seed]

5. PRUEBAS RECOMENDADAS PARA LA MEMORIA
- N = 2000, 5000, 10000
- modo_submuestreo = first y random
- hilos = 1, 2, 4, 8
- schedule = static y dynamic

6. RESULTADOS A COMENTAR
- tiempo de construccion de la matriz
- tiempo de busqueda del vecino mas cercano
- tiempo de conteo dentro de radio R
- diferencia entre static y dynamic
- diferencia entre first y random
- diferencia entre matriz completa y version alternativa