cmake_minimum_required(VERSION 3.20)
project(SimulacionDeLaDifusionDelCalor C)

set(CMAKE_C_STANDARD 99)

# 1. Buscamos el paquete de OpenMP
find_package(OpenMP REQUIRED)

# 2. Creamos el ejecutable
add_executable(SimulacionDeLaDifusionDelCalor main.c)

# 3. Si se encuentra OpenMP, aplicamos las banderas de forma manual y automática
if(OpenMP_C_FOUND)
    # Esto añade -fopenmp a la compilación
    target_compile_options(SimulacionDeLaDifusionDelCalor PUBLIC ${OpenMP_C_FLAGS})

    # Esto añade -fopenmp al enlazado (el paso que te está fallando)
    target_link_libraries(SimulacionDeLaDifusionDelCalor PUBLIC OpenMP::OpenMP_C)

    # Refuerzo extra para asegurar que las banderas se hereden
    set_target_properties(SimulacionDeLaDifusionDelCalor PROPERTIES
            COMPILE_FLAGS "${OpenMP_C_FLAGS}"
            LINK_FLAGS "${OpenMP_C_FLAGS}")
endif()
