#!/bin/bash

opts_from_python=$(python3.5-config --cflags --ldflags | sed 's/-Wstrict-prototypes//g' | sed 's/-DNDEBUG//g')

g++ -std=c++11 -Wall -fPIC -O3 -shared \
    -I$CONDA_ENV_PATH/include $opts_from_python \
    rectangling.cpp \
    -l trng4 \
    -o rectangling.so
