#pragma once 
typedef double k2c_float; 
#include "./include/k2c_declarations.h" 
const int example_input_shapes[1][5] = {
{ 0,8,32,0,0 },
}; 

void example(k2c_tensor* input_1_input, k2c_tensor* dense_3_output); 
void example_initialize(); 
void example_terminate(); 
