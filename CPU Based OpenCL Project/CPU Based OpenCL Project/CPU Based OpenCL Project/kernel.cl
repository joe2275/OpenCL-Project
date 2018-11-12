


__kernel void vec_mul(__global int* A,__global int* B,__global int* C) {
   int col=get_global_id(0);
   int row=get_global_id(1);
   int sum=0;

   for(int k=0;k<1000;k++){
         sum += A [1000 * row + k]*B[k*1000 + col];
   }
   C[1000*row+col]=sum;
}