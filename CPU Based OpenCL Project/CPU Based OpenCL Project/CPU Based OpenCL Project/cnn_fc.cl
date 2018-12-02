

__kernel void cnn_fc(__global float * inputs, __global float * outputs, __global float * weights, __global float * biases, int D2, int D1) { 
	
	int i = get_global_id(0);
	int j;
	float sum = 0;
	
	for(j=0; j<D1; ++j) { 
		sum += inputs[j] * weights[i * D1 + j];
	}
	sum += biases[i];
	outputs[i] = (sum > 0 ? sum : 0);
}
