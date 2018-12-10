

__kernel void cnn_pooling(__global float * inputs, __global float * outputs, int D, int N) { 
	
	int i = get_global_id(0);
	int l_i = i%(N*N);
	int k, l;

	int j = i / (N*N);
	int x = l_i / N;
	int y = l_i % N;
	float pixel;

	float max_num = 0;

	for(k=0; k<2; ++k) { 
		for(l=0; l<2; ++l) {
			pixel = inputs[j * 4 * N * N + x * 2 * N * 2 + k * N * 2 + y * 2 + l];
			max_num = (max_num < pixel ? pixel : max_num);
		}
	}

	outputs[i] = max_num;
}
