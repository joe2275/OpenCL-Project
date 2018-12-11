

__kernel void cnn_convolution(__global float * inputs, __global float * outputs, __global float * filters, __global float * biases, int D2, int D1, int N) { 
	
	int i = get_global_id(0);
	int l_i = i%(N*N);
	int d_i = i/(N*N*D1);
	int l_d_i = (i/(N*N))%D1;
	int j, k, l, m, x, y;
	int l_x, l_y;
	float sum = 0;

	j = i / (N*N);
	x = l_i / N;
	y = l_i % N;

	sum = 0;
	for(l=0; l<3; ++l) {
		for(m=0; m<3; ++m) { 
			l_x = x + l - 1;
			l_y = y + m - 1;
			if (l_x >= 0 && l_x < N && l_y >= 0 && l_y < N) {
				sum += inputs[l_d_i * N * N + l_x * N + l_y] * filters[d_i * D1 * 3 * 3 + l_d_i * 3 * 3 + l * 3 + m];
			}
		}
	}

	outputs[i] = sum;
}
