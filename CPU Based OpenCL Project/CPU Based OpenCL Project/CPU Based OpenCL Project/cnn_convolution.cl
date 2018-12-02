

__kernel void cnn_convolution(__global float * inputs, __global float * outputs, __global float * filters, __global float * biases, int D2, int D1, int N) { 
	
	int i = get_global_id(0);
	int l_i = get_local_id(0);
	int j, k, l, m, x, y;
	int l_x, l_y;
	float sum = 0;

	j = i / (N*N);
	x = l_i / N;
	y = l_i % N;

	sum = 0;
	for(k=0; k<D1; ++k) { 
		for(l=0; l<3; ++l) {
			for(m=0; m<3; ++m) { 
				l_x = x + l - 1;
				l_y = y + m - 1;

				if (l_x >= 0 && l_x < N && l_y >= 0 && l_y < N) {
					sum += inputs[k * N * N + l_x * N + l_y] * filters[j * D1 * 3 * 3 + k * 3 * 3 + l * 3 + m];
				}
			}
		}
	}

	sum += biases[j];
	outputs[i] = (sum > 0 ? sum : -sum);
}
