

__kernel void cnn_convolution(__global float * inputs, __global float * outputs, __global float * filters, __global float * biases, int D2, int D1, int N) { 
	
	int i = get_global_id(0);

	int j;

	for(j=0; j<D2; j++) {
		float * conv_input = inputs + N * N * i;
		float * conv_output = outputs + N * N * j;
		float * conv_filter = filters + 3 * 3 * (j * D1 + i);

		int i, j, k, l;
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				float sum = 0;
				for (k = 0; k < 3; k++) {
					for (l = 0; l < 3; l++) {
						int x = i + k - 1;
						int y = j + l - 1;
						if (x >= 0 && x < N && y >= 0 && y < N)
							sum += input[x * N + y] * filter[k * 3 + l];
					}
				}
				output[i * N + j] += sum;
			}
		}
	}
	
}
