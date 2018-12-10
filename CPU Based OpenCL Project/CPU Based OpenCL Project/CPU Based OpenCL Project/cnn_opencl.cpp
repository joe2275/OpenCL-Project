//#include "stdafx.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "cnn.h"

/*
* TODO
* Define global variables here. For example,
* cl_platform_id platform;
*/

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
	printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
	exit(EXIT_FAILURE); \
	}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem *dataBuffer, *lastBuffer, resultBuffer[21], filterBuffer[21], biasBuffer[21];
cl_program program[3];
cl_kernel kernel[3];
cl_platform_id platform_id = NULL;
cl_event read_event;
cl_int err;

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;

	FILE *file = fopen(file_name, "rb");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s ", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	//for (int i = 0; i < length; i++) {
	//	buf[0] = source_code[i];

	//	if (buf[0] == '\n')
	//		cnt++;
	//}

	source_code[length] = '\0';

	fclose(file);

	*len = length - cnt;

	return source_code;
}

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	/*
	* TODO
	* Initialize OpenCL objects as global variables. For example,
	* clGetPlatformIDs(1, &platform, NULL);
	*/

	char * source_code[3];
	size_t source_size[3];

	source_code[0] = get_source_code("cnn_convolution.cl", &source_size[0]);
	source_code[1] = get_source_code("cnn_pooling.cl", &source_size[1]);
	source_code[2] = get_source_code("cnn_fc.cl", &source_size[2]);

	/*플랫폼, 디바이스 정보를 얻음*/
	//2번째 파라미터는 얻을 플랫폼 id
	err = clGetPlatformIDs(1, &platform_id, NULL);

	//3번째 파라미터는 얻어야할 디바이스 수를 지정한다.
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

	/*OPenCL컨텍스트 생성*/
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

	/*커맨드 큐 생성*/
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);

	/*미리 로드한 소스 코드로 커널 프로그램을 생성*/
	program[0] = clCreateProgramWithSource(context, 1, (const char**)&source_code[0], &source_size[0], &err);
	program[1] = clCreateProgramWithSource(context, 1, (const char**)&source_code[1], &source_size[1], &err);
	program[2] = clCreateProgramWithSource(context, 1, (const char**)&source_code[2], &source_size[2], &err);

	/*커널 프로그램 빌드*/
	err = clBuildProgram(program[0], 1, &device_id, NULL, NULL, NULL);
	err = clBuildProgram(program[1], 1, &device_id, NULL, NULL, NULL);
	err = clBuildProgram(program[2], 1, &device_id, NULL, NULL, NULL);
	/*OpenCL 커널 생성*/
	kernel[0] = clCreateKernel(program[0], "cnn_convolution", &err);
	kernel[1] = clCreateKernel(program[1], "cnn_pooling", &err);
	kernel[2] = clCreateKernel(program[2], "cnn_fc", &err);
}

void execute_convolution_layer(cl_mem data_buffer, cl_mem output_buffer, cl_mem filter_buffer, cl_mem bias_buffer, int D2, int D1, int N)
{
	err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&data_buffer);
	err = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&output_buffer);
	err = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void*)&filter_buffer);
	err = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), (void*)&bias_buffer);
	err = clSetKernelArg(kernel[0], 4, sizeof(int), &D2);
	err = clSetKernelArg(kernel[0], 5, sizeof(int), &D1);
	err = clSetKernelArg(kernel[0], 6, sizeof(int), &N);

	/*OpenCL 커널 실행*/
	size_t global_size = D2 * N * N;
	size_t local_size = 1;

	err = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global_size, &local_size, 0, NULL, NULL);
}

void execute_pooling_layer(cl_mem data_buffer, cl_mem output_buffer, int D, int N)
{
	err = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&data_buffer);
	err = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&output_buffer);
	err = clSetKernelArg(kernel[1], 2, sizeof(int), &D);
	err = clSetKernelArg(kernel[1], 3, sizeof(int), &N);

	/*OpenCL 커널 실행*/
	size_t global_size = D * N * N;
	size_t local_size = 1;

	err = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global_size, &local_size, 0, NULL, NULL);
}

void execute_fc_layer(cl_mem data_buffer, cl_mem output_buffer, cl_mem filter_buffer, cl_mem bias_buffer, int D2, int D1)
{
	err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void*)&data_buffer);
	err = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void*)&output_buffer);
	err = clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void*)&filter_buffer);
	err = clSetKernelArg(kernel[2], 3, sizeof(cl_mem), (void*)&bias_buffer);
	err = clSetKernelArg(kernel[2], 4, sizeof(int), &D2);
	err = clSetKernelArg(kernel[2], 5, sizeof(int), &D1);

	/*OpenCL 커널 실행*/
	size_t global_size = D2;
	size_t local_size = 1;

	err = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global_size, &local_size, 0, NULL, NULL);
}

static void softmax(float *output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float *fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}


void read_buffer(cl_mem outputBuffer, float * output, int num_output)
{
	/*실행 결과를 메모리 버퍼에서 얻음*/
	err = clEnqueueReadBuffer(command_queue, outputBuffer, CL_TRUE, 0, sizeof(float) * num_output, output, 0, NULL, NULL);
}
void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	/*
	* TODO
	* Implement here.
	* Write classification results to labels and confidences.
	* See "cnn_seq.c" if you don't know what to do.
	*/
	// slice the network into weights and biases
	float *w1_1, *b1_1, *w1_2, *b1_2;
	float *w2_1, *b2_1, *w2_2, *b2_2;
	float *w3_1, *b3_1, *w3_2, *b3_2, *w3_3, *b3_3;
	float *w4_1, *b4_1, *w4_2, *b4_2, *w4_3, *b4_3;
	float *w5_1, *b5_1, *w5_2, *b5_2, *w5_3, *b5_3;
	float *w1, *b1, *w2, *b2, *w3, *b3;
	w1_1 = network[0]; b1_1 = network[1];
	w1_2 = network[2]; b1_2 = network[3];
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	// allocate memory for output of each layer
	float *c1_1, *c1_2, *p1;
	float *c2_1, *c2_2, *p2;
	float *c3_1, *c3_2, *c3_3, *p3;
	float *c4_1, *c4_2, *c4_3, *p4;
	float *c5_1, *c5_2, *c5_3, *p5;
	float *fc1, *fc2, *fc3;
	c1_1 = alloc_layer(64 * 32 * 32);
	c1_2 = alloc_layer(64 * 32 * 32);
	p1 = alloc_layer(64 * 16 * 16);
	c2_1 = alloc_layer(128 * 16 * 16);
	c2_2 = alloc_layer(128 * 16 * 16);
	p2 = alloc_layer(128 * 8 * 8);
	c3_1 = alloc_layer(256 * 8 * 8);
	c3_2 = alloc_layer(256 * 8 * 8);
	c3_3 = alloc_layer(256 * 8 * 8);
	p3 = alloc_layer(256 * 4 * 4);
	c4_1 = alloc_layer(512 * 4 * 4);
	c4_2 = alloc_layer(512 * 4 * 4);
	c4_3 = alloc_layer(512 * 4 * 4);
	p4 = alloc_layer(512 * 2 * 2);
	c5_1 = alloc_layer(512 * 2 * 2);
	c5_2 = alloc_layer(512 * 2 * 2);
	c5_3 = alloc_layer(512 * 2 * 2);
	p5 = alloc_layer(512 * 1 * 1);
	fc1 = alloc_layer(512);
	fc2 = alloc_layer(512);
	fc3 = alloc_layer(10);
	dataBuffer = (cl_mem*)malloc(sizeof(cl_mem) * num_images);
	lastBuffer = (cl_mem*)malloc(sizeof(cl_mem) * num_images);
	for (int i = 0; i < num_images; i++) {
		dataBuffer[i] = clCreateBuffer(context, 0, sizeof(float) * 3 * 32 * 32, NULL, &err);
		lastBuffer[i] = clCreateBuffer(context, 0, sizeof(float) * 10, NULL, &err);
		err = clEnqueueWriteBuffer(command_queue, dataBuffer[i], CL_FALSE, 0, sizeof(float) * 3 * 32 * 32, images + i * 3 * 32 * 32, 0, NULL, NULL);
		err = clEnqueueWriteBuffer(command_queue, lastBuffer[i], CL_FALSE, 0, sizeof(float) * 10, fc3, 0, NULL, NULL);
	}

	/*메모리 버퍼 생성*/
	resultBuffer[0] = clCreateBuffer(context, 0, sizeof(float) * 64 * 32 * 32, NULL, &err);
	filterBuffer[0] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 3 * 64, NULL, &err);
	biasBuffer[0] = clCreateBuffer(context, 0, sizeof(float) * 64, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[0], CL_FALSE, 0, sizeof(float) * 64 * 32 * 32, c1_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[0], CL_FALSE, 0, sizeof(float) * 3 * 3 * 3 * 64, w1_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[0], CL_FALSE, 0, sizeof(float) * 64, b1_1, 0, NULL, NULL);

	resultBuffer[1] = clCreateBuffer(context, 0, sizeof(float) * 64 * 32 * 32, NULL, &err);
	filterBuffer[1] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 64 * 64, NULL, &err);
	biasBuffer[1] = clCreateBuffer(context, 0, sizeof(float) * 64, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[1], CL_FALSE, 0, sizeof(float) * 64 * 32 * 32, c1_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[1], CL_FALSE, 0, sizeof(float) * 3 * 3 * 64 * 64, w1_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[1], CL_FALSE, 0, sizeof(float) * 64, b1_2, 0, NULL, NULL);

	resultBuffer[2] = clCreateBuffer(context, 0, sizeof(float) * 64 * 16 * 16, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[2], CL_FALSE, 0, sizeof(float) * 64 * 16 * 16, p1, 0, NULL, NULL);

	resultBuffer[3] = clCreateBuffer(context, 0, sizeof(float) * 128 * 16 * 16, NULL, &err);
	filterBuffer[3] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 128 * 64, NULL, &err);
	biasBuffer[3] = clCreateBuffer(context, 0, sizeof(float) * 128, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[3], CL_FALSE, 0, sizeof(float) * 128 * 16 * 16, c2_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[3], CL_FALSE, 0, sizeof(float) * 3 * 3 * 128 * 64, w2_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[3], CL_FALSE, 0, sizeof(float) * 128, b2_1, 0, NULL, NULL);

	resultBuffer[4] = clCreateBuffer(context, 0, sizeof(float) * 128 * 16 * 16, NULL, &err);
	filterBuffer[4] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 128 * 128, NULL, &err);
	biasBuffer[4] = clCreateBuffer(context, 0, sizeof(float) * 128, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[4], CL_FALSE, 0, sizeof(float) * 128 * 16 * 16, c2_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[4], CL_FALSE, 0, sizeof(float) * 3 * 3 * 128 * 128, w2_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[4], CL_FALSE, 0, sizeof(float) * 128, b2_2, 0, NULL, NULL);

	resultBuffer[5] = clCreateBuffer(context, 0, sizeof(float) * 128 * 8 * 8, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[5], CL_FALSE, 0, sizeof(float) * 128 * 8 * 8, p2, 0, NULL, NULL);

	resultBuffer[6] = clCreateBuffer(context, 0, sizeof(float) * 256 * 8 * 8, NULL, &err);
	filterBuffer[6] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 256 * 128, NULL, &err);
	biasBuffer[6] = clCreateBuffer(context, 0, sizeof(float) * 256, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[6], CL_FALSE, 0, sizeof(float) * 256 * 8 * 8, c3_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[6], CL_FALSE, 0, sizeof(float) * 3 * 3 * 256 * 128, w3_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[6], CL_FALSE, 0, sizeof(float) * 256, b3_1, 0, NULL, NULL);

	resultBuffer[7] = clCreateBuffer(context, 0, sizeof(float) * 256 * 8 * 8, NULL, &err);
	filterBuffer[7] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 256 * 256, NULL, &err);
	biasBuffer[7] = clCreateBuffer(context, 0, sizeof(float) * 256, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[7], CL_FALSE, 0, sizeof(float) * 256 * 8 * 8, c3_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[7], CL_FALSE, 0, sizeof(float) * 3 * 3 * 256 * 256, w3_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[7], CL_FALSE, 0, sizeof(float) * 256, b3_2, 0, NULL, NULL);

	resultBuffer[8] = clCreateBuffer(context, 0, sizeof(float) * 256 * 8 * 8, NULL, &err);
	filterBuffer[8] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 256 * 256, NULL, &err);
	biasBuffer[8] = clCreateBuffer(context, 0, sizeof(float) * 256, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[8], CL_FALSE, 0, sizeof(float) * 256 * 8 * 8, c3_3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[8], CL_FALSE, 0, sizeof(float) * 3 * 3 * 256 * 256, w3_3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[8], CL_FALSE, 0, sizeof(float) * 256, b3_3, 0, NULL, NULL);

	resultBuffer[9] = clCreateBuffer(context, 0, sizeof(float) * 256 * 4 * 4, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[9], CL_FALSE, 0, sizeof(float) * 256 * 4 * 4, p3, 0, NULL, NULL);

	resultBuffer[10] = clCreateBuffer(context, 0, sizeof(float) * 512 * 4 * 4, NULL, &err);
	filterBuffer[10] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 512 * 256, NULL, &err);
	biasBuffer[10] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[10], CL_FALSE, 0, sizeof(float) * 512 * 4 * 4, c4_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[10], CL_FALSE, 0, sizeof(float) * 3 * 3 * 512 * 256, w4_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[10], CL_FALSE, 0, sizeof(float) * 512, b4_1, 0, NULL, NULL);

	resultBuffer[11] = clCreateBuffer(context, 0, sizeof(float) * 512 * 4 * 4, NULL, &err);
	filterBuffer[11] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
	biasBuffer[11] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[11], CL_FALSE, 0, sizeof(float) * 512 * 4 * 4, c4_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[11], CL_FALSE, 0, sizeof(float) * 3 * 3 * 512 * 512, w4_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[11], CL_FALSE, 0, sizeof(float) * 512, b4_2, 0, NULL, NULL);

	resultBuffer[12] = clCreateBuffer(context, 0, sizeof(float) * 512 * 4 * 4, NULL, &err);
	filterBuffer[12] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
	biasBuffer[12] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[12], CL_FALSE, 0, sizeof(float) * 512 * 4 * 4, c4_3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[12], CL_FALSE, 0, sizeof(float) * 3 * 3 * 512 * 512, w4_3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[12], CL_FALSE, 0, sizeof(float) * 512, b4_3, 0, NULL, NULL);

	resultBuffer[13] = clCreateBuffer(context, 0, sizeof(float) * 512 * 2 * 2, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[13], CL_FALSE, 0, sizeof(float) * 512 * 2 * 2, p4, 0, NULL, NULL);

	resultBuffer[14] = clCreateBuffer(context, 0, sizeof(float) * 512 * 2 * 2, NULL, &err);
	filterBuffer[14] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
	biasBuffer[14] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[14], CL_FALSE, 0, sizeof(float) * 512 * 2 * 2, c5_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[14], CL_FALSE, 0, sizeof(float) * 3 * 3 * 512 * 512, w5_1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[14], CL_FALSE, 0, sizeof(float) * 512, b5_1, 0, NULL, NULL);

	resultBuffer[15] = clCreateBuffer(context, 0, sizeof(float) * 512 * 2 * 2, NULL, &err);
	filterBuffer[15] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
	biasBuffer[15] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[15], CL_FALSE, 0, sizeof(float) * 512 * 2 * 2, c5_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[15], CL_FALSE, 0, sizeof(float) * 3 * 3 * 512 * 512, w5_2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[15], CL_FALSE, 0, sizeof(float) * 512, b5_2, 0, NULL, NULL);

	resultBuffer[16] = clCreateBuffer(context, 0, sizeof(float) * 512 * 2 * 2, NULL, &err);
	filterBuffer[16] = clCreateBuffer(context, 0, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
	biasBuffer[16] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[16], CL_FALSE, 0, sizeof(float) * 512 * 2 * 2, c5_3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[16], CL_FALSE, 0, sizeof(float) * 3 * 3 * 512 * 512, w5_3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[16], CL_FALSE, 0, sizeof(float) * 512, b5_3, 0, NULL, NULL);

	resultBuffer[17] = clCreateBuffer(context, 0, sizeof(float) * 512 * 1 * 1, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[17], CL_FALSE, 0, sizeof(float) * 512 * 1 * 1, p5, 0, NULL, NULL);

	resultBuffer[18] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	filterBuffer[18] = clCreateBuffer(context, 0, sizeof(float) * 512 * 512, NULL, &err);
	biasBuffer[18] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[18], CL_FALSE, 0, sizeof(float) * 512, fc1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[18], CL_FALSE, 0, sizeof(float) * 512 * 512, w1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[18], CL_FALSE, 0, sizeof(float) * 512, b1, 0, NULL, NULL);

	resultBuffer[19] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	filterBuffer[19] = clCreateBuffer(context, 0, sizeof(float) * 512 * 512, NULL, &err);
	biasBuffer[19] = clCreateBuffer(context, 0, sizeof(float) * 512, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[19], CL_FALSE, 0, sizeof(float) * 512, fc2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[19], CL_FALSE, 0, sizeof(float) * 512 * 512, w2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[19], CL_FALSE, 0, sizeof(float) * 512, b2, 0, NULL, NULL);

	resultBuffer[20] = clCreateBuffer(context, 0, sizeof(float) * 10, NULL, &err);
	filterBuffer[20] = clCreateBuffer(context, 0, sizeof(float) * 10 * 512, NULL, &err);
	biasBuffer[20] = clCreateBuffer(context, 0, sizeof(float) * 10, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer[20], CL_FALSE, 0, sizeof(float) * 10, fc3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterBuffer[20], CL_FALSE, 0, sizeof(float) * 10 * 512, w3, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, biasBuffer[20], CL_FALSE, 0, sizeof(float) * 10, b3, 0, NULL, NULL);

	for (int i = 0; i < num_images; ++i)
	{
		execute_convolution_layer(dataBuffer[i], resultBuffer[0], filterBuffer[0], biasBuffer[0], 64, 3, 32);
		execute_convolution_layer(resultBuffer[0], resultBuffer[1], filterBuffer[1], biasBuffer[1], 64, 64, 32);

		execute_pooling_layer(resultBuffer[1], resultBuffer[2], 64, 16);

		execute_convolution_layer(resultBuffer[2], resultBuffer[3], filterBuffer[3], biasBuffer[3], 128, 64, 16);
		execute_convolution_layer(resultBuffer[3], resultBuffer[4], filterBuffer[4], biasBuffer[4], 128, 128, 16);

		execute_pooling_layer(resultBuffer[4], resultBuffer[5], 128, 8);

		execute_convolution_layer(resultBuffer[5], resultBuffer[6], filterBuffer[6], biasBuffer[6], 256, 128, 8);
		execute_convolution_layer(resultBuffer[6], resultBuffer[7], filterBuffer[7], biasBuffer[7], 256, 256, 8);
		execute_convolution_layer(resultBuffer[7], resultBuffer[8], filterBuffer[8], biasBuffer[8], 256, 256, 8);

		execute_pooling_layer(resultBuffer[8], resultBuffer[9], 256, 4);

		execute_convolution_layer(resultBuffer[9], resultBuffer[10], filterBuffer[10], biasBuffer[10], 512, 256, 4);
		execute_convolution_layer(resultBuffer[10], resultBuffer[11], filterBuffer[11], biasBuffer[11], 512, 512, 4);
		execute_convolution_layer(resultBuffer[11], resultBuffer[12], filterBuffer[12], biasBuffer[12], 512, 512, 4);

		execute_pooling_layer(resultBuffer[12], resultBuffer[13], 512, 2);


		execute_convolution_layer(resultBuffer[13], resultBuffer[14], filterBuffer[14], biasBuffer[14], 512, 512, 2);
		execute_convolution_layer(resultBuffer[14], resultBuffer[15], filterBuffer[15], biasBuffer[15], 512, 512, 2);
		execute_convolution_layer(resultBuffer[15], resultBuffer[16], filterBuffer[16], biasBuffer[16], 512, 512, 2);

		execute_pooling_layer(resultBuffer[16], resultBuffer[17], 512, 1);

		execute_fc_layer(resultBuffer[17], resultBuffer[18], filterBuffer[18], biasBuffer[18], 512, 512);
		execute_fc_layer(resultBuffer[18], resultBuffer[19], filterBuffer[19], biasBuffer[19], 512, 512);
		execute_fc_layer(resultBuffer[19], lastBuffer[i], filterBuffer[20], biasBuffer[20], 10, 512);
		read_buffer(lastBuffer[i], fc3, 10);

		softmax(fc3, 10);

		labels[i] = find_max(fc3, 10);
		confidences[i] = fc3[labels[i]];
		printf("%d : %f\n", labels[i], confidences[i]);
	}

	//clFinish(command_queue);
	//for (int i = 0; i < num_images; i++) {
	//	read_buffer(lastBuffer[i], fc3, 10);

	//	softmax(fc3, 10);

	//	labels[i] = find_max(fc3, 10);
	//	confidences[i] = fc3[labels[i]];
	//}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);
}