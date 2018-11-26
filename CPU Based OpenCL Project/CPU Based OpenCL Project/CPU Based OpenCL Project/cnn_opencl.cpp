#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"

/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem dataBuffer = NULL, resultBuffer = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform_id = NULL;
cl_int err;

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;

	FILE *file = fopen(file_name, "r");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s ", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];

		if (buf[0] == '\n')
			cnt++;
	}

	source_code[length - cnt] = '\0';

	fclose(file);

	*len = length - cnt;

	return source_code;
}

void cnn_init() {
	/*
	 * TODO
	 * Initialize OpenCL objects as global variables. For example,
	 * clGetPlatformIDs(1, &platform, NULL);
	 */

	time_t s_time, e_time;

	char *source_code;
	size_t source_size;

	source_code = get_source_code("cnn_kernel.cl", &source_size);

	/*�÷���, ����̽� ������ ����*/
	//2��° �Ķ���ʹ� ���� �÷��� id
	err = clGetPlatformIDs(1, &platform_id, NULL);

	//3��° �Ķ���ʹ� ������ ����̽� ���� �����Ѵ�.
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

	/*OPenCL���ؽ�Ʈ ����*/
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

	/*Ŀ�ǵ� ť ����*/
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);

	/*�̸� �ε��� �ҽ� �ڵ�� Ŀ�� ���α׷��� ����*/
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);

	/*Ŀ�� ���α׷� ����*/
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	/*OpenCL Ŀ�� ����*/
	kernel = clCreateKernel(program, "cnn_kernel", &err);



	/*�޸� ���� ����*/
	dataBuffer = clCreateBuffer(context, 0, sizeof(int)*REDUCTION_SIZE, NULL, &err);
	resultBuffer = clCreateBuffer(context, 0, sizeof(int)* REDUCTION_SIZE / WORK_GROUP_SIZE, NULL, &err);
}

void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	/*
	 * TODO
	 * Implement here.
	 * Write classification results to labels and confidences.
	 * See "cnn_seq.c" if you don't know what to do.
	 */
}