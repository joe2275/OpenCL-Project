#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 16777216

void makerandom(int *t) {
	int i;
	for (i = 0; i < SIZE; i++) {
		*(t + i) = rand();
	}
}

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

int main()
{
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem dataBuffer = NULL, resultBuffer = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_int err;

	char *source_code;
	size_t source_size;

	source_code = get_source_code("sum_reduction.cl", &source_size);

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
	kernel = clCreateKernel(program, "vec_mul", &err);



	/*�޸� ���� ����*/
	dataBuffer = clCreateBuffer(context, 0, sizeof(int)*SIZE, NULL, &err);
	resultBuffer = clCreateBuffer(context, 0, sizeof(int)* 65536, NULL, &err);

	/*���� ����*/
	int* t = (int*)malloc(sizeof(int)*SIZE);

	int* result = (int*)calloc(65536, sizeof(int));

	makerandom(t);

	err = clEnqueueWriteBuffer(command_queue, dataBuffer, CL_TRUE, 0, sizeof(int)*SIZE, t, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, resultBuffer, CL_TRUE, 0, sizeof(int)*SIZE, result, 0, NULL, NULL);

	int size = SIZE;
	/*OpenCL Ŀ�� �Ķ���� ����*/
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dataBuffer);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&resultBuffer);
	err = clSetKernelArg(kernel, 2, sizeof(int)*65536, (void*)NULL);
	err = clSetKernelArg(kernel, 3, sizeof(int), (void*)&size);

	/*OpenCL Ŀ�� ����*/
	size_t global_size = SIZE;
	size_t local_size = 256;
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

	/*���� ����� �޸� ���ۿ��� ����*/
	err = clEnqueueReadBuffer(command_queue, resultBuffer, CL_TRUE, 0, sizeof(int)* 65536, result, 0, NULL, NULL);

	/*��� ���*/
	for (int i = 0; i < 65536; i++) {
		printf("[%d] : %d\n", i, result[i]);
	}
}