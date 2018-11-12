#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 1000
#define HEIGHT 1000

void makerandom(int *t) {
	int i;
	for (i = 0; i < WIDTH*HEIGHT; i++) {
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
	cl_mem Aobj = NULL, Bobj = NULL, Cobj = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_int err;

	int* result = (int*)malloc(sizeof(int)*WIDTH*HEIGHT);
	int i;

	FILE *fp;
	char fileName[] = "./kernel.cl";
	char *source_code;
	size_t source_size;

	source_code = get_source_code("kernel.cl", &source_size);

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
	Aobj = clCreateBuffer(context, 0, sizeof(int)*WIDTH*HEIGHT, NULL, &err);
	Bobj = clCreateBuffer(context, 0, sizeof(int)*WIDTH*HEIGHT, NULL, &err);
	Cobj = clCreateBuffer(context, 0, sizeof(int)*WIDTH*HEIGHT, NULL, &err);

	/*���� ����*/
	int* t1 = (int*)malloc(sizeof(int)*WIDTH*HEIGHT);
	int* t2 = (int*)malloc(sizeof(int)*WIDTH*HEIGHT);

	makerandom(t1);
	makerandom(t2);

	err = clEnqueueWriteBuffer(command_queue, Aobj, CL_TRUE, 0, sizeof(int)*WIDTH*HEIGHT, t1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, Bobj, CL_TRUE, 0, sizeof(int)*WIDTH*HEIGHT, t2, 0, NULL, NULL);

	int size = WIDTH;
	/*OpenCL Ŀ�� �Ķ���� ����*/
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&Aobj);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&Bobj);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&Cobj);


	/*OpenCL Ŀ�� ����*/
	size_t global_size[2] = { HEIGHT,WIDTH };
	size_t local_size[2] = { 2,2 };
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

	/*���� ����� �޸� ���ۿ��� ����*/
	err = clEnqueueReadBuffer(command_queue, Cobj, CL_TRUE, 0, sizeof(int)*WIDTH*HEIGHT, result, 0, NULL, NULL);

	/*��� ���*/
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			int sum = 0;
			for (int k = 0; k < WIDTH; k++) {
				sum += t1[i*WIDTH + k] * t2[k*WIDTH + j];
			}
			printf("Parallel [%d %d] = %d / Sequential %d \n", i, j, result[HEIGHT*i + j], sum);
		}
	}
}