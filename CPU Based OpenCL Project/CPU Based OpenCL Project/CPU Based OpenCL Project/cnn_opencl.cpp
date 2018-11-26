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
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);

	/*커널 프로그램 빌드*/
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	/*OpenCL 커널 생성*/
	kernel = clCreateKernel(program, "cnn_kernel", &err);



	/*메모리 버퍼 생성*/
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