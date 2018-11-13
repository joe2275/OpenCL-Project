#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

#include "bmpfuncs.h"

#define INPUT_FILE_NAME "input.bmp"
#define OUTPUT_FILE_NAME "output.bmp"

#define CHECK_ERROR(err) if(err != CL_SUCCESS) { printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); exit(EXIT_FAILURE); }

static float theta = 3.14159 / 6;

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

int main() {
	float sin_theta = sinf(theta);
	float cos_theta = cosf(theta);

	int image_width, image_height;
	float *input_image = readImage(INPUT_FILE_NAME, &image_width, &image_height);
	float *output_image = (float*)calloc(image_width * image_height, sizeof(float));

	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem Aobj = NULL, Bobj = NULL, Cobj = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_int err;

	FILE *fp;
	char *source_code;
	size_t source_size;

	source_code = get_source_code("image_rotation.cl", &source_size);

	/*플랫폼, 디바이스 정보를 얻음*/
	//2번째 파라미터는 얻을 플랫폼 id
	err = clGetPlatformIDs(1, &platform_id, NULL);
	CHECK_ERROR(err);

	//3번째 파라미터는 얻어야할 디바이스 수를 지정한다.
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
	CHECK_ERROR(err);

	/*OPenCL컨텍스트 생성*/
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	CHECK_ERROR(err);

	/*커맨드 큐 생성*/
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	CHECK_ERROR(err);

	/*미리 로드한 소스 코드로 커널 프로그램을 생성*/
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	/*커널 프로그램 빌드*/
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	CHECK_ERROR(err);

	/*OpenCL 커널 생성*/
	kernel = clCreateKernel(program, "image_rotate", &err);
	CHECK_ERROR(err);


	/*메모리 버퍼 생성*/
	Aobj = clCreateBuffer(context, 0, sizeof(float)*image_width*image_height, NULL, &err);
	CHECK_ERROR(err);
	Bobj = clCreateBuffer(context, 0, sizeof(float)*image_width*image_height, NULL, &err);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(command_queue, Aobj, CL_TRUE, 0, sizeof(float)*image_width*image_height, output_image, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(command_queue, Bobj, CL_TRUE, 0, sizeof(float)*image_width*image_height, input_image, 0, NULL, NULL);
	CHECK_ERROR(err);

	/*OpenCL 커널 파라미터 설정*/
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&Aobj);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&Bobj);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(int), (void*)&image_width);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(int), (void*)&image_height);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 4, sizeof(float), &sin_theta);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 5, sizeof(float), &cos_theta);
	CHECK_ERROR(err);

	/*OpenCL 커널 실행*/
	size_t global_size[2] = { image_width,image_height };
	size_t local_size[2] = { 2,2 };
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	/*실행 결과를 메모리 버퍼에서 얻음*/
	err = clEnqueueReadBuffer(command_queue, Aobj, CL_TRUE, 0, sizeof(float)*image_width*image_height, output_image, 0, NULL, NULL);
	CHECK_ERROR(err);

	/*결과 출력*/
	storeImage(output_image, OUTPUT_FILE_NAME, image_height, image_width, INPUT_FILE_NAME);
	//for (int i = 0; i < image_height; i++) {
	//	for (int j = 0; j < image_width; j++) {
	//		printf("[%d, %d] : %f\n", i, j, output_image[i*image_width + j]);
	//	}
	//}
	free(input_image);
	free(output_image);
	return 0;
}