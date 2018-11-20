//#include <CL/cl.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <time.h>
//
//#define REDUCTION_SIZE 16777216
//#define WORK_GROUP_SIZE 256
//
//char *get_source_code(const char *file_name, size_t *len) {
//	char *source_code;
//	char buf[2] = "\0";
//	int cnt = 0;
//	size_t length;
//
//	FILE *file = fopen(file_name, "r");
//
//	if (file == NULL) {
//		printf("[%s:%d] Failed to open %s ", __FILE__, __LINE__, file_name);
//		exit(EXIT_FAILURE);
//	}
//
//	fseek(file, 0, SEEK_END);
//	length = (size_t)ftell(file);
//	rewind(file);
//
//	source_code = (char*)malloc(length + 1);
//	fread(source_code, length, 1, file);
//
//	for (int i = 0; i < length; i++) {
//		buf[0] = source_code[i];
//
//		if (buf[0] == '\n')
//			cnt++;
//	}
//
//	source_code[length - cnt] = '\0';
//
//	fclose(file);
//
//	*len = length - cnt;
//
//	return source_code;
//}
//
//int main()
//{
//	cl_device_id device_id = NULL;
//	cl_context context = NULL;
//	cl_command_queue command_queue = NULL;
//	cl_mem dataBuffer = NULL, resultBuffer = NULL;
//	cl_program program = NULL;
//	cl_kernel kernel = NULL;
//	cl_platform_id platform_id = NULL;
//	cl_int err;
//	cl_event read_event;
//
//	time_t s_time, e_time;
//	cl_ulong start_time, end_time;
//
//	char *source_code;
//	size_t source_size;
//
//	source_code = get_source_code("integral.cl", &source_size);
//
//	/*�÷���, ����̽� ������ ����*/
//	//2��° �Ķ���ʹ� ���� �÷��� id
//	err = clGetPlatformIDs(1, &platform_id, NULL);
//
//	//3��° �Ķ���ʹ� ������ ����̽� ���� �����Ѵ�.
//	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
//
//	/*OPenCL���ؽ�Ʈ ����*/
//	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
//
//	/*Ŀ�ǵ� ť ����*/
//	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
//
//	/*�̸� �ε��� �ҽ� �ڵ�� Ŀ�� ���α׷��� ����*/
//	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
//
//	/*Ŀ�� ���α׷� ����*/
//	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
//
//	/*OpenCL Ŀ�� ����*/
//	kernel = clCreateKernel(program, "integral", &err);
//
//
//
//	/*�޸� ���� ����*/
//	//dataBuffer = clCreateBuffer(context, 0, sizeof(double)*REDUCTION_SIZE, NULL, &err);
//	resultBuffer = clCreateBuffer(context, 0, sizeof(double)* REDUCTION_SIZE / WORK_GROUP_SIZE, NULL, &err);
//
//	/*���� ����*/
//	//double* t = (double*)malloc(sizeof(double)*REDUCTION_SIZE);
//
//	double* result = (double*)calloc(REDUCTION_SIZE/WORK_GROUP_SIZE, sizeof(double));
//
//	//err = clEnqueueWriteBuffer(command_queue, dataBuffer, CL_TRUE, 0, sizeof(double)*REDUCTION_SIZE, t, 0, NULL, NULL);
//	err = clEnqueueWriteBuffer(command_queue, resultBuffer, CL_TRUE, 0, sizeof(double)*REDUCTION_SIZE / WORK_GROUP_SIZE, result, 0, NULL, NULL);
//
//
//
//	int size = REDUCTION_SIZE;
//	/*OpenCL Ŀ�� �Ķ���� ����*/
//	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&resultBuffer);
//	err = clSetKernelArg(kernel, 1, sizeof(double)*WORK_GROUP_SIZE, (void*)NULL);
//	err = clSetKernelArg(kernel, 2, sizeof(int), (void*)&size);
//
//	/*OpenCL Ŀ�� ����*/
//	size_t global_size = REDUCTION_SIZE;
//	size_t local_size = WORK_GROUP_SIZE;
//
//	// start �ð� ����
//	s_time = clock();
//
//	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
//
//	/*���� ����� �޸� ���ۿ��� ����*/
//	err = clEnqueueReadBuffer(command_queue, resultBuffer, CL_TRUE, 0, sizeof(double)* REDUCTION_SIZE / WORK_GROUP_SIZE, result, 0, NULL, &read_event);
//
//	clFinish(command_queue);
//	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
//	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
//
//	double sum = 0;
//	/*��� ���*/
//	for (int i = 0; i < REDUCTION_SIZE / WORK_GROUP_SIZE; i++) {
//		sum += result[i];
//	}
//
//	// end �ð� ����
//	e_time = clock();
//
//	printf("[Parallel]\nIntegral value : %lf\nTime taken : %d ms\n", sum, e_time - s_time);
//	printf("Elapsed Time by Event : %lu ns \n\n", end_time - start_time);
//	// start �ð� ����
//	s_time = clock();
//
//	sum = 0;
//	for (int i = 0; i < REDUCTION_SIZE; i++)
//	{
//		double dx = 1.0 / REDUCTION_SIZE;
//		double x = dx * i;
//		double fx = 3 * x*x + 2 * x + 1;
//		
//		sum += fx * dx;
//	}
//
//	// end �ð� ����
//	e_time = clock();
//
//	printf("[Sequential]\nIntegral value : %lf\nTime taken : %d ms\n", sum, e_time - s_time);
//}