

__kernel void integral(__global double *g_sum ,__local double *l_fx, int TotalNum) 
{
	int i = get_global_id(0); 
	int l_i = get_local_id(0);
	double dx = 1.0 / TotalNum;
	double x = dx * i;

	l_fx[l_i] = (i < TotalNum) ? 3*x*x + 2*x + 1 : 0;
	l_fx[l_i] *= dx;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) 
	{ 
		if (l_i < p) l_fx[l_i] += l_fx[l_i+p]; 
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (l_i == 0) 
	{ 
		g_sum[get_group_id(0)] = l_fx[0];
	}
}