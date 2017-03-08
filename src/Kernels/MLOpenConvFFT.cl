__constant float2 twiddles[31] = {
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(9.8078528040323043057924223830923438e-01f, -1.9509032201612824808378832130983938e-01f),
(float2)(9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f),
(float2)(7.0710678118654757273731092936941423e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-1.9509032201612819257263709005201235e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(7.0710678118654757273731092936941423e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(6.1232339957367660358688201472919830e-17f, -1.0000000000000000000000000000000000e+00f),
(float2)(-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(5.5557023301960228867102387084742077e-01f, -8.3146961230254523567140267914510332e-01f),
(float2)(-3.8268343236508972626808144923415966e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-9.8078528040323043057924223830923438e-01f, -1.9509032201612860890627132448571501e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(-9.2387953251128684950543856757576577e-01f, 3.8268343236508967075693021797633264e-01f),
(float2)(1.9509032201612833135051516819657991e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(-9.2387953251128673848313610506011173e-01f, -3.8268343236508989280153514300764073e-01f),
(float2)(-5.5557023301960217764872140833176672e-01f, 8.3146961230254523567140267914510332e-01f),
};


#define fptype float

#define fvect2 float2

#define C8Q  0.70710678118654752440084436210485f

__attribute__((always_inline)) void 
FwdRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}

__attribute__((always_inline)) void 
InvRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}

__attribute__((always_inline)) void 
FwdRad8B1(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + (fvect2)(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * (fvect2)((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + (fvect2)(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * (fvect2)((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}

__attribute__((always_inline)) void 
InvRad8B1(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + (fvect2)((*R7).y, -(*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * (fvect2)((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + (fvect2)((*R6).y, -(*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * (fvect2)((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}


__attribute__((always_inline)) void
FwdPassIN(uint me, uint inOffset, uint outOffset, __global const float *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{
	(*R0) = (float2)(0, 0);
	(*R1) = (float2)(0, 0);
	(*R2) = (float2)(0, 0);
	(*R3) = (float2)(0, 0);
	(*R4) = (float2)(0, 0);
	(*R5) = (float2)(0, 0);
	(*R6) = (float2)(0, 0);
	(*R7) = (float2)(0, 0);
	

	if((me%32) < 27)
	{	
	(*R0).x = bufIn[inOffset + ( (me/32)*12*27 + (me%32) +  0 + 0*54 )];
	(*R1).x = bufIn[inOffset + ( (me/32)*12*27 + (me%32) +  0 + 1*54 )];
	(*R2).x = bufIn[inOffset + ( (me/32)*12*27 + (me%32) +  0 + 2*54 )];
	(*R3).x = bufIn[inOffset + ( (me/32)*12*27 + (me%32) +  0 + 3*54 )];
	(*R4).x = bufIn[inOffset + ( (me/32)*12*27 + (me%32) +  0 + 4*54 )];
	(*R5).x = bufIn[inOffset + ( (me/32)*12*27 + (me%32) +  0 + 5*54 )];
	
	(*R0).y = bufIn[inOffset + ( (me/32)*12*27 + (me%32) + 27 + 0*54 )];
	(*R1).y = bufIn[inOffset + ( (me/32)*12*27 + (me%32) + 27 + 1*54 )];
	(*R2).y = bufIn[inOffset + ( (me/32)*12*27 + (me%32) + 27 + 2*54 )];
	(*R3).y = bufIn[inOffset + ( (me/32)*12*27 + (me%32) + 27 + 3*54 )];
	(*R4).y = bufIn[inOffset + ( (me/32)*12*27 + (me%32) + 27 + 4*54 )];
	(*R5).y = bufIn[inOffset + ( (me/32)*12*27 + (me%32) + 27 + 5*54 )];
	}
	
	if(me < 27)	
	{
	(*R6).x = bufIn[inOffset + ( me + 24*27 )];
	(*R6).y = bufIn[inOffset + ( me + 25*27 )];
	
	(*R7).x = bufIn[inOffset + ( me + 26*27 )];
	}

	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(me < 32)	
	{	
	bufOut[outOffset + me] = (float2)(0, 0);
	}

	bufOut[outOffset + ((1 +  0 + (me/32)*6 + 0)%17)*32 + ((2 + me)%32)] = (*R0);
	bufOut[outOffset + ((1 +  0 + (me/32)*6 + 1)%17)*32 + ((2 + me)%32)] = (*R1);
	bufOut[outOffset + ((1 +  0 + (me/32)*6 + 2)%17)*32 + ((2 + me)%32)] = (*R2);
	bufOut[outOffset + ((1 +  0 + (me/32)*6 + 3)%17)*32 + ((2 + me)%32)] = (*R3);
	bufOut[outOffset + ((1 +  0 + (me/32)*6 + 4)%17)*32 + ((2 + me)%32)] = (*R4);
	bufOut[outOffset + ((1 +  0 + (me/32)*6 + 5)%17)*32 + ((2 + me)%32)] = (*R5);	
	bufOut[outOffset + ((1 + 12 + (me/32)*2 + 0)%17)*32 + ((2 + me)%32)] = (*R6);
	bufOut[outOffset + ((1 + 12 + (me/32)*2 + 1)%17)*32 + ((2 + me)%32)] = (*R7);
}


__attribute__((always_inline)) void
FwdPassWE(uint me, uint inOffset, uint outOffset, __global const float *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	__local float *ldsf = (__local float *)bufOut;

	if(me < 25)
	{
	(*R0).x = bufIn[inOffset + me];
	ldsf[me] = (*R0).x;
	}

	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	(*R0) = (float2)(0, 0);
	(*R1) = (float2)(0, 0);
	(*R2) = (float2)(0, 0);
	(*R3) = (float2)(0, 0);
	(*R4) = (float2)(0, 0);
	(*R5) = (float2)(0, 0);
	(*R6) = (float2)(0, 0);
	(*R7) = (float2)(0, 0);
	
	
	if(me < 5)
	{
	(*R0).x = ldsf[4*5 + 4 - me];
	(*R0).y = ldsf[3*5 + 4 - me];
	(*R1).x = ldsf[2*5 + 4 - me];
	(*R1).y = ldsf[1*5 + 4 - me];
	(*R2).x = ldsf[0*5 + 4 - me];
	}

	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	bufOut[outOffset + ((me/32)*8 + 0)*32 + (me%32)] = (*R0);
	bufOut[outOffset + ((me/32)*8 + 1)*32 + (me%32)] = (*R1);
	bufOut[outOffset + ((me/32)*8 + 2)*32 + (me%32)] = (*R2);
	bufOut[outOffset + ((me/32)*8 + 3)*32 + (me%32)] = (*R3);
	bufOut[outOffset + ((me/32)*8 + 4)*32 + (me%32)] = (*R4);
	bufOut[outOffset + ((me/32)*8 + 5)*32 + (me%32)] = (*R5);	
	bufOut[outOffset + ((me/32)*8 + 6)*32 + (me%32)] = (*R6);
	bufOut[outOffset + ((me/32)*8 + 7)*32 + (me%32)] = (*R7);
	
	if(me < 32)	
	{	
	bufOut[outOffset + 512 + me] = (float2)(0, 0);
	}	
}

__attribute__((always_inline)) void
FwdPass0(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{



	(*R0) = bufIn[inOffset + ( me +  0 )];
	(*R1) = bufIn[inOffset + ( me +  4 )];
	(*R2) = bufIn[inOffset + ( me +  8 )];
	(*R3) = bufIn[inOffset + ( me + 12 )];
	(*R4) = bufIn[inOffset + ( me + 16 )];
	(*R5) = bufIn[inOffset + ( me + 20 )];
	(*R6) = bufIn[inOffset + ( me + 24 )];
	(*R7) = bufIn[inOffset + ( me + 28 )];



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);


	bufOut[outOffset + ( me*8 + 0 )] = (*R0);
	bufOut[outOffset + ( me*8 + 1 )] = (*R1);
	bufOut[outOffset + ( me*8 + 2 )] = (*R2);
	bufOut[outOffset + ( me*8 + 3 )] = (*R3);
	bufOut[outOffset + ( me*8 + 4 )] = (*R4);
	bufOut[outOffset + ( me*8 + 5 )] = (*R5);
	bufOut[outOffset + ( me*8 + 6 )] = (*R6);
	bufOut[outOffset + ( me*8 + 7 )] = (*R7);


}

__attribute__((always_inline)) void
FwdPass1(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	(*R0) = bufIn[inOffset + ( me*2 + 0 +  0 )];
	(*R4) = bufIn[inOffset + ( me*2 + 1 +  0 )];
	(*R1) = bufIn[inOffset + ( me*2 + 0 +  8 )];
	(*R5) = bufIn[inOffset + ( me*2 + 1 +  8 )];
	(*R2) = bufIn[inOffset + ( me*2 + 0 + 16 )];
	(*R6) = bufIn[inOffset + ( me*2 + 1 + 16 )];
	(*R3) = bufIn[inOffset + ( me*2 + 0 + 24 )];
	(*R7) = bufIn[inOffset + ( me*2 + 1 + 24 )];




	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);



	bufOut[outOffset + ( 2*me + 0 +  0 )] = (*R0);
	bufOut[outOffset + ( 2*me + 1 +  0 )] = (*R4);
	bufOut[outOffset + ( 2*me + 0 +  8 )] = (*R1);
	bufOut[outOffset + ( 2*me + 1 +  8 )] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 16 )] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 16 )] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 24 )] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 24 )] = (*R7);

	
}

__attribute__((always_inline)) void
FwdPass1b(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{
	(*R0) = bufIn[inOffset + ( me +  1  )];
	(*R1) = bufIn[inOffset + ( me +  5  )];
	(*R2) = bufIn[inOffset + ( me +  9  )];
	(*R3) = bufIn[inOffset + ( me + 13  )];
	(*R4) = bufIn[inOffset + ( 32 - ( me +  1  ) )];
	(*R5) = bufIn[inOffset + ( 32 - ( me +  5  ) )];
	(*R6) = bufIn[inOffset + ( 32 - ( me +  9  ) )];
	(*R7) = bufIn[inOffset + ( 32 - ( me + 13  ) )];	
	
	float2 dc;
	if(me < 1)
	{
		dc = bufIn[inOffset];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	bufOut[outOffset +  0 + ( me +  1  )] = (float2)( ((*R0).x + (*R4).x)*0.5, +((*R0).y - (*R4).y)*0.5 );
	bufOut[outOffset +  0 + ( me +  5  )] = (float2)( ((*R1).x + (*R5).x)*0.5, +((*R1).y - (*R5).y)*0.5 );
	bufOut[outOffset +  0 + ( me +  9  )] = (float2)( ((*R2).x + (*R6).x)*0.5, +((*R2).y - (*R6).y)*0.5 );
	bufOut[outOffset +  0 + ( me + 13  )] = (float2)( ((*R3).x + (*R7).x)*0.5, +((*R3).y - (*R7).y)*0.5 );

	bufOut[outOffset + 17 + ( me +  1  )] = (float2)( ((*R0).y + (*R4).y)*0.5, +(-(*R0).x + (*R4).x)*0.5 );
	bufOut[outOffset + 17 + ( me +  5  )] = (float2)( ((*R1).y + (*R5).y)*0.5, +(-(*R1).x + (*R5).x)*0.5 );
	bufOut[outOffset + 17 + ( me +  9  )] = (float2)( ((*R2).y + (*R6).y)*0.5, +(-(*R2).x + (*R6).x)*0.5 );
	bufOut[outOffset + 17 + ( me + 13  )] = (float2)( ((*R3).y + (*R7).y)*0.5, +(-(*R3).x + (*R7).x)*0.5 );	
	
	if(me < 1)
	{
		bufOut[outOffset +  0] = (float2)(dc.x, 0);
		bufOut[outOffset + 17] = (float2)(dc.y, 0);
	}

}

__attribute__((always_inline)) void
FwdPass2(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ( me +  0 )*17];
	(*R1) = bufIn[inOffset + ( me +  4 )*17];
	(*R2) = bufIn[inOffset + ( me +  8 )*17];
	(*R3) = bufIn[inOffset + ( me + 12 )*17];
	(*R4) = bufIn[inOffset + ( me + 16 )*17];
	(*R5) = bufIn[inOffset + ( me + 20 )*17];
	(*R6) = bufIn[inOffset + ( me + 24 )*17];
	(*R7) = bufIn[inOffset + ( me + 28 )*17];



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);

	bufOut[outOffset + ( me*8 + 0 )*17] = (*R0);
	bufOut[outOffset + ( me*8 + 1 )*17] = (*R1);
	bufOut[outOffset + ( me*8 + 2 )*17] = (*R2);
	bufOut[outOffset + ( me*8 + 3 )*17] = (*R3);
	bufOut[outOffset + ( me*8 + 4 )*17] = (*R4);
	bufOut[outOffset + ( me*8 + 5 )*17] = (*R5);
	bufOut[outOffset + ( me*8 + 6 )*17] = (*R6);
	bufOut[outOffset + ( me*8 + 7 )*17] = (*R7);


}

__attribute__((always_inline)) void
FwdPass3(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ( me*2 + 0 +  0 )*17];
	(*R4) = bufIn[inOffset + ( me*2 + 1 +  0 )*17];
	(*R1) = bufIn[inOffset + ( me*2 + 0 +  8 )*17];
	(*R5) = bufIn[inOffset + ( me*2 + 1 +  8 )*17];
	(*R2) = bufIn[inOffset + ( me*2 + 0 + 16 )*17];
	(*R6) = bufIn[inOffset + ( me*2 + 1 + 16 )*17];
	(*R3) = bufIn[inOffset + ( me*2 + 0 + 24 )*17];
	(*R7) = bufIn[inOffset + ( me*2 + 1 + 24 )*17];

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);

	bufOut[outOffset + ( 2*me + 0 +  0 )*17] = (*R0);
	bufOut[outOffset + ( 2*me + 1 +  0 )*17] = (*R4);
	bufOut[outOffset + ( 2*me + 0 +  8 )*17] = (*R1);
	bufOut[outOffset + ( 2*me + 1 +  8 )*17] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 16 )*17] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 16 )*17] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 24 )*17] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 24 )*17] = (*R7);

}


__attribute__((always_inline)) void
FwdPass4(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ( me + 0*64 )];
	(*R1) = bufIn[inOffset + ( me + 1*64 )];
	(*R2) = bufIn[inOffset + ( me + 2*64 )];
	(*R3) = bufIn[inOffset + ( me + 3*64 )];
	(*R4) = bufIn[inOffset + ( me + 4*64 )];
	(*R5) = bufIn[inOffset + ( me + 5*64 )];
	(*R6) = bufIn[inOffset + ( me + 6*64 )];
	(*R7) = bufIn[inOffset + ( me + 7*64 )];
	
	bufOut[outOffset + ( me + 0*64 )] = (*R0);
	bufOut[outOffset + ( me + 1*64 )] = (*R1);
	bufOut[outOffset + ( me + 2*64 )] = (*R2);
	bufOut[outOffset + ( me + 3*64 )] = (*R3);
	bufOut[outOffset + ( me + 4*64 )] = (*R4);
	bufOut[outOffset + ( me + 5*64 )] = (*R5);
	bufOut[outOffset + ( me + 6*64 )] = (*R6);
	bufOut[outOffset + ( me + 7*64 )] = (*R7);
	
	
	if(me < 32)
	{
		(*R0) = bufIn[inOffset + (512 + me)];
		bufOut[outOffset + (512 + me)] = (*R0);
	}

}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_fwd_in(__global const float * restrict gbIn, __global float2 * restrict gbOut )
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[544];

	__global const float *lwbIn;
	__global float2 *lwbOut;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	lwbIn = gbIn + batch*729;
	lwbOut = gbOut + batch*544;

	FwdPassIN(me, 0, 0, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);	
	FwdPass0(me%4, (me/4)*32, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	FwdPass1(me%4, (me/4)*32, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	FwdPass1b(me%4, (me/4)*32, (me/4)*34, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	
	barrier(CLK_LOCAL_MEM_FENCE);	
	FwdPass2(me%4, (me/4), (me/4), lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	FwdPass3(me%4, (me/4), (me/4), lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(me < 4)
	{
	FwdPass2(me%4, 16, 16, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	FwdPass3(me%4, 16, 16, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);	
	}
	
	FwdPass4(me, 0, 0, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);	
}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_fwd_we(__global const float * restrict gbIn, __global float2 * restrict gbOut )
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[544];

	__global const float *lwbIn;
	__global float2 *lwbOut;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	lwbIn = gbIn + batch*25;
	lwbOut = gbOut + 544*64*128 + batch*544;

	FwdPassWE(me, 0, 0, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);	
	FwdPass0(me%4, (me/4)*32, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	FwdPass1(me%4, (me/4)*32, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	FwdPass1b(me%4, (me/4)*32, (me/4)*34, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	
	barrier(CLK_LOCAL_MEM_FENCE);	
	FwdPass2(me%4, (me/4), (me/4), lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	FwdPass3(me%4, (me/4), (me/4), lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(me < 4)
	{
	FwdPass2(me%4, 16, 16, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	FwdPass3(me%4, 16, 16, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);	
	}
	
	FwdPass4(me, 0, 0, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);	
}




__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_transpose1(__global float2 * restrict gb)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[4096];

	uint iOffset;
	uint oOffset;
	__global const float2 *lwbIn;
	__global float2 *lwbOut;

	float2 R0;

	uint bm = batch%9;
	uint bd = batch/9;
	
	iOffset = bm*64 + bd*34816; 
	oOffset = 544*(192*128 + 64) + bm*528384 + bd*64; 
	
	lwbIn = gb + iOffset;
	lwbOut = gb + oOffset;
	
	if(bm == 8)
	{
		for(uint t=0; t<8; t++)
		{
			R0 = lwbIn[(me%32) + (me/32)*544 + t*4352];
			lds[(me%32)*64 + (me/32) + t*8] = R0;
		}	
	}
	else
	{
		for(uint t=0; t<16; t++)
		{
			R0 = lwbIn[(me%64) + (me/64)*544 + t*2176];
			lds[(me%64)*64 + (me/64) + t*4] = R0;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if(bm == 8)
	{
		for(uint t=0; t<8; t++)
		{
			R0 = lds[me + t*256];
			lwbOut[(me%64) + (me/64)*8256 + t*33024] = R0;
		}	
	}
	else
	{
		for(uint t=0; t<16; t++)
		{
			R0 = lds[me + t*256];
			lwbOut[(me%64) + (me/64)*8256 + t*33024] = R0;
		}
	}
}


__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_transpose2(__global float2 * restrict gb)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[4096];

	uint iOffset;
	uint oOffset;
	__global const float2 *lwbIn;
	__global float2 *lwbOut;

	float2 R0;

	uint bm = batch%9;
	uint bd = batch/9;
	
	iOffset = 544*64*128 + bm*64 + bd*34816; 
	oOffset = 544*(192*128 + 64) + 544*(64*128 + 64) + bm*790528 + bd*64;
	
	lwbIn = gb + iOffset;
	lwbOut = gb + oOffset;
	
	if(bm == 8)
	{
		for(uint t=0; t<8; t++)
		{
			R0 = lwbIn[(me%32) + (me/32)*544 + t*4352];
			lds[(me%32)*64 + (me/32) + t*8] = R0;
		}	
	}
	else
	{	
		for(uint t=0; t<16; t++)
		{
			R0 = lwbIn[(me%64) + (me/64)*544 + t*2176];
			lds[(me%64)*64 + (me/64) + t*4] = R0;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(bm == 8)
	{
		for(uint t=0; t<8; t++)
		{
			R0 = lds[me + t*256];
			lwbOut[(me%64) + (me/64)*12352 + t*49408] = R0;
		}	
	}
	else
	{	
		for(uint t=0; t<16; t++)
		{
			R0 = lds[me + t*256];
			lwbOut[(me%64) + (me/64)*12352 + t*49408] = R0;
		}
	}

}



__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_transpose3(__global float2 * restrict gb)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[4096];

	uint iOffset;
	uint oOffset;
	__global const float2 *lwbIn;
	__global float2 *lwbOut;

	float2 R0;

	uint bm = batch%9;
	uint bd = batch/9;
	
	iOffset = bm*1576960 + bd*64;	
	oOffset = 544*(192*128 + 64) + bm*64 + bd*34816; 

	
	lwbIn = gb + iOffset;
	lwbOut = gb + oOffset;
	
	if(bm == 8)
	{
		for(uint t=0; t<8; t++)
		{
			R0 = lwbIn[(me%64) + (me/64)*24640 + t*98560];
			lds[(me%64)*64 + (me/64) + t*4] = R0;
		}	
	}
	else
	{	
		for(uint t=0; t<16; t++)
		{
			R0 = lwbIn[(me%64) + (me/64)*24640 + t*98560];
			lds[(me%64)*64 + (me/64) + t*4] = R0;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(bm == 8)
	{
		for(uint t=0; t<8; t++)
		{
			R0 = lds[(me%32) + (me/32)*64 + t*512];
			lwbOut[(me%32) + (me/32)*544 + t*4352] = R0;
		}	
	}
	else
	{
		for(uint t=0; t<16; t++)
		{
			R0 = lds[me + t*256];
			lwbOut[(me%64) + (me/64)*544 + t*2176] = R0;
		}
	}

}




__attribute__((always_inline)) void
InvPassA(uint me, uint inOffset, uint outOffset, __global const float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ( me + 0*64 )];
	(*R1) = bufIn[inOffset + ( me + 1*64 )];
	(*R2) = bufIn[inOffset + ( me + 2*64 )];
	(*R3) = bufIn[inOffset + ( me + 3*64 )];
	(*R4) = bufIn[inOffset + ( me + 4*64 )];
	(*R5) = bufIn[inOffset + ( me + 5*64 )];
	(*R6) = bufIn[inOffset + ( me + 6*64 )];
	(*R7) = bufIn[inOffset + ( me + 7*64 )];
	
	bufOut[outOffset + ( me + 0*64 )] = (*R0);
	bufOut[outOffset + ( me + 1*64 )] = (*R1);
	bufOut[outOffset + ( me + 2*64 )] = (*R2);
	bufOut[outOffset + ( me + 3*64 )] = (*R3);
	bufOut[outOffset + ( me + 4*64 )] = (*R4);
	bufOut[outOffset + ( me + 5*64 )] = (*R5);
	bufOut[outOffset + ( me + 6*64 )] = (*R6);
	bufOut[outOffset + ( me + 7*64 )] = (*R7);
	
	
	if(me < 32)
	{
		(*R0) = bufIn[inOffset + (512 + me)];
		bufOut[outOffset + (512 + me)] = (*R0);
	}
	
}

__attribute__((always_inline)) void
InvPass0(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{



	(*R0) = bufIn[inOffset + ( me +  0 )*17];
	(*R1) = bufIn[inOffset + ( me +  4 )*17];
	(*R2) = bufIn[inOffset + ( me +  8 )*17];
	(*R3) = bufIn[inOffset + ( me + 12 )*17];
	(*R4) = bufIn[inOffset + ( me + 16 )*17];
	(*R5) = bufIn[inOffset + ( me + 20 )*17];
	(*R6) = bufIn[inOffset + ( me + 24 )*17];
	(*R7) = bufIn[inOffset + ( me + 28 )*17];



	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);


	bufOut[outOffset + (me*8 + 0 )*17] = (*R0);
	bufOut[outOffset + (me*8 + 1 )*17] = (*R1);
	bufOut[outOffset + (me*8 + 2 )*17] = (*R2);
	bufOut[outOffset + (me*8 + 3 )*17] = (*R3);
	bufOut[outOffset + (me*8 + 4 )*17] = (*R4);
	bufOut[outOffset + (me*8 + 5 )*17] = (*R5);
	bufOut[outOffset + (me*8 + 6 )*17] = (*R6);
	bufOut[outOffset + (me*8 + 7 )*17] = (*R7);

}

__attribute__((always_inline)) void
InvPass1(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	(*R0) = bufIn[inOffset + ( me*2 + 0 +  0 )*17];
	(*R4) = bufIn[inOffset + ( me*2 + 1 +  0 )*17];
	(*R1) = bufIn[inOffset + ( me*2 + 0 +  8 )*17];
	(*R5) = bufIn[inOffset + ( me*2 + 1 +  8 )*17];
	(*R2) = bufIn[inOffset + ( me*2 + 0 + 16 )*17];
	(*R6) = bufIn[inOffset + ( me*2 + 1 + 16 )*17];
	(*R3) = bufIn[inOffset + ( me*2 + 0 + 24 )*17];
	(*R7) = bufIn[inOffset + ( me*2 + 1 + 24 )*17];



	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);


	(*R0) = (*R0) * 3.1250000000000000e-02f;
	(*R4) = (*R4) * 3.1250000000000000e-02f;
	(*R1) = (*R1) * 3.1250000000000000e-02f;
	(*R5) = (*R5) * 3.1250000000000000e-02f;
	(*R2) = (*R2) * 3.1250000000000000e-02f;
	(*R6) = (*R6) * 3.1250000000000000e-02f;
	(*R3) = (*R3) * 3.1250000000000000e-02f;
	(*R7) = (*R7) * 3.1250000000000000e-02f;


	bufOut[outOffset + ( 2*me + 0 +  0 )*17] = (*R0);
	bufOut[outOffset + ( 2*me + 1 +  0 )*17] = (*R4);
	bufOut[outOffset + ( 2*me + 0 +  8 )*17] = (*R1);
	bufOut[outOffset + ( 2*me + 1 +  8 )*17] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 16 )*17] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 16 )*17] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 24 )*17] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 24 )*17] = (*R7);

}


__attribute__((always_inline)) void
InvPass1b(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset +  0 + ( me +  1  )];
	(*R1) = bufIn[inOffset +  0 + ( me +  5  )];
	(*R2) = bufIn[inOffset +  0 + ( me +  9  )];
	(*R3) = bufIn[inOffset +  0 + ( me + 13  )];
	(*R4) = bufIn[inOffset + 17 + ( me +  1  )];
	(*R5) = bufIn[inOffset + 17 + ( me +  5  )];
	(*R6) = bufIn[inOffset + 17 + ( me +  9  )];
	(*R7) = bufIn[inOffset + 17 + ( me + 13  )];	
	
	float2 dc;
	if(me < 1)
	{
		dc.x = bufIn[inOffset +  0].x;
		dc.y = bufIn[inOffset + 17].x;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	bufOut[outOffset + ( 32 - (me +  1  ) )] = (float2)( (*R0).x + (*R4).y, -(*R0).y + (*R4).x );
	bufOut[outOffset + ( 32 - (me +  5  ) )] = (float2)( (*R1).x + (*R5).y, -(*R1).y + (*R5).x );
	bufOut[outOffset + ( 32 - (me +  9  ) )] = (float2)( (*R2).x + (*R6).y, -(*R2).y + (*R6).x );
	bufOut[outOffset + ( 32 - (me + 13  ) )] = (float2)( (*R3).x + (*R7).y, -(*R3).y + (*R7).x );
	bufOut[outOffset +       ( me +  1  )]   = (float2)( (*R0).x - (*R4).y,  (*R0).y + (*R4).x );
	bufOut[outOffset +       ( me +  5  )]   = (float2)( (*R1).x - (*R5).y,  (*R1).y + (*R5).x );
	bufOut[outOffset +       ( me +  9  )]   = (float2)( (*R2).x - (*R6).y,  (*R2).y + (*R6).x );
	bufOut[outOffset +       ( me + 13  )]   = (float2)( (*R3).x - (*R7).y,  (*R3).y + (*R7).x );
	
	if(me < 1)
	{	
		bufOut[outOffset + 0] = dc;
	}

}


__attribute__((always_inline)) void
InvPass2(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ( me +  0 )];
	(*R1) = bufIn[inOffset + ( me +  4 )];
	(*R2) = bufIn[inOffset + ( me +  8 )];
	(*R3) = bufIn[inOffset + ( me + 12 )];
	(*R4) = bufIn[inOffset + ( me + 16 )];
	(*R5) = bufIn[inOffset + ( me + 20 )];
	(*R6) = bufIn[inOffset + ( me + 24 )];
	(*R7) = bufIn[inOffset + ( me + 28 )];


	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);


	bufOut[outOffset + (me*8 + 0 )] = (*R0);
	bufOut[outOffset + (me*8 + 1 )] = (*R1);
	bufOut[outOffset + (me*8 + 2 )] = (*R2);
	bufOut[outOffset + (me*8 + 3 )] = (*R3);
	bufOut[outOffset + (me*8 + 4 )] = (*R4);
	bufOut[outOffset + (me*8 + 5 )] = (*R5);
	bufOut[outOffset + (me*8 + 6 )] = (*R6);
	bufOut[outOffset + (me*8 + 7 )] = (*R7);

}


__attribute__((always_inline)) void
InvPass3(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ( me*2 + 0 +  0 )];
	(*R4) = bufIn[inOffset + ( me*2 + 1 +  0 )];
	(*R1) = bufIn[inOffset + ( me*2 + 0 +  8 )];
	(*R5) = bufIn[inOffset + ( me*2 + 1 +  8 )];
	(*R2) = bufIn[inOffset + ( me*2 + 0 + 16 )];
	(*R6) = bufIn[inOffset + ( me*2 + 1 + 16 )];
	(*R3) = bufIn[inOffset + ( me*2 + 0 + 24 )];
	(*R7) = bufIn[inOffset + ( me*2 + 1 + 24 )];



	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);

	bufOut[outOffset + ( 2*me + 0 +  0 )] = (*R0) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 1 +  0 )] = (*R4) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 0 +  8 )] = (*R1) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 1 +  8 )] = (*R5) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 0 + 16 )] = (*R2) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 1 + 16 )] = (*R6) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 0 + 24 )] = (*R3) * 3.1250000000000000e-02f;
	bufOut[outOffset + ( 2*me + 1 + 24 )] = (*R7) * 3.1250000000000000e-02f;


}


__attribute__((always_inline)) void
InvPassOUT(uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __global float *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{

	(*R0) = bufIn[inOffset + ((2 +  0 + (me/32)*6 + 0)%16)*32 + ((4 + me)%32)];
	(*R1) = bufIn[inOffset + ((2 +  0 + (me/32)*6 + 1)%16)*32 + ((4 + me)%32)];
	(*R2) = bufIn[inOffset + ((2 +  0 + (me/32)*6 + 2)%16)*32 + ((4 + me)%32)];
	(*R3) = bufIn[inOffset + ((2 +  0 + (me/32)*6 + 3)%16)*32 + ((4 + me)%32)];
	(*R4) = bufIn[inOffset + ((2 +  0 + (me/32)*6 + 4)%16)*32 + ((4 + me)%32)];
	(*R5) = bufIn[inOffset + ((2 +  0 + (me/32)*6 + 5)%16)*32 + ((4 + me)%32)];	
	(*R6) = bufIn[inOffset + ((2 + 12 + (me/32)*2 + 0)%16)*32 + ((4 + me)%32)];
	(*R7) = bufIn[inOffset + ((2 + 12 + (me/32)*2 + 1)%16)*32 + ((4 + me)%32)];
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if((me%32) < 27)
	{	
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) +  0 + 0*54 )] = (*R0).x;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) +  0 + 1*54 )] = (*R1).x;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) +  0 + 2*54 )] = (*R2).x;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) +  0 + 3*54 )] = (*R3).x;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) +  0 + 4*54 )] = (*R4).x;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) +  0 + 5*54 )] = (*R5).x;
	                                                          
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) + 27 + 0*54 )] = (*R0).y;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) + 27 + 1*54 )] = (*R1).y;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) + 27 + 2*54 )] = (*R2).y;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) + 27 + 3*54 )] = (*R3).y;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) + 27 + 4*54 )] = (*R4).y;
	bufOut[outOffset + ( (me/32)*12*27 + (me%32) + 27 + 5*54 )] = (*R5).y;
	}
	
	if(me < 27)	
	{
	bufOut[outOffset + ( me + 24*27 )] = (*R6).x;
	bufOut[outOffset + ( me + 25*27 )] = (*R6).y;
	
	bufOut[outOffset + ( me + 26*27 )] = (*R7).x;
	}	

}



__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_back(__global const float2 * restrict gbIn, __global float * restrict gbOut)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[544];

	__global const float2 *lwbIn;
	__global float *lwbOut;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;


	lwbIn = 544*(192*128 + 64) + gbIn + batch*544;
	lwbOut = gbOut + batch*729;

	InvPassA(me, 0, 0, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);	
	
	InvPass0(me%4, (me/4), (me/4), lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	InvPass1(me%4, (me/4), (me/4), lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(me<4)
	{
	InvPass0(me%4, 16, 16, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	InvPass1(me%4, 16, 16, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	}	
    
	barrier(CLK_LOCAL_MEM_FENCE);
	InvPass1b(me%4, (me/4)*34, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);	
	barrier(CLK_LOCAL_MEM_FENCE);	
	InvPass2(me%4, (me/4)*32, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	InvPass3(me%4, (me/4)*32, (me/4)*32, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	barrier(CLK_LOCAL_MEM_FENCE);
	InvPassOUT(me, 0, 0, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}


// =================================================================================================




/* Cijk_Alik_Bljk_CB_DU16_E01_E11_LU16_MT064_MT164_NLA04_NLB04_NLCA01_NLCB01_NLPA04_NLPB04_TT004_TT104_TTE04_WG016_WG116_WGE16 */

/* tile parameters */
#define WG_0I  16
#define WG_1J  16
#define UT_0I   4
#define UT_1J   4
#define MT_0I  64
#define MT_1J  64
#define UNROLL 16
#define PAD     1

/* num loads parallel and perpendicular to coalesced dimension */
#define NL_COAL_A 1
#define NL_COAL_B 1
#define NL_PERP_A 4
#define NL_PERP_B 4

#define LS_COAL_A (UNROLL/NL_COAL_A)
#define LS_PERP_A (MT_0I/NL_PERP_A)
#define LS_COAL_B (UNROLL/NL_COAL_B)
#define LS_PERP_B (MT_1J/NL_PERP_B)

/* global memory indices */
#define GLOBAL_C(IDX0I, IDX1J, IDXK) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J + (IDXK)*strideCK )
#define GLOBAL_A(IDXL, IDX0I, IDXK) ( (IDXL)*strideAL + (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GLOBAL_B(IDXL, IDX1J, IDXK) ( (IDXL)*strideBL + (IDX1J)*strideB1J + (IDXK)*strideBK )


/* data types */
#define TYPE_A     float2
#define TYPE_B     float2
#define TYPE_C     float2
//#define TYPE_ALPHA float2
//#define TYPE_BETA  float2
#define MAD(A,B,DST) mad(A,B,DST)

/* MADs */
#define TYPE_MAD(MULA,MULB,DST) \
  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); \
  DST.s0 = MAD( -MULA.s1, MULB.s1, DST.s0 ); \
  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); \
  DST.s1 = MAD(  MULA.s1, MULB.s0, DST.s1 );
#define TYPE_MAD_WRITE( DST, ALPHA, REG, BETA ) \
  /* (1) */ \
  /* (2) */ \
  /* (3) */ \
  DST = REG;

/* 4x4 micro-tile */
#define MICRO_TILE \
  rA[0] = localA[offA + 0*WG_0I]; \
  rA[1] = localA[offA + 1*WG_0I]; \
  rA[2] = localA[offA + 2*WG_0I]; \
  rA[3] = localA[offA + 3*WG_0I]; \
  rB[0] = localB[offB + 0*WG_1J]; \
  rB[1] = localB[offB + 1*WG_1J]; \
  rB[2] = localB[offB + 2*WG_1J]; \
  rB[3] = localB[offB + 3*WG_1J]; \
  offA += (MT_0I+PAD); \
  offB += (MT_1J+PAD); \
  TYPE_MAD(rA[0],rB[0],rC[0][0]); \
  TYPE_MAD(rA[0],rB[1],rC[0][1]); \
  TYPE_MAD(rA[0],rB[2],rC[0][2]); \
  TYPE_MAD(rA[0],rB[3],rC[0][3]); \
  TYPE_MAD(rA[1],rB[0],rC[1][0]); \
  TYPE_MAD(rA[1],rB[1],rC[1][1]); \
  TYPE_MAD(rA[1],rB[2],rC[1][2]); \
  TYPE_MAD(rA[1],rB[3],rC[1][3]); \
  TYPE_MAD(rA[2],rB[0],rC[2][0]); \
  TYPE_MAD(rA[2],rB[1],rC[2][1]); \
  TYPE_MAD(rA[2],rB[2],rC[2][2]); \
  TYPE_MAD(rA[2],rB[3],rC[2][3]); \
  TYPE_MAD(rA[3],rB[0],rC[3][0]); \
  TYPE_MAD(rA[3],rB[1],rC[3][1]); \
  TYPE_MAD(rA[3],rB[2],rC[3][2]); \
  TYPE_MAD(rA[3],rB[3],rC[3][3]); \
  mem_fence(CLK_LOCAL_MEM_FENCE);

/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideAL 1
#define strideBL 1


/* kernel */
__attribute__((reqd_work_group_size(WG_0I,WG_1J,1)))
__kernel void cgemm(
  __global float2 *gb,
  float2 const alpha,
  float2 const beta,
  unsigned int const offsetC,
  unsigned int const offsetA,
  unsigned int const offsetB,
  unsigned int const strideC1J,
  unsigned int const strideCK,
  unsigned int const strideA0I,
  unsigned int const strideAK,
  unsigned int const strideB1J,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK,
  unsigned int const sizeL ) {

  /* apply offsets */
  __global float2       * C = gb + offsetC;
  __global float2 const * A = gb + offsetA;
  __global float2 const * B = gb + offsetB;


  /* allocate registers */
  TYPE_C rC[UT_0I][UT_1J] = {{0}};
  TYPE_A rA[UT_0I];
  TYPE_B rB[UT_1J];

  /* allocate local memory */
  __local TYPE_A localA[UNROLL*(MT_0I+PAD)];
  __local TYPE_B localB[UNROLL*(MT_1J+PAD)];

  /* c indices (group) */
  unsigned int g0I = get_group_id(0); // d0, tensorA
  unsigned int g1J = get_group_id(1); // d1, tensorB
  unsigned int gK = ( get_group_id(2) ) % sizeK;

  /* c indices (local) */
  unsigned int l0I = get_local_id(0); // d0
  unsigned int l1J = get_local_id(1); // d1
  unsigned int loadSerial = l0I + l1J*WG_0I;
  unsigned int a0I = loadSerial/LS_COAL_A;
  unsigned int b1J = loadSerial/LS_COAL_B;

  /* unrolled summation index */
  unsigned int aL = loadSerial%LS_COAL_A;
  unsigned int bL = loadSerial%LS_COAL_B;

  /* other non-unrolled summation indices (all start at zero) */

  /* where will this thread read from global memory */
  A += GLOBAL_A( (unsigned long)aL, (unsigned long)a0I+g0I*MT_0I, (unsigned long)gK );
  B += GLOBAL_B( (unsigned long)bL, (unsigned long)b1J+g1J*MT_1J, (unsigned long)gK );

  /* where will this thread write to local memory */
  __local TYPE_A *lA = localA + a0I + aL*(MT_0I+PAD);
  __local TYPE_B *lB = localB + b1J + bL*(MT_1J+PAD);

  /* conditionals to guard against loading A out-of-bounds */
  bool condA_0_0 = ( a0I+g0I*MT_0I+0*LS_PERP_A >= size0I);
  bool condA_0_1 = ( a0I+g0I*MT_0I+1*LS_PERP_A >= size0I);
  bool condA_0_2 = ( a0I+g0I*MT_0I+2*LS_PERP_A >= size0I);
  bool condA_0_3 = ( a0I+g0I*MT_0I+3*LS_PERP_A >= size0I);

  /* conditionals to guard against loading B out-of-bounds */
  bool condB_0_0 = ( b1J+g1J*MT_1J+0*LS_PERP_B >= size1J);
  bool condB_0_1 = ( b1J+g1J*MT_1J+1*LS_PERP_B >= size1J);
  bool condB_0_2 = ( b1J+g1J*MT_1J+2*LS_PERP_B >= size1J);
  bool condB_0_3 = ( b1J+g1J*MT_1J+3*LS_PERP_B >= size1J);

  /* registers used for global -> local loads */
  TYPE_A a_0_0, a_0_1, a_0_2, a_0_3;
  TYPE_B b_0_0, b_0_1, b_0_2, b_0_3;

  /* iterate over summation indice(s) */
  unsigned int sumIterL = sizeL / UNROLL;
  do {

    barrier(CLK_LOCAL_MEM_FENCE);
    /* load A global -> local */
    a_0_0 = ( condA_0_0 ) ? (float2)(0.0, 0.0) : A[ 0*LS_COAL_A*strideAL + 0*LS_PERP_A*strideA0I];
    a_0_1 = ( condA_0_1 ) ? (float2)(0.0, 0.0) : A[ 0*LS_COAL_A*strideAL + 1*LS_PERP_A*strideA0I];
    a_0_2 = ( condA_0_2 ) ? (float2)(0.0, 0.0) : A[ 0*LS_COAL_A*strideAL + 2*LS_PERP_A*strideA0I];
    a_0_3 = ( condA_0_3 ) ? (float2)(0.0, 0.0) : A[ 0*LS_COAL_A*strideAL + 3*LS_PERP_A*strideA0I];

    /* load B global -> local */
    b_0_0 = ( condB_0_0 ) ? (float2)(0.0, 0.0) : B[ 0*LS_COAL_B*strideBL + 0*LS_PERP_B*strideB1J];
    b_0_1 = ( condB_0_1 ) ? (float2)(0.0, 0.0) : B[ 0*LS_COAL_B*strideBL + 1*LS_PERP_B*strideB1J];
    b_0_2 = ( condB_0_2 ) ? (float2)(0.0, 0.0) : B[ 0*LS_COAL_B*strideBL + 2*LS_PERP_B*strideB1J];
    b_0_3 = ( condB_0_3 ) ? (float2)(0.0, 0.0) : B[ 0*LS_COAL_B*strideBL + 3*LS_PERP_B*strideB1J];

    lA[ 0*LS_COAL_A*(MT_0I+PAD) + 0*LS_PERP_A ] = a_0_0;
    lA[ 0*LS_COAL_A*(MT_0I+PAD) + 1*LS_PERP_A ] = a_0_1;
    lA[ 0*LS_COAL_A*(MT_0I+PAD) + 2*LS_PERP_A ] = a_0_2;
    lA[ 0*LS_COAL_A*(MT_0I+PAD) + 3*LS_PERP_A ] = a_0_3;

    lB[ 0*LS_COAL_B*(MT_1J+PAD) + 0*LS_PERP_B ] = b_0_0;
    lB[ 0*LS_COAL_B*(MT_1J+PAD) + 1*LS_PERP_B ] = b_0_1;
    lB[ 0*LS_COAL_B*(MT_1J+PAD) + 2*LS_PERP_B ] = b_0_2;
    lB[ 0*LS_COAL_B*(MT_1J+PAD) + 3*LS_PERP_B ] = b_0_3;

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int offA = l0I; // d0
    unsigned int offB = l1J; // d1

    /* do fmas */
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE

    A += (long) strideAL*UNROLL;
    B += (long) strideBL*UNROLL;
  } while (--sumIterL > 0);

  /* which global Cij index */
  unsigned int globalC0I = g0I*MT_0I + l0I;
  unsigned int globalC1J = g1J*MT_1J + l1J;
  unsigned int globalCK = gK;

  /* write global C */
  float type_fma_tmp;
  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 0*WG_1J, (unsigned long) globalCK) ], alpha, rC[0][0], beta) } }
  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 1*WG_1J, (unsigned long) globalCK) ], alpha, rC[0][1], beta) } }
  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 2*WG_1J, (unsigned long) globalCK) ], alpha, rC[0][2], beta) } }
  if (globalC0I + 0*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 3*WG_1J, (unsigned long) globalCK) ], alpha, rC[0][3], beta) } }
  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 0*WG_1J, (unsigned long) globalCK) ], alpha, rC[1][0], beta) } }
  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 1*WG_1J, (unsigned long) globalCK) ], alpha, rC[1][1], beta) } }
  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 2*WG_1J, (unsigned long) globalCK) ], alpha, rC[1][2], beta) } }
  if (globalC0I + 1*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 3*WG_1J, (unsigned long) globalCK) ], alpha, rC[1][3], beta) } }
  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 0*WG_1J, (unsigned long) globalCK) ], alpha, rC[2][0], beta) } }
  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 1*WG_1J, (unsigned long) globalCK) ], alpha, rC[2][1], beta) } }
  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 2*WG_1J, (unsigned long) globalCK) ], alpha, rC[2][2], beta) } }
  if (globalC0I + 2*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 3*WG_1J, (unsigned long) globalCK) ], alpha, rC[2][3], beta) } }
  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 0*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 0*WG_1J, (unsigned long) globalCK) ], alpha, rC[3][0], beta) } }
  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 1*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 1*WG_1J, (unsigned long) globalCK) ], alpha, rC[3][1], beta) } }
  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 2*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 2*WG_1J, (unsigned long) globalCK) ], alpha, rC[3][2], beta) } }
  if (globalC0I + 3*WG_0I < size0I) {  if (globalC1J + 3*WG_1J < size1J) {  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 3*WG_1J, (unsigned long) globalCK) ], alpha, rC[3][3], beta) } }

}
#undef UNROLL
#undef WG_0I
#undef WG_1J
#undef UT_0I
#undef UT_1J
#undef MT_0I
#undef MT_1J
#undef NL_COAL_A
#undef NL_COAL_B
#undef NL_PERP_A
#undef NL_PERP_B
#undef LS_COAL_A
#undef LS_PERP_A
#undef LS_COAL_B
#undef LS_PERP_B
#undef GLOBAL_C
#undef GLOBAL_A
#undef GLOBAL_B
#undef TYPE_C
#undef TYPE_A
#undef TYPE_B
#undef MICRO_TILE
#undef strideC0I
#undef strideAL
#undef strideBL



/* Kernel Parameters
  ProblemType: Cijk_Alik_Bljk_CB
  VectorWidthGlobalLoad: 4
  DepthU: 16
  MacroTileMaxRatio: 2
  LoopDoWhile: True
  PadLDS: 1
  ThreadTileEdge: 4
  EdgeMultiKernel: False
  KernelSerial: True
  VectorWidthLocalLoad: 4
  AtomicAccumulate: False
  NumLoadsA: 4
  NumLoadsB: 4
  EdgeType: Branch
  ThreadTile1: 4
  ThreadTile0: 4
  WorkGroupSchedule: 1
  Edge1: True
  LoadMacInterleave: 4
  ThreadTileShape: 0
  WorkGroupEdge: 16
  VectorWidthLocalStore: 4
  KernelMaxSizes: [0, 0, 0]
  LoopTail: False
  WorkGroupShape: 0
  ProblemType: Cijk_Alik_Bljk_CB
  VectorWidthGlobalStore: 4
  Prefetch: False
  WorkGroupMapping: 1
  LoopUnroll: 16
  WorkGroup0: 16
  WorkGroup1: 16
  SplitU: 1
  MacroTile0: 64
  MacroTile1: 64
  Edge0: True
  NumLoadsCoalescedA: 1
  NumLoadsCoalescedB: 1
  NumLoadsPerpendicularA: 4
  NumLoadsPerpendicularB: 4
*/

