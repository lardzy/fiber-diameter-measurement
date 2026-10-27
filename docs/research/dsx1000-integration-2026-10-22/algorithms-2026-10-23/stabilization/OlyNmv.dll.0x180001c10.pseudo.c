// ev_43c990c218361964d5b507513f44a5a8d404d7b940ddbfc07089c24d62071aae
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

undefined8 OLYNMV_CalcDstImageInfo(int *param_1,uint *param_2,uint param_3)

{
  uint uVar1;
  
                    /* 0x1c10  1  OLYNMV_CalcDstImageInfo */
  if (DAT_18001a758 == '\0') {
    return 0xfffffffe;
  }
  if (param_3 == 0) {
    param_3 = 1;
  }
  uVar1 = *param_2 / param_3;
  if ((uVar1 != 0) && (param_2[1] / param_3 != 0)) {
    param_1[1] = param_2[1] / param_3;
    *param_1 = uVar1;
    param_1[4] = uVar1 * 4;
    param_1[2] = 1;
    param_1[3] = 4;
    return 0;
  }
  return 0xffffffff;
}

