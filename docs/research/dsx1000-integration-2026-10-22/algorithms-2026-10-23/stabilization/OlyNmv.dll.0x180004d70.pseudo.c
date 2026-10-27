// ev_4058c7c46e47e5ea935564cdab5f9c4ba9ce7eec417b8c045e43c7ca10049a5e
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

ulonglong OLYNMV_CreateWindow(int *param_1)

{
  uint uVar1;
  int iVar2;
  ulonglong uVar3;
  uint uVar4;
  undefined1 auStack_d8 [32];
  int local_b8;
  int local_b4;
  _OutputArray local_b0 [40];
  undefined8 local_88;
  undefined8 local_80;
  longlong local_78;
  int *piStack_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined8 local_58;
  undefined8 local_50;
  undefined8 *local_48;
  longlong *local_40;
  longlong local_38 [4];
  
                    /* 0x4d70  3  OLYNMV_CreateWindow */
  local_38[2] = DAT_18001a000 ^ (ulonglong)auStack_d8;
  local_48 = &local_80;
  local_78 = 0;
  piStack_70 = (int *)0x0;
  local_40 = local_38;
  local_38[1] = 0;
  local_38[0] = 0;
  local_88 = 0x42ff0000;
  local_80 = 0;
  local_58 = 0;
  local_60 = 0;
  local_68 = 0;
  local_50 = 0;
  if (DAT_18001a758 == '\0') {
    uVar1 = 0xfffffffe;
  }
  else {
    if (((((*param_1 != 0) && (param_1[1] != 0)) && (param_1[2] == 1)) &&
        ((param_1[3] == 4 && ((uint)(*param_1 * 4) <= (uint)param_1[4])))) &&
       (*(longlong *)(param_1 + 6) != 0)) {
      cv::_OutputArray::_OutputArray(local_b0,(Mat *)&local_88);
      local_b8 = *param_1;
      local_b4 = param_1[1];
      cv::createHanningWindow(local_b0,&local_b8,5);
      uVar1 = param_1[4];
      if ((uint)local_38[0] < (uint)param_1[4]) {
        uVar1 = (uint)local_38[0];
      }
      if (param_1[1] != 0) {
        uVar3 = 0;
        do {
          memcpy((void *)((ulonglong)(uint)((int)uVar3 * param_1[4]) + *(longlong *)(param_1 + 6)),
                 (void *)(uVar3 * local_38[0] + local_78),(ulonglong)uVar1);
          uVar4 = (int)uVar3 + 1;
          uVar3 = (ulonglong)uVar4;
        } while (uVar4 < (uint)param_1[1]);
      }
      uVar3 = 0;
      if ((piStack_70 != (int *)0x0) &&
         (iVar2 = cv::_interlockedExchangeAdd(piStack_70,-1), uVar3 = 0, iVar2 == 1)) {
        cv::Mat::deallocate((Mat *)&local_88);
      }
      goto LAB_180004edf;
    }
    uVar1 = 0xffffffff;
  }
  uVar3 = (ulonglong)uVar1;
LAB_180004edf:
  local_58 = 0;
  local_60 = 0;
  local_68 = 0;
  local_78 = 0;
  *(undefined4 *)local_48 = 0;
  piStack_70 = (int *)0x0;
  if (local_40 != local_38) {
    cv::fastFree(local_40);
  }
  return uVar3;
}

