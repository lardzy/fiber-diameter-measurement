// ev_b846845dfcadd616b18bba0b6b78988a3b70b2fdf25569b3130982b33124cde4
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

int OLYNMV_EstimateMV(double *param_1,double *param_2,double *param_3,uint *param_4,uint *param_5,
                     uint *param_6,uint param_7,char param_8,double param_9,char param_10)

{
  uint uVar1;
  uint uVar2;
  uint uVar3;
  uint uVar4;
  int iVar5;
  uint uVar6;
  undefined1 auStackY_238 [32];
  int local_1e8 [2];
  double local_1e0;
  double local_1d8;
  undefined8 local_1d0;
  _InputArray local_1c8 [32];
  _InputArray local_1a8 [32];
  _InputArray local_188 [32];
  Mat local_168 [16];
  undefined8 local_158;
  int *local_150;
  undefined8 local_148;
  undefined8 local_140;
  undefined8 local_138;
  undefined4 *local_128;
  undefined1 *local_120;
  undefined1 local_118 [16];
  Mat local_108 [16];
  undefined8 local_f8;
  int *local_f0;
  undefined8 local_e8;
  undefined8 local_e0;
  undefined8 local_d8;
  undefined4 *local_c8;
  undefined1 *local_c0;
  undefined1 local_b8 [16];
  Mat local_a8 [16];
  undefined8 local_98;
  int *local_90;
  undefined8 local_88;
  undefined8 local_80;
  undefined8 local_78;
  undefined4 *local_68;
  undefined1 *local_60;
  undefined1 local_58 [24];
  ulonglong local_40;
  
                    /* 0x4f30  4  OLYNMV_EstimateMV */
  local_1d0 = 0xfffffffffffffffe;
  local_40 = DAT_18001a000 ^ (ulonglong)auStackY_238;
  local_1e8[0] = 0;
  if (DAT_18001a758 == '\0') {
    local_1e8[0] = -2;
  }
  else {
    uVar6 = 1;
    if (1 < param_7) {
      uVar6 = param_7;
    }
    uVar1 = *param_4;
    if (((((((((uVar1 == 0) || (uVar2 = param_4[1], uVar2 == 0)) || (param_4[2] != 1)) ||
            ((param_4[3] != 4 || (param_4[4] < uVar1 * 4)))) || (*(longlong *)(param_4 + 6) == 0))
          || (((uVar3 = *param_5, uVar3 == 0 || (param_5[1] == 0)) ||
              ((param_5[2] != 1 ||
               (((param_5[3] != 4 || (param_5[4] < uVar3 * 4)) || (*(longlong *)(param_5 + 6) == 0))
               )))))) || ((uVar4 = *param_6, uVar4 == 0 || (param_6[1] == 0)))) ||
        ((param_6[2] != 1 ||
         (((param_6[3] != 4 || (param_6[4] < uVar4 * 4)) ||
          ((*(longlong *)(param_6 + 6) == 0 ||
           (((uVar1 != uVar3 || (uVar2 != param_5[1])) || (uVar1 != uVar4)))))))))) ||
       (uVar2 != param_6[1])) {
      local_1e8[0] = -1;
    }
    else {
      FUN_180002580((uint *)local_a8,param_4[1],*param_4,param_4,*(undefined8 *)(param_4 + 6),
                    (ulonglong)param_4[4]);
      FUN_180002580((uint *)local_108,param_5[1],*param_5,param_4,*(undefined8 *)(param_5 + 6),
                    (ulonglong)param_5[4]);
      FUN_180002580((uint *)local_168,param_6[1],*param_6,param_4,*(undefined8 *)(param_6 + 6),
                    (ulonglong)param_6[4]);
      cv::_InputArray::_InputArray(local_1a8,local_168);
      cv::_InputArray::_InputArray(local_1c8,local_108);
      cv::_InputArray::_InputArray(local_188,local_a8);
      FUN_180003d10(&local_1e0,(undefined8 *)local_188,(undefined8 *)local_1c8,
                    (undefined8 *)local_1a8,param_8,param_9,param_10,param_3,local_1e8);
      *param_1 = (double)uVar6 * local_1e0;
      *param_2 = (double)uVar6 * local_1d8;
      if ((local_150 != (int *)0x0) &&
         (iVar5 = cv::_interlockedExchangeAdd(local_150,-1), iVar5 == 1)) {
        cv::Mat::deallocate(local_168);
      }
      local_138 = 0;
      local_140 = 0;
      local_148 = 0;
      local_158 = 0;
      *local_128 = 0;
      local_150 = (int *)0x0;
      if (local_120 != local_118) {
        cv::fastFree(local_120);
      }
      if ((local_f0 != (int *)0x0) && (iVar5 = cv::_interlockedExchangeAdd(local_f0,-1), iVar5 == 1)
         ) {
        cv::Mat::deallocate(local_108);
      }
      local_d8 = 0;
      local_e0 = 0;
      local_e8 = 0;
      local_f8 = 0;
      *local_c8 = 0;
      local_f0 = (int *)0x0;
      if (local_c0 != local_b8) {
        cv::fastFree(local_c0);
      }
      if ((local_90 != (int *)0x0) && (iVar5 = cv::_interlockedExchangeAdd(local_90,-1), iVar5 == 1)
         ) {
        cv::Mat::deallocate(local_a8);
      }
      local_78 = 0;
      local_80 = 0;
      local_88 = 0;
      local_98 = 0;
      *local_68 = 0;
      local_90 = (int *)0x0;
      if (local_60 != local_58) {
        cv::fastFree(local_60);
      }
    }
  }
  return local_1e8[0];
}

