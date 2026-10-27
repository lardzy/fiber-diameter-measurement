// ev_fd20fcabd92215ad27a65c1ee90e7ab34b1d0694b76d0f383481568736005cec
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180002ad0(undefined8 *param_1,double param_2)

{
  uint uVar1;
  uint uVar2;
  double dVar3;
  longlong lVar4;
  int iVar5;
  undefined4 *puVar6;
  int iVar7;
  ulonglong uVar8;
  longlong lVar9;
  longlong lVar10;
  ulonglong uVar11;
  double dVar12;
  undefined1 auStack_c8 [32];
  Mat local_a8 [16];
  longlong local_98;
  int *local_90;
  undefined8 local_88;
  undefined8 local_80;
  undefined8 local_78;
  uint *local_68;
  longlong *local_60;
  longlong local_58 [2];
  ulonglong local_48;
  
  local_48 = DAT_18001a000 ^ (ulonglong)auStack_c8;
  dVar3 = 1.0;
  if (param_2 < 1.0) {
    dVar12 = 0.1;
    dVar3 = 0.1;
    if (0.1 < param_2) goto LAB_180002b2b;
  }
  param_2 = dVar3;
  dVar12 = param_2;
LAB_180002b2b:
  (**(code **)*param_1)(dVar12,local_a8,0xffffffff);
  uVar1 = local_68[1];
  uVar2 = *local_68;
  iVar7 = (int)((double)(int)uVar1 * param_2 * 0.5);
  iVar5 = (int)((double)(int)uVar2 * param_2 * 0.5);
  if (0 < (int)uVar2) {
    uVar8 = (ulonglong)uVar2;
    lVar9 = 0;
    do {
      lVar10 = lVar9 * local_58[0] + local_98;
      if (0 < (int)uVar1) {
        puVar6 = (undefined4 *)(lVar10 + 4);
        lVar4 = 0;
        uVar11 = (ulonglong)uVar1;
        do {
          if (((iVar7 <= lVar4) && (lVar4 < (int)(uVar1 - iVar7))) ||
             ((iVar5 <= lVar9 && (lVar9 < (int)(uVar2 - iVar5))))) {
            *(undefined4 *)(lVar10 + lVar4 * 8) = 0;
            *puVar6 = 0;
          }
          lVar4 = lVar4 + 1;
          puVar6 = puVar6 + 2;
          uVar11 = uVar11 - 1;
        } while (uVar11 != 0);
      }
      lVar9 = lVar9 + 1;
      uVar8 = uVar8 - 1;
    } while (uVar8 != 0);
  }
  if ((local_90 != (int *)0x0) && (iVar5 = cv::_interlockedExchangeAdd(local_90,-1), iVar5 == 1)) {
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
  return;
}

