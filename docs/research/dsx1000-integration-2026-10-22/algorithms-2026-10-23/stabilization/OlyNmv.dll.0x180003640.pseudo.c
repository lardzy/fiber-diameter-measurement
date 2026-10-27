// ev_33a285b3b4a7a1886755bd9fc6d8b491d49ccc9e8fe254e6c624c3c9ab777dc3
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

double * FUN_180003640(double *param_1,undefined8 *param_2,int *param_3,double param_4,
                      undefined4 *param_5)

{
  int iVar1;
  int iVar2;
  longlong lVar3;
  int iVar4;
  longlong lVar5;
  undefined8 uVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  double dVar10;
  double dVar11;
  double dVar12;
  double dVar13;
  undefined4 uVar14;
  undefined4 uVar15;
  undefined1 auStack_408 [32];
  double *local_3e8;
  undefined8 local_3e0;
  longlong local_3d8;
  int local_3d0;
  int iStack_3cc;
  double dStack_3c8;
  undefined8 *local_3c0;
  undefined8 *puStack_3b8;
  double *local_3b0;
  undefined4 uStack_3a8;
  undefined4 uStack_3a4;
  undefined8 local_3a0;
  undefined8 local_398;
  double dStack_390;
  undefined8 *local_388;
  undefined8 *puStack_380;
  double *local_378;
  undefined8 uStack_370;
  undefined8 local_368;
  double local_358 [3];
  double dStack_340;
  undefined8 local_338;
  undefined8 uStack_330;
  void *local_328;
  undefined8 uStack_320;
  void *local_318;
  undefined8 uStack_310;
  void *local_308;
  undefined8 uStack_300;
  void *local_2f8;
  undefined8 uStack_2f0;
  undefined8 local_2e8;
  void *pvStack_2e0;
  undefined8 local_2d8;
  undefined8 local_2d0;
  undefined8 local_2c8;
  undefined8 local_2c0;
  undefined8 local_2b8;
  undefined8 uStack_2b0;
  undefined1 local_2a8;
  undefined8 *local_2a0;
  void *local_288;
  undefined8 uStack_280;
  void *local_278;
  undefined8 uStack_270;
  void *local_268;
  undefined8 uStack_260;
  void *local_258;
  undefined8 uStack_250;
  Mat local_1d8 [16];
  longlong local_1c8;
  int *local_1c0;
  undefined8 local_1b8;
  undefined8 local_1b0;
  undefined8 local_1a8;
  int *local_198;
  undefined1 *local_190;
  undefined1 local_188 [16];
  double local_178;
  double local_170;
  double local_168;
  double local_160;
  double local_158;
  double local_150;
  double local_148;
  double local_140;
  double local_138;
  undefined8 local_128;
  undefined8 local_120;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  undefined8 local_f0;
  undefined8 local_e8;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined8 local_c8;
  undefined8 local_c0;
  undefined8 local_b8;
  undefined8 local_b0;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  ulonglong local_88;
  
  local_3a0 = 0xfffffffffffffffe;
  local_88 = DAT_18001a000 ^ (ulonglong)auStack_408;
  (**(code **)*param_2)(param_2,local_1d8,0xffffffff);
  lVar3 = local_1c8;
  iVar4 = *local_198;
  iVar1 = local_198[1];
  lVar9 = (longlong)iVar1;
  iVar2 = *param_3;
  lVar8 = (longlong)iVar2;
  if ((0 < iVar2) && (iVar2 + 1 < iVar1)) {
    iVar2 = param_3[1];
    lVar7 = (longlong)iVar2;
    if ((0 < iVar2) && (iVar2 + 1 < iVar4)) {
      uVar14 = 0;
      uVar15 = 0xbff00000;
      local_d8 = 0xbff0000000000000;
      local_128 = 0xbff0000000000000;
      lVar5 = (lVar7 + -1) * lVar9 + lVar8;
      local_178 = (double)*(float *)(local_1c8 + -4 + lVar5 * 4);
      local_d0 = 0;
      local_120 = 0xbff0000000000000;
      local_170 = (double)*(float *)(local_1c8 + lVar5 * 4);
      dVar10 = 1.0;
      local_c8 = 0x3ff0000000000000;
      local_118 = 0xbff0000000000000;
      local_168 = (double)*(float *)(local_1c8 + 4 + lVar5 * 4);
      local_c0 = 0xbff0000000000000;
      local_110 = 0;
      lVar5 = lVar9 * lVar7 + lVar8;
      local_160 = (double)*(float *)(local_1c8 + -4 + lVar5 * 4);
      local_b8 = 0;
      local_108 = 0;
      local_158 = (double)*(float *)(local_1c8 + lVar5 * 4);
      local_b0 = 0x3ff0000000000000;
      local_100 = 0;
      local_150 = (double)*(float *)(local_1c8 + 4 + lVar5 * 4);
      local_a8 = 0xbff0000000000000;
      local_f8 = 0x3ff0000000000000;
      lVar5 = (lVar7 + 1) * lVar9 + lVar8;
      local_148 = (double)*(float *)(local_1c8 + -4 + lVar5 * 4);
      local_a0 = 0;
      local_f0 = 0x3ff0000000000000;
      local_140 = (double)*(float *)(local_1c8 + lVar5 * 4);
      local_98 = 0x3ff0000000000000;
      local_e8 = 0x3ff0000000000000;
      local_138 = (double)*(float *)(local_1c8 + 4 + lVar5 * 4);
      local_3d8 = lVar8;
      local_3e8 = malloc(0x18);
      if (local_3e8 == (double *)0x0) {
                    /* WARNING: Subroutine does not return */
        FUN_1800027a0();
      }
      local_3e0 = 3;
      lVar8 = lVar7 * lVar9 + local_3d8;
      dVar12 = (double)*(float *)(lVar3 + -4 + lVar8 * 4);
      dVar11 = (double)*(float *)(lVar3 + 4 + lVar8 * 4);
      dVar13 = (double)*(float *)(lVar3 + lVar8 * 4) * 4.0;
      *local_3e8 = ((dVar12 - dVar11) * -1.0) / ((dVar12 * 2.0 - dVar13) + dVar11 * 2.0);
      dVar12 = (double)*(float *)(local_1c8 +
                                 (((longlong)param_3[1] + -1) * lVar9 + (longlong)*param_3) * 4);
      dVar11 = (double)*(float *)(local_1c8 +
                                 (((longlong)param_3[1] + 1) * lVar9 + (longlong)*param_3) * 4);
      local_3e8[1] = ((dVar12 - dVar11) * -1.0) / ((dVar12 * 2.0 - dVar13) + dVar11 * 2.0);
      local_3e8[2] = 1.0;
      if ((param_4 < 1.0) && (dVar10 = param_4, param_4 <= 0.0)) {
        dVar10 = 0.0;
      }
      local_3c0 = &local_d8;
      puStack_3b8 = &local_128;
      local_3b0 = &local_178;
      uStack_3a8 = 3;
      uStack_3a4 = 9;
      local_398 = CONCAT44(iVar4,iVar1);
      uStack_370 = 0x900000003;
      local_368 = 0;
      local_358[0] = 100.0;
      local_358[1] = 1.97626258336499e-321;
      local_3d0 = iVar1;
      iStack_3cc = iVar4;
      dStack_3c8 = dVar10;
      dStack_390 = dVar10;
      local_388 = local_3c0;
      puStack_380 = puStack_3b8;
      local_378 = local_3b0;
      local_358[2] = sqrt(2.220446049250313e-16);
      local_338 = 0;
      uStack_330 = 0;
      local_328 = (void *)0x0;
      uStack_320 = 0;
      local_318 = (void *)0x0;
      uStack_310 = 0;
      local_308 = (void *)0x0;
      uStack_300 = 0;
      local_2f8 = (void *)0x0;
      uStack_2f0 = 0;
      local_2e8 = 0;
      pvStack_2e0 = (void *)0x0;
      local_2d8 = 0;
      local_2a0 = &local_398;
      local_288 = (void *)0x0;
      uStack_280 = 0;
      local_278 = (void *)0x0;
      uStack_270 = 0;
      local_268 = (void *)0x0;
      uStack_260 = 0;
      local_258 = (void *)0x0;
      uStack_250 = 0;
      local_2c0 = 0;
      local_2c8 = 0;
      local_2d0 = 0;
      local_2b8 = 0;
      uStack_2b0 = 0;
      local_2a8 = 0;
      dStack_340 = local_358[2];
      uVar6 = FUN_180005920(local_358,&local_3e8);
      if ((int)uVar6 != 0) {
        do {
          uVar6 = FUN_180005ca0(local_358,&local_3e8);
        } while ((int)uVar6 == -1);
      }
      dVar10 = *local_3e8;
      if (1.0 <= dVar10) {
LAB_180003ab3:
        dVar10 = 1.0;
      }
      else if (-1.0 < dVar10) {
        if (1.0 <= dVar10) goto LAB_180003ab3;
      }
      else {
        dVar10 = -1.0;
      }
      *local_3e8 = dVar10;
      dVar10 = local_3e8[1];
      if (1.0 <= dVar10) {
LAB_180003adb:
        uVar14 = 0;
        uVar15 = 0x3ff00000;
      }
      else if (-1.0 < dVar10) {
        if (1.0 <= dVar10) goto LAB_180003adb;
        uVar14 = SUB84(dVar10,0);
        uVar15 = (undefined4)((ulonglong)dVar10 >> 0x20);
      }
      local_3e8[1] = (double)CONCAT44(uVar15,uVar14);
      iVar4 = param_3[1];
      dVar10 = local_3e8[1];
      *param_1 = (double)*param_3 - *local_3e8;
      param_1[1] = (double)iVar4 - dVar10;
      free(local_258);
      free(local_268);
      free(local_278);
      free(local_288);
      free(pvStack_2e0);
      free(local_2f8);
      free(local_308);
      free(local_318);
      free(local_328);
      free(local_3e8);
      if ((local_1c0 != (int *)0x0) &&
         (iVar4 = cv::_interlockedExchangeAdd(local_1c0,-1), iVar4 == 1)) {
        cv::Mat::deallocate(local_1d8);
      }
      goto LAB_180003c01;
    }
  }
  *param_5 = 0xfffffffc;
  iVar4 = param_3[1];
  *param_1 = (double)*param_3;
  param_1[1] = (double)iVar4;
  if ((local_1c0 != (int *)0x0) && (iVar4 = cv::_interlockedExchangeAdd(local_1c0,-1), iVar4 == 1))
  {
    cv::Mat::deallocate(local_1d8);
  }
LAB_180003c01:
  local_1a8 = 0;
  local_1b0 = 0;
  local_1b8 = 0;
  local_1c8 = 0;
  *local_198 = 0;
  local_1c0 = (int *)0x0;
  if (local_190 != local_188) {
    cv::fastFree(local_190);
  }
  return param_1;
}

