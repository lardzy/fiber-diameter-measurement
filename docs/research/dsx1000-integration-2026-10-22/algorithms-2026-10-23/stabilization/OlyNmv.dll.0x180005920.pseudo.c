// ev_998162eaf720507ee33d26c758717af3a583d221aa306f561a9713bbdf6a3d68
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

undefined8 FUN_180005920(double *param_1,undefined8 *param_2)

{
  double dVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  void *pvVar4;
  double *pdVar5;
  undefined8 uVar6;
  longlong lVar7;
  ulonglong uVar8;
  double dVar9;
  
  dVar9 = (double)param_2[1];
  lVar7 = 0;
  param_1[0x18] = dVar9;
  param_1[0x19] = (double)(longlong)*(int *)((longlong)param_1[0x17] + 0x2c);
  if (dVar9 != param_1[0x1b]) {
    free((void *)param_1[0x1a]);
    if (dVar9 == 0.0) {
      param_1[0x1a] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0((ulonglong)dVar9);
      param_1[0x1a] = (double)pvVar4;
    }
  }
  param_1[0x1b] = dVar9;
  dVar9 = param_1[0x18];
  if (dVar9 != param_1[0x1d]) {
    free((void *)param_1[0x1c]);
    if (dVar9 == 0.0) {
      param_1[0x1c] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0((ulonglong)dVar9);
      param_1[0x1c] = (double)pvVar4;
    }
  }
  param_1[0x1d] = dVar9;
  dVar9 = param_1[0x18];
  if (dVar9 != param_1[0x1f]) {
    free((void *)param_1[0x1e]);
    if (dVar9 == 0.0) {
      param_1[0x1e] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0((ulonglong)dVar9);
      param_1[0x1e] = (double)pvVar4;
    }
  }
  param_1[0x1f] = dVar9;
  dVar9 = param_1[0x19];
  if (dVar9 != param_1[0x21]) {
    free((void *)param_1[0x20]);
    if (dVar9 == 0.0) {
      param_1[0x20] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0((ulonglong)dVar9);
      param_1[0x20] = (double)pvVar4;
    }
  }
  param_1[0x21] = dVar9;
  dVar9 = param_1[0x19];
  if (dVar9 != param_1[7]) {
    free((void *)param_1[6]);
    if (dVar9 == 0.0) {
      param_1[6] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0((ulonglong)dVar9);
      param_1[6] = (double)pvVar4;
    }
  }
  param_1[7] = dVar9;
  dVar9 = param_1[0x19];
  dVar1 = param_1[0x18];
  if ((dVar9 != 0.0) && (dVar1 != 0.0)) {
    auVar2._8_8_ = 0;
    auVar2._0_8_ = SUB168(SEXT816(0x7fffffffffffffff),8);
    if (SUB168((auVar2 << 0x40 | ZEXT816(0x7fffffffffffffff)) / SEXT816((longlong)dVar1),0) <
        (longlong)dVar9) {
                    /* WARNING: Subroutine does not return */
      FUN_1800027a0();
    }
    if ((dVar1 != 0.0) &&
       (auVar3._8_8_ = 0, auVar3._0_8_ = SUB168(SEXT816(0x7fffffffffffffff),8),
       SUB168((auVar3 << 0x40 | ZEXT816(0x7fffffffffffffff)) / SEXT816((longlong)dVar1),0) <
       (longlong)dVar9)) {
                    /* WARNING: Subroutine does not return */
      FUN_1800027a0();
    }
  }
  uVar8 = (longlong)dVar9 * (longlong)dVar1;
  if (uVar8 - (longlong)param_1[0xd] * (longlong)param_1[0xe] != 0) {
    free((void *)param_1[0xc]);
    if (uVar8 == 0) {
      param_1[0xc] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0(uVar8);
      param_1[0xc] = (double)pvVar4;
    }
  }
  param_1[0xd] = dVar9;
  param_1[0xe] = dVar1;
  if (*(char *)(param_1 + 0x16) == '\0') {
    dVar9 = param_1[0x18];
    if (dVar9 != param_1[0xb]) {
      free((void *)param_1[10]);
      if (dVar9 == 0.0) {
        param_1[10] = 0.0;
      }
      else {
        pvVar4 = FUN_1800058c0((ulonglong)dVar9);
        param_1[10] = (double)pvVar4;
      }
    }
    param_1[0xb] = dVar9;
  }
  dVar9 = param_1[0x18];
  if (dVar9 != param_1[9]) {
    free((void *)param_1[8]);
    if (dVar9 == 0.0) {
      param_1[8] = 0.0;
    }
    else {
      pvVar4 = FUN_1800058c0((ulonglong)dVar9);
      param_1[8] = (double)pvVar4;
    }
  }
  param_1[9] = dVar9;
  dVar9 = param_1[0x18];
  param_1[0x11] = 0.0;
  param_1[0x12] = 0.0;
  if ((((((longlong)dVar9 < 1) || ((longlong)param_1[0x19] < (longlong)dVar9)) ||
       (param_1[2] <= 0.0 && param_1[2] != 0.0)) ||
      ((param_1[3] <= 0.0 && param_1[3] != 0.0 || (param_1[4] <= 0.0 && param_1[4] != 0.0)))) ||
     (((longlong)param_1[1] < 1 || (*param_1 <= 0.0)))) {
LAB_180005c75:
    uVar6 = 0;
  }
  else {
    if ((*(char *)(param_1 + 0x16) != '\0') && (0 < (longlong)dVar9)) {
      pdVar5 = (double *)param_1[10];
      do {
        if (*pdVar5 <= 0.0) goto LAB_180005c75;
        lVar7 = lVar7 + 1;
        pdVar5 = pdVar5 + 1;
      } while (lVar7 < (longlong)dVar9);
    }
    param_1[0x11] = 4.94065645841247e-324;
    uVar6 = FUN_1800034f0((int *)param_1[0x17],param_2,(longlong *)(param_1 + 6));
    if ((int)uVar6 < 0) {
      uVar6 = 9;
    }
    else {
      dVar9 = FUN_180007580((longlong *)(param_1 + 6));
      param_1[0x22] = 0.0;
      param_1[0x13] = 4.94065645841247e-324;
      uVar6 = 0xfffffffe;
      param_1[0x14] = dVar9;
    }
  }
  return uVar6;
}

