// ev_d2f6dc3bbd6e9396ad512be13b53df9cabc9b1ba680df8e9a6a712e86ccba27f
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Removing unreachable block (ram,0x000180006f83) */
/* WARNING: Removing unreachable block (ram,0x000180006faa) */
/* WARNING: Removing unreachable block (ram,0x000180006f91) */
/* WARNING: Removing unreachable block (ram,0x000180006f78) */
/* WARNING: Removing unreachable block (ram,0x000180006f63) */

int FUN_180006e60(int *param_1,longlong *param_2,longlong *param_3)

{
  double **ppdVar1;
  longlong lVar2;
  double *pdVar3;
  ulonglong uVar4;
  double dVar5;
  double dVar6;
  ulonglong *puVar7;
  ulonglong uVar8;
  ulonglong uVar9;
  ulonglong uVar10;
  double **ppdVar11;
  double *pdVar12;
  int iVar13;
  double *pdVar14;
  uint uVar15;
  undefined1 auVar16 [16];
  undefined1 auVar17 [16];
  double *local_res20;
  void *local_158;
  ulonglong uStack_150;
  void *local_148;
  ulonglong uStack_140;
  double *local_138;
  ulonglong local_130;
  ulonglong local_128;
  ulonglong local_120;
  longlong local_118;
  ulonglong local_110;
  ulonglong local_108;
  longlong *local_f8;
  ulonglong local_f0;
  undefined1 local_e8 [8];
  undefined8 local_e0;
  void **ppvStack_d8;
  void **local_d0;
  undefined8 uStack_c8;
  double local_c0;
  undefined8 local_b8;
  undefined8 local_b0;
  void **ppvStack_a8;
  void **local_a0;
  undefined8 uStack_98;
  
  local_b8 = 0xfffffffffffffffe;
  uVar4 = param_2[1];
  local_res20 = (double *)0x3cb0000000000000;
  ppdVar1 = (double **)(param_1 + 0xc);
  ppdVar11 = &local_res20;
  if (2.220446049250313e-16 < (double)*ppdVar1 || (double)*ppdVar1 == 2.220446049250313e-16) {
    ppdVar11 = ppdVar1;
  }
  dVar5 = sqrt((double)*ppdVar11);
  local_148 = (void *)0x0;
  uStack_140 = 0;
  local_158 = (void *)0x0;
  uStack_150 = 0;
  uVar10 = 0;
  if (uVar4 == 0) {
    local_res20 = (double *)0x0;
  }
  else {
    if (0x1fffffffffffffff < uVar4) {
                    /* WARNING: Subroutine does not return */
      FUN_1800027a0();
    }
    local_res20 = malloc(uVar4 * 8);
    if ((local_res20 == (double *)0x0) && (uVar4 * 8 != 0)) {
                    /* WARNING: Subroutine does not return */
      FUN_1800027a0();
    }
  }
  pdVar12 = local_res20;
  lVar2 = ((longlong)uVar4 / 2) * 2;
  uVar8 = uVar10;
  if (0 < lVar2) {
    do {
      pdVar14 = (double *)(*param_2 + uVar8 * 8);
      dVar6 = pdVar14[1];
      local_res20[uVar8] = *pdVar14;
      (local_res20 + uVar8)[1] = dVar6;
      uVar8 = uVar8 + 2;
    } while ((longlong)uVar8 < lVar2);
  }
  local_138 = local_res20;
  local_130 = uVar4;
  FUN_18000c9d0(param_2,(longlong *)&local_138,lVar2,uVar4);
  uVar8 = (ulonglong)param_1[0xb];
  if (uVar8 != uStack_140) {
    free(local_148);
    if (uVar8 == 0) {
      local_148 = (void *)0x0;
    }
    else {
      local_148 = FUN_1800058c0(uVar8);
    }
  }
  uVar9 = (ulonglong)param_1[0xb];
  uStack_140 = uVar8;
  if (uVar9 != uStack_150) {
    free(local_158);
    if (uVar9 == 0) {
      local_158 = (void *)0x0;
    }
    else {
      local_158 = FUN_1800058c0(uVar9);
    }
  }
  uStack_150 = uVar9;
  FUN_1800034f0(param_1,&local_138,(longlong *)&local_148);
  iVar13 = 1;
  pdVar14 = pdVar12;
  if (0 < (longlong)uVar4) {
    local_118 = -(longlong)pdVar12;
    iVar13 = 1;
    uVar8 = uVar10;
    do {
      dVar6 = ABS(*pdVar12) * dVar5;
      if (dVar6 == 0.0) {
        dVar6 = dVar5;
      }
      *pdVar12 = *pdVar12 + dVar6;
      FUN_1800034f0(param_1,&local_138,(longlong *)&local_158);
      iVar13 = iVar13 + 1;
      *pdVar12 = *(double *)((longlong)pdVar12 + *param_2 + local_118);
      ppvStack_d8 = &local_158;
      local_d0 = &local_148;
      local_e0 = local_b0;
      uStack_c8 = uStack_98;
      uVar9 = param_3[1];
      local_110 = *param_3 + uVar9 * uVar10 * 8;
      local_128 = uVar9;
      uVar10 = uVar9;
      if ((local_110 & 7) == 0) {
        local_120 = (ulonglong)(-(int)(local_110 >> 3) & 1);
        puVar7 = &local_128;
        if ((longlong)local_120 <= (longlong)uVar9) {
          puVar7 = &local_120;
        }
        uVar10 = *puVar7;
      }
      lVar2 = uVar10 + ((longlong)(uVar9 - uVar10) / 2) * 2;
      local_108 = uVar9;
      local_f8 = param_3;
      local_f0 = uVar9;
      local_c0 = dVar6;
      ppvStack_a8 = ppvStack_d8;
      local_a0 = local_d0;
      FUN_18000af20((longlong)local_e8,(longlong *)&local_110,0,uVar10);
      for (; (longlong)uVar10 < lVar2; uVar10 = uVar10 + 2) {
        pdVar14 = (double *)((longlong)*local_d0 + uVar10 * 8);
        pdVar3 = (double *)((longlong)*ppvStack_d8 + uVar10 * 8);
        auVar16._0_8_ = *pdVar3 - *pdVar14;
        auVar16._8_8_ = pdVar3[1] - pdVar14[1];
        auVar17._8_8_ = local_c0;
        auVar17._0_8_ = local_c0;
        auVar17 = divpd(auVar16,auVar17);
        *(undefined1 (*) [16])(local_110 + uVar10 * 8) = auVar17;
      }
      FUN_18000af20((longlong)local_e8,(longlong *)&local_110,lVar2,uVar9);
      uVar15 = (int)uVar8 + 1;
      uVar8 = (ulonglong)uVar15;
      pdVar12 = pdVar12 + 1;
      uVar10 = (ulonglong)(int)uVar15;
      pdVar14 = local_res20;
    } while ((longlong)uVar10 < (longlong)uVar4);
  }
  free(pdVar14);
  free(local_158);
  free(local_148);
  return iVar13;
}

