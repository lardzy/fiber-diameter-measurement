// ev_d9f8b641eb53e6b0ccd6f6537c7a24071f782defb0b019537c3cd00de0bc46c1
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */
/* WARNING: Removing unreachable block (ram,0x000180006c17) */
/* WARNING: Removing unreachable block (ram,0x000180006c29) */
/* WARNING: Removing unreachable block (ram,0x000180006c35) */
/* WARNING: Removing unreachable block (ram,0x000180006c41) */
/* WARNING: Removing unreachable block (ram,0x000180006c4d) */
/* WARNING: Removing unreachable block (ram,0x0001800063f5) */
/* WARNING: Removing unreachable block (ram,0x000180006410) */
/* WARNING: Removing unreachable block (ram,0x000180006403) */
/* WARNING: Removing unreachable block (ram,0x000180006c47) */
/* WARNING: Removing unreachable block (ram,0x000180006c3b) */
/* WARNING: Removing unreachable block (ram,0x000180006c2f) */
/* WARNING: Removing unreachable block (ram,0x000180006c23) */
/* WARNING: Removing unreachable block (ram,0x000180006c0b) */
/* WARNING: Removing unreachable block (ram,0x0001800062d2) */
/* WARNING: Removing unreachable block (ram,0x000180005fb8) */
/* WARNING: Removing unreachable block (ram,0x000180005d94) */
/* WARNING: Removing unreachable block (ram,0x000180005fa3) */
/* WARNING: Removing unreachable block (ram,0x0001800060b8) */
/* WARNING: Removing unreachable block (ram,0x0001800062e7) */
/* WARNING: Removing unreachable block (ram,0x000180006c11) */
/* WARNING: Removing unreachable block (ram,0x000180005d7f) */
/* WARNING: Removing unreachable block (ram,0x000180006c1d) */

undefined8 FUN_180005ca0(double *param_1,void **param_2)

{
  void **ppvVar1;
  double **ppdVar2;
  undefined8 *puVar3;
  undefined8 *puVar4;
  undefined4 *puVar5;
  undefined4 *puVar6;
  double *pdVar7;
  double *pdVar8;
  double dVar9;
  void **ppvVar10;
  undefined1 auVar11 [16];
  undefined1 auVar12 [16];
  double dVar13;
  uint uVar14;
  undefined4 uVar15;
  uint uVar16;
  int iVar17;
  undefined8 uVar18;
  void *pvVar19;
  void *pvVar20;
  double *pdVar21;
  longlong *plVar22;
  double **ppdVar23;
  longlong lVar24;
  longlong lVar25;
  longlong lVar26;
  double *pdVar27;
  double dVar28;
  double dVar29;
  double dVar30;
  undefined1 auVar31 [16];
  double dVar32;
  double dVar33;
  undefined1 auStackY_298 [32];
  undefined8 local_268;
  void **ppvStack_260;
  void **local_258;
  double *pdStack_250;
  double local_248;
  void **local_240;
  undefined1 *puStack_238;
  undefined1 local_230;
  undefined7 uStack_22f;
  double *pdStack_228;
  void *local_220;
  void **local_218;
  undefined8 local_210;
  void *local_208 [2];
  void **local_1f8;
  void **local_1f0;
  void **local_1e0;
  void *pvStack_1d8;
  undefined8 local_1d0;
  double *pdStack_1c8;
  void *local_1c0;
  void *local_1b8;
  undefined1 local_1af [7];
  double local_1a8;
  void **ppvStack_1a0;
  void **local_198;
  double *pdStack_190;
  double local_188;
  double local_180;
  void *local_178;
  longlong local_170;
  longlong local_168;
  undefined1 local_160 [16];
  void *local_150 [11];
  undefined8 local_f8;
  double *local_e8;
  ulonglong local_e0;
  
  local_210 = 0xfffffffffffffffe;
  local_e0 = DAT_18001a000 ^ (ulonglong)auStackY_298;
  pdVar27 = param_1 + 0xc;
  local_218 = param_2;
  iVar17 = FUN_180006e60((int *)param_1[0x17],(longlong *)param_2,(longlong *)pdVar27);
  if (iVar17 < 0) {
    uVar18 = 9;
  }
  else {
    if (iVar17 < 1) {
      param_1[0x12] = (double)((longlong)param_1[0x12] + 1);
    }
    else {
      param_1[0x11] = (double)((longlong)param_1[0x11] + (longlong)iVar17);
    }
    ppvVar1 = (void **)(param_1 + 0x1c);
    dVar29 = param_1[0xe];
    if (dVar29 != param_1[0x1d]) {
      free(*ppvVar1);
      if (dVar29 == 0.0) {
        *ppvVar1 = (void *)0x0;
      }
      else {
        pvVar19 = FUN_1800058c0((ulonglong)dVar29);
        *ppvVar1 = pvVar19;
      }
    }
    param_1[0x1d] = dVar29;
    lVar26 = 0;
    if (0 < (longlong)dVar29) {
      do {
        puStack_238 = (undefined1 *)param_1[0xd];
        local_240 = (void **)((longlong)*pdVar27 + (longlong)puStack_238 * lVar26 * 8);
        pdStack_228 = pdVar27;
        local_220 = puStack_238;
        dVar28 = FUN_180013690((longlong *)&local_240);
        *(double *)((longlong)*ppvVar1 + lVar26 * 8) = dVar28;
        lVar26 = lVar26 + 1;
      } while (lVar26 < (longlong)dVar29);
    }
    FUN_1800072a0((longlong *)&local_178,(longlong *)pdVar27);
    if ((local_170 != 0) && (local_168 != 0)) {
      auVar31._8_8_ = 0;
      auVar31._0_8_ = SUB168(SEXT816(0x7fffffffffffffff),8);
      if (SUB168((auVar31 << 0x40 | ZEXT816(0x7fffffffffffffff)) / SEXT816(local_168),0) < local_170
         ) {
                    /* WARNING: Subroutine does not return */
        FUN_1800027a0();
      }
      if (local_168 != 0) {
        auVar11._8_8_ = 0;
        auVar11._0_8_ = SUB168(SEXT816(0x7fffffffffffffff),8);
        if (SUB168((auVar11 << 0x40 | ZEXT816(0x7fffffffffffffff)) / SEXT816(local_168),0) <
            local_170) {
                    /* WARNING: Subroutine does not return */
          FUN_1800027a0();
        }
        if ((local_168 != 0) &&
           (auVar12._8_8_ = 0, auVar12._0_8_ = SUB168(SEXT816(0x7fffffffffffffff),8),
           SUB168((auVar12 << 0x40 | ZEXT816(0x7fffffffffffffff)) / SEXT816(local_168),0) <
           local_170)) {
                    /* WARNING: Subroutine does not return */
          FUN_1800027a0();
        }
      }
    }
    FUN_180007500(pdVar27,local_170 * local_168,local_170,local_168);
    dVar29 = param_1[0xe];
    dVar28 = param_1[0xd];
    lVar26 = (((longlong)dVar29 * (longlong)dVar28) / 2) * 2;
    lVar24 = 0;
    if (0 < lVar26) {
      do {
        puVar3 = (undefined8 *)((longlong)local_178 + lVar24 * 8);
        uVar18 = puVar3[1];
        puVar4 = (undefined8 *)((longlong)*pdVar27 + lVar24 * 8);
        *puVar4 = *puVar3;
        puVar4[1] = uVar18;
        lVar24 = lVar24 + 2;
      } while (lVar24 < lVar26);
    }
    FUN_18000c9d0((longlong *)&local_178,(longlong *)pdVar27,lVar26,
                  (longlong)dVar29 * (longlong)dVar28);
    FUN_180006d80((longlong *)(param_1 + 0xf),(longlong *)local_150);
    if (param_1[0x13] == 4.94065645841247e-324) {
      if ((*(char *)(param_1 + 0x16) == '\0') && (lVar26 = 0, 0 < (longlong)param_1[0x18])) {
        do {
          dVar29 = *(double *)((longlong)*ppvVar1 + lVar26 * 8);
          if (dVar29 == 0.0) {
            dVar29 = 1.0;
          }
          *(double *)((longlong)param_1[10] + lVar26 * 8) = dVar29;
          lVar26 = lVar26 + 1;
        } while (lVar26 < (longlong)param_1[0x18]);
      }
      ppvStack_260 = (void **)(param_1 + 10);
      local_258 = param_2;
      dVar29 = FUN_1800080a0((longlong)&local_268 + 1);
      param_1[0x2a] = dVar29;
      dVar29 = *param_1 * dVar29;
      param_1[0x27] = dVar29;
      if (dVar29 == 0.0) {
        param_1[0x27] = *param_1;
      }
    }
    pdVar27 = param_1 + 0x20;
    dVar29 = param_1[7];
    if (dVar29 != param_1[0x21]) {
      free((void *)*pdVar27);
      if (dVar29 == 0.0) {
        *pdVar27 = 0.0;
      }
      else {
        pvVar19 = FUN_1800058c0((ulonglong)dVar29);
        *pdVar27 = (double)pvVar19;
      }
    }
    param_1[0x21] = dVar29;
    lVar26 = ((longlong)dVar29 / 2) * 2;
    lVar24 = 0;
    if (0 < lVar26) {
      do {
        puVar3 = (undefined8 *)((longlong)param_1[6] + lVar24 * 8);
        uVar18 = puVar3[1];
        puVar4 = (undefined8 *)((longlong)*pdVar27 + lVar24 * 8);
        *puVar4 = *puVar3;
        puVar4[1] = uVar18;
        lVar24 = lVar24 + 2;
      } while (lVar24 < lVar26);
    }
    FUN_18000c9d0((longlong *)(param_1 + 6),(longlong *)pdVar27,lVar26,(longlong)dVar29);
    local_240 = &local_178;
    puStack_238 = local_160;
    local_230 = 1;
    pdStack_228 = (double *)local_f8;
    local_220 = (void *)0x0;
    FUN_18000ae20((longlong *)&local_240,(longlong *)pdVar27,(double *)&local_e8);
    pvVar19 = (void *)param_1[0x18];
    ppvVar10 = (void **)*pdVar27;
    local_220 = (void *)param_1[0x21];
    pdVar7 = param_1 + 8;
    local_240 = ppvVar10;
    puStack_238 = pvVar19;
    pdStack_228 = pdVar27;
    if (pvVar19 != (void *)param_1[9]) {
      free((void *)*pdVar7);
      if (pvVar19 == (void *)0x0) {
        *pdVar7 = 0.0;
      }
      else {
        pvVar20 = FUN_1800058c0((ulonglong)pvVar19);
        *pdVar7 = (double)pvVar20;
      }
    }
    param_1[9] = (double)pvVar19;
    lVar26 = ((longlong)pvVar19 / 2) * 2;
    lVar24 = 0;
    if (0 < lVar26) {
      do {
        pvVar20 = (ppvVar10 + lVar24)[1];
        puVar3 = (undefined8 *)((longlong)*pdVar7 + lVar24 * 8);
        *puVar3 = ppvVar10[lVar24];
        puVar3[1] = pvVar20;
        lVar24 = lVar24 + 2;
      } while (lVar24 < lVar26);
    }
    FUN_18000c9d0((longlong *)&local_240,(longlong *)pdVar7,lVar26,(longlong)pvVar19);
    ppdVar2 = (double **)(param_1 + 0x15);
    *ppdVar2 = (double *)0x0;
    if ((param_1[0x14] != 0.0) && (lVar26 = 0, 0 < (longlong)param_1[0x18])) {
      dVar29 = param_1[0xf];
      pvVar19 = *ppvVar1;
      pdVar27 = param_1 + 0xc;
      do {
        if (*(double *)((longlong)pvVar19 + (longlong)*(int *)((longlong)dVar29 + lVar26 * 4) * 8)
            != 0.0) {
          local_268 = *pdVar7;
          ppvStack_260 = (void **)(lVar26 + 1);
          local_248 = param_1[9];
          local_180 = param_1[0x14];
          local_198 = local_258;
          puStack_238 = (undefined1 *)param_1[0xd];
          local_240 = (void **)((longlong)*pdVar27 + (longlong)puStack_238 * lVar26 * 8);
          local_1d0 = CONCAT71(uStack_22f,local_230);
          pdStack_250 = pdVar7;
          pdStack_228 = pdVar27;
          local_220 = puStack_238;
          local_1f8 = local_240;
          local_1f0 = ppvStack_260;
          local_1e0 = local_240;
          pvStack_1d8 = puStack_238;
          pdStack_1c8 = pdVar27;
          local_1c0 = puStack_238;
          local_1b8 = puStack_238;
          local_1a8 = local_268;
          ppvStack_1a0 = ppvStack_260;
          pdStack_190 = pdVar7;
          local_188 = local_248;
          dVar28 = FUN_1800092b0(&local_1f8,(longlong)local_1af);
          dVar29 = param_1[0xf];
          pvVar19 = *ppvVar1;
          local_e8 = (double *)
                     ABS(dVar28 / *(double *)
                                   ((longlong)pvVar19 +
                                   (longlong)*(int *)((longlong)dVar29 + lVar26 * 4) * 8));
          ppdVar23 = &local_e8;
          if ((double)local_e8 < (double)*ppdVar2 || (double)local_e8 == (double)*ppdVar2) {
            ppdVar23 = ppdVar2;
          }
          *ppdVar2 = *ppdVar23;
        }
        lVar26 = lVar26 + 1;
        param_2 = local_218;
      } while (lVar26 < (longlong)param_1[0x18]);
    }
    if (param_1[4] < (double)*ppdVar2) {
      if (*(char *)(param_1 + 0x16) == '\0') {
        ppvVar10 = (void **)(param_1 + 10);
        dVar29 = param_1[0x1d];
        ppvStack_260 = ppvVar10;
        local_258 = ppvVar1;
        if (dVar29 != param_1[0xb]) {
          free(*ppvVar10);
          if (dVar29 == 0.0) {
            *ppvVar10 = (void *)0x0;
          }
          else {
            pvVar19 = FUN_1800058c0((ulonglong)dVar29);
            *ppvVar10 = pvVar19;
          }
        }
        param_1[0xb] = dVar29;
        lVar26 = ((longlong)dVar29 / 2) * 2;
        lVar24 = 0;
        if (0 < lVar26) {
          do {
            auVar31 = maxpd(*(undefined1 (*) [16])((longlong)*ppvVar10 + lVar24 * 8),
                            *(undefined1 (*) [16])((longlong)*ppvVar1 + lVar24 * 8));
            *(undefined1 (*) [16])((longlong)*ppvVar10 + lVar24 * 8) = auVar31;
            lVar24 = lVar24 + 2;
          } while (lVar24 < lVar26);
        }
        FUN_1800118f0((longlong)&local_268,(longlong *)ppvVar10,lVar26,(longlong)dVar29);
      }
      ppvVar10 = (void **)(param_1 + 0x1a);
      do {
        pdVar27 = param_1 + 0x27;
        FUN_1800081d0(&local_178,(void **)(param_1 + 10),(void **)(param_1 + 8),(void **)*pdVar27,
                      (void ***)(param_1 + 0x22),ppvVar10);
        dVar29 = param_1[0x1b];
        param_1[0x1b] = dVar29;
        lVar26 = ((longlong)dVar29 / 2) * 2;
        lVar24 = 0;
        if (0 < lVar26) {
          do {
            puVar5 = (undefined4 *)((longlong)*ppvVar10 + lVar24 * 8);
            uVar14 = puVar5[1];
            uVar15 = puVar5[2];
            uVar16 = puVar5[3];
            puVar6 = (undefined4 *)((longlong)*ppvVar10 + lVar24 * 8);
            *puVar6 = *puVar5;
            puVar6[1] = uVar14 ^ 0x80000000;
            puVar6[2] = uVar15;
            puVar6[3] = uVar16 ^ 0x80000000;
            lVar24 = lVar24 + 2;
          } while (lVar24 < lVar26);
        }
        ppvStack_260 = ppvVar10;
        FUN_180011b70((longlong)&local_268,(longlong *)ppvVar10,lVar26,(longlong)dVar29);
        dVar29 = param_1[0x1b];
        ppvStack_260 = param_2;
        local_258 = ppvVar10;
        if (dVar29 != param_1[0x1d]) {
          free(*ppvVar1);
          if (dVar29 == 0.0) {
            *ppvVar1 = (void *)0x0;
          }
          else {
            pvVar19 = FUN_1800058c0((ulonglong)dVar29);
            *ppvVar1 = pvVar19;
          }
        }
        param_1[0x1d] = dVar29;
        lVar26 = ((longlong)dVar29 / 2) * 2;
        lVar24 = 0;
        if (0 < lVar26) {
          do {
            pdVar7 = (double *)((longlong)*param_2 + lVar24 * 8);
            dVar9 = pdVar7[1];
            pdVar21 = (double *)((longlong)*ppvVar10 + lVar24 * 8);
            dVar28 = pdVar21[1];
            pdVar8 = (double *)((longlong)*ppvVar1 + lVar24 * 8);
            *pdVar8 = *pdVar7 + *pdVar21;
            pdVar8[1] = dVar9 + dVar28;
            lVar24 = lVar24 + 2;
          } while (lVar24 < lVar26);
        }
        FUN_180011d40((longlong)&local_268,(longlong *)ppvVar1,lVar26,(longlong)dVar29);
        pdVar7 = param_1 + 0x29;
        ppvStack_260 = (void **)(param_1 + 10);
        local_258 = ppvVar10;
        dVar29 = FUN_1800080a0((longlong)&local_268 + 1);
        *pdVar7 = dVar29;
        if (param_1[0x13] == 4.94065645841247e-324) {
          pdVar21 = pdVar27;
          if (*pdVar7 <= *pdVar27 && *pdVar27 != *pdVar7) {
            pdVar21 = pdVar7;
          }
          *pdVar27 = *pdVar21;
        }
        uVar18 = FUN_1800034f0((int *)param_1[0x17],ppvVar1,(longlong *)(param_1 + 0x20));
        if ((int)uVar18 < 0) {
          uVar18 = 9;
          goto LAB_180006c8e;
        }
        param_1[0x11] = (double)((longlong)param_1[0x11] + 1);
        dVar29 = FUN_180007580((longlong *)(param_1 + 0x20));
        param_1[0x2b] = dVar29;
        param_1[0x2c] = -1.0;
        if (dVar29 * 0.1 < param_1[0x14]) {
          dVar29 = dVar29 / param_1[0x14];
          param_1[0x2c] = 1.0 - dVar29 * dVar29;
        }
        ppvStack_260 = local_150;
        local_e8 = param_1 + 0xc;
        local_258 = ppvVar10;
        plVar22 = FUN_180009000(&local_e8,&local_240,(longlong)&local_268 + 1);
        pdVar27 = param_1 + 0x1e;
        plVar22 = FUN_18000afc0(plVar22,(longlong *)local_208);
        dVar29 = (double)plVar22[1];
        if (dVar29 != param_1[0x1f]) {
          free((void *)*pdVar27);
          if (dVar29 == 0.0) {
            *pdVar27 = 0.0;
          }
          else {
            pvVar19 = FUN_1800058c0((ulonglong)dVar29);
            *pdVar27 = (double)pvVar19;
          }
        }
        param_1[0x1f] = dVar29;
        lVar26 = ((longlong)dVar29 / 2) * 2;
        lVar24 = 0;
        if (0 < lVar26) {
          do {
            puVar3 = (undefined8 *)(*plVar22 + lVar24 * 8);
            uVar18 = puVar3[1];
            puVar4 = (undefined8 *)((longlong)*pdVar27 + lVar24 * 8);
            *puVar4 = *puVar3;
            puVar4[1] = uVar18;
            lVar24 = lVar24 + 2;
          } while (lVar24 < lVar26);
        }
        FUN_18000c9d0(plVar22,(longlong *)pdVar27,lVar26,(longlong)dVar29);
        free(local_208[0]);
        free(pdStack_228);
        free(puStack_238);
        dVar29 = param_1[0x14];
        dVar28 = FUN_180007580((longlong *)pdVar27);
        dVar13 = (dVar28 / dVar29) * (dVar28 / dVar29);
        param_1[0x25] = dVar13;
        dVar28 = param_1[0x22];
        dVar9 = *pdVar7;
        dVar30 = sqrt(dVar28);
        param_2 = local_218;
        dVar32 = (dVar30 * dVar9) / dVar29;
        dVar32 = dVar32 * dVar32;
        param_1[0x26] = dVar32;
        dVar30 = dVar32 * 2.0 + dVar13;
        param_1[0x2e] = dVar30;
        dVar33 = -(dVar32 + dVar13);
        param_1[0x2d] = dVar33;
        lVar26 = 0;
        param_1[0x28] = 0.0;
        if (dVar30 != 0.0) {
          param_1[0x28] = param_1[0x2c] / dVar30;
        }
        dVar30 = param_1[0x28];
        if (0.25 < dVar30) {
          if ((dVar28 == 0.0) || (0.75 <= dVar30)) {
            param_1[0x27] = dVar9 * 2.0;
            dVar28 = dVar28 * 0.5;
            goto LAB_180006896;
          }
        }
        else {
          if (0.0 <= param_1[0x2c]) {
            param_1[0x24] = 0.5;
          }
          else {
            param_1[0x24] = (-(dVar32 + dVar13) * 0.5) / (param_1[0x2c] * 0.5 + dVar33);
          }
          if ((dVar29 <= param_1[0x2b] * 0.1) || (param_1[0x24] <= 0.1 && param_1[0x24] != 0.1)) {
            param_1[0x24] = 0.1;
          }
          local_e8 = (double *)(dVar9 / 0.1);
          ppdVar2 = (double **)(param_1 + 0x27);
          ppdVar23 = &local_e8;
          if ((double)*ppdVar2 <= dVar9 / 0.1) {
            ppdVar23 = ppdVar2;
          }
          *ppdVar2 = (double *)(param_1[0x24] * (double)*ppdVar23);
          dVar28 = dVar28 / param_1[0x24];
LAB_180006896:
          param_1[0x22] = dVar28;
        }
        if (0.0001 <= dVar30) {
          pvVar19 = (void *)param_1[0x1d];
          if (pvVar19 != local_218[1]) {
            free(*local_218);
            if (pvVar19 == (void *)0x0) {
              *param_2 = (void *)0x0;
            }
            else {
              pvVar20 = FUN_1800058c0((ulonglong)pvVar19);
              *param_2 = pvVar20;
            }
          }
          param_2[1] = pvVar19;
          lVar24 = ((longlong)pvVar19 / 2) * 2;
          lVar25 = lVar26;
          if (0 < lVar24) {
            do {
              puVar3 = (undefined8 *)((longlong)*ppvVar1 + lVar25 * 8);
              uVar18 = puVar3[1];
              puVar4 = (undefined8 *)((longlong)*param_2 + lVar25 * 8);
              *puVar4 = *puVar3;
              puVar4[1] = uVar18;
              lVar25 = lVar25 + 2;
            } while (lVar25 < lVar24);
          }
          FUN_18000c9d0((longlong *)ppvVar1,(longlong *)param_2,lVar24,(longlong)pvVar19);
          ppvStack_260 = (void **)(param_1 + 10);
          local_258 = param_2;
          pvVar19 = param_2[1];
          if (pvVar19 != (void *)param_1[0x1d]) {
            free(*ppvVar1);
            if (pvVar19 == (void *)0x0) {
              *ppvVar1 = (void *)0x0;
            }
            else {
              pvVar20 = FUN_1800058c0((ulonglong)pvVar19);
              *ppvVar1 = pvVar20;
            }
          }
          param_1[0x1d] = (double)pvVar19;
          lVar24 = ((longlong)pvVar19 / 2) * 2;
          if (0 < lVar24) {
            do {
              pdVar27 = (double *)((longlong)param_1[10] + lVar26 * 8);
              dVar29 = pdVar27[1];
              pdVar7 = (double *)((longlong)*param_2 + lVar26 * 8);
              dVar28 = pdVar7[1];
              pdVar21 = (double *)((longlong)*ppvVar1 + lVar26 * 8);
              *pdVar21 = *pdVar27 * *pdVar7;
              pdVar21[1] = dVar29 * dVar28;
              lVar26 = lVar26 + 2;
            } while (lVar26 < lVar24);
          }
          FUN_180011f90((longlong)&local_268,(longlong *)ppvVar1,lVar24,(longlong)pvVar19);
          dVar29 = param_1[0x21];
          if (dVar29 != param_1[7]) {
            free((void *)param_1[6]);
            if (dVar29 == 0.0) {
              param_1[6] = 0.0;
            }
            else {
              pvVar19 = FUN_1800058c0((ulonglong)dVar29);
              param_1[6] = (double)pvVar19;
            }
          }
          param_1[7] = dVar29;
          lVar26 = ((longlong)dVar29 / 2) * 2;
          lVar24 = 0;
          if (0 < lVar26) {
            do {
              puVar3 = (undefined8 *)((longlong)param_1[0x20] + lVar24 * 8);
              uVar18 = puVar3[1];
              puVar4 = (undefined8 *)((longlong)param_1[6] + lVar24 * 8);
              *puVar4 = *puVar3;
              puVar4[1] = uVar18;
              lVar24 = lVar24 + 2;
            } while (lVar24 < lVar26);
          }
          FUN_18000c9d0((longlong *)(param_1 + 0x20),(longlong *)(param_1 + 6),lVar26,
                        (longlong)dVar29);
          dVar29 = FUN_180007580((longlong *)ppvVar1);
          param_1[0x2a] = dVar29;
          param_1[0x14] = param_1[0x2b];
          param_1[0x13] = (double)((longlong)param_1[0x13] + 1);
        }
        dVar29 = param_1[0x2c];
        dVar28 = param_1[2];
        if ((((ABS(dVar29) <= dVar28) && (param_1[0x2e] <= dVar28)) && (param_1[0x28] * 0.5 <= 1.0))
           && (param_1[0x27] <= param_1[0x2a] * param_1[3])) {
          uVar18 = 3;
          goto LAB_180006c8e;
        }
        if (((ABS(dVar29) <= dVar28) && (param_1[0x2e] <= dVar28)) && (param_1[0x28] * 0.5 <= 1.0))
        {
          uVar18 = 1;
          goto LAB_180006c8e;
        }
        if (param_1[0x27] <= param_1[0x2a] * param_1[3]) {
          uVar18 = 2;
          goto LAB_180006c8e;
        }
        if ((longlong)param_1[1] <= (longlong)param_1[0x11]) {
          uVar18 = 5;
          goto LAB_180006c8e;
        }
        if (((ABS(dVar29) <= 2.220446049250313e-16) && (param_1[0x2e] <= 2.220446049250313e-16)) &&
           (param_1[0x28] * 0.5 <= 1.0)) {
          uVar18 = 6;
          goto LAB_180006c8e;
        }
        if (param_1[0x27] <= param_1[0x2a] * 2.220446049250313e-16) {
          uVar18 = 7;
          goto LAB_180006c8e;
        }
        if (param_1[0x15] <= 2.220446049250313e-16) {
          uVar18 = 8;
          goto LAB_180006c8e;
        }
      } while (param_1[0x28] <= 0.0001 && param_1[0x28] != 0.0001);
      uVar18 = 0xffffffff;
    }
    else {
      uVar18 = 4;
    }
LAB_180006c8e:
    FUN_180006d00(&local_178);
  }
  return uVar18;
}

