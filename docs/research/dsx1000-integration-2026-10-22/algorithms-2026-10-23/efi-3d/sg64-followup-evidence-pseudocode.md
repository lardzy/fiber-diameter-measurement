## procedure_pseudo_code {'procedure': '0x180006bc0'}
Evidence: `ev_01ee25be30e71ef85c344be6ea1c1e91d71bb9e4b85b8c84a9b56591a2ec790c`
```c

void FUN_180006bc0(int *param_1)

{
  double dVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  char *pcVar4;
  int *piVar5;
  int iVar6;
  ulonglong uVar7;
  int iVar8;
  uint uVar9;
  int iVar11;
  int iVar12;
  int iVar13;
  int *piVar14;
  int iVar15;
  int iVar16;
  int iVar17;
  int iVar18;
  int iVar19;
  int iVar20;
  int iVar21;
  longlong lVar22;
  double dVar23;
  double dVar24;
  double dVar25;
  double dVar26;
  double dVar27;
  undefined8 local_a0;
  undefined8 local_98;
  ulonglong uVar10;
  
  iVar13 = param_1[0xb];
  iVar19 = param_1[1];
  iVar18 = *param_1;
  uVar7 = 0;
  if (4 < param_1[10]) {
    local_a0 = 0;
    local_98 = 0;
    pcVar4 = cv::String::allocate((String *)&local_a0,0x22);
    builtin_strncpy(pcVar4,"param->nLapPyrLevel < MAX_PYR_BUFS",0x22);
    cv::error(-0xd7,(String *)&local_a0,"sg::ForEachResizeTable",
              "x:\\doc\\olympus-dsx-r\\02_srcandbin\\01_trunk\\olyextendfocusimagesg\\olyextendfocusimagesg\\SweepGuideCommon.h"
              ,0x39e);
    cv::String::deallocate((String *)&local_a0);
  }
  piVar14 = param_1 + 0x96;
  *piVar14 = iVar18;
  param_1[0x97] = iVar19;
  iVar15 = 1;
  piVar5 = piVar14;
  if (0 < param_1[10]) {
    do {
      piVar5[2] = (*piVar5 + 1) / 2;
      piVar5[3] = (piVar5[1] + 1) / 2;
      iVar15 = iVar15 + 1;
      piVar5 = piVar5 + 2;
    } while (iVar15 <= param_1[10]);
  }
  lVar22 = (longlong)(iVar13 + -1);
  iVar13 = param_1[lVar22 * 2 + 0x96];
  iVar15 = param_1[lVar22 * 2 + 0x97];
  dVar26 = (double)iVar13;
  iVar8 = (int)ROUND(*(double *)(param_1 + 0x24) * dVar26);
  dVar27 = (double)iVar15;
  iVar20 = (int)ROUND(*(double *)(param_1 + 0x24) * dVar27);
  iVar16 = (int)ROUND(dVar26 * *(double *)(param_1 + 0x2c));
  iVar11 = (int)ROUND(dVar27 * *(double *)(param_1 + 0x2c));
  dVar23 = 0.0;
  if (0 < param_1[10]) {
    piVar5 = param_1 + 0xb0;
    uVar10 = uVar7;
    do {
      dVar1 = 0.0;
      iVar6 = iVar13;
      iVar21 = iVar15;
      if ((longlong)uVar7 < lVar22) {
        iVar17 = *piVar14;
        iVar12 = piVar14[1];
        dVar1 = 0.0;
        dVar25 = 0.0;
        if (iVar13 * iVar15 == 0) {
          iVar6 = (int)ROUND((double)iVar17 * 0.0);
          iVar21 = (int)ROUND((double)iVar12 * 0.0);
        }
        else {
          dVar25 = dVar26 / (double)iVar17;
          dVar1 = dVar27 / (double)iVar12;
        }
        piVar5[-4] = 3;
LAB_180006e44:
        piVar5[-3] = iVar17;
        piVar5[-2] = iVar12;
        piVar5[-1] = iVar6;
        *piVar5 = iVar21;
        *(double *)(piVar5 + 4) = dVar1;
        *(double *)(piVar5 + 2) = dVar25;
      }
      else if (lVar22 < (longlong)uVar7) {
        iVar17 = *piVar14;
        iVar12 = piVar14[1];
        dVar25 = 0.0;
        if (iVar13 * iVar15 == 0) {
          iVar6 = (int)ROUND((double)iVar17 * 0.0);
          iVar21 = (int)ROUND((double)iVar12 * 0.0);
        }
        else {
          dVar25 = dVar26 / (double)iVar17;
          dVar1 = dVar27 / (double)iVar12;
        }
        piVar5[-4] = 1;
        goto LAB_180006e44;
      }
      uVar9 = (int)uVar10 + 1;
      uVar10 = (ulonglong)uVar9;
      uVar7 = uVar7 + 1;
      piVar14 = piVar14 + 2;
      piVar5 = piVar5 + 0x22;
    } while ((int)uVar9 < param_1[10]);
  }
  dVar25 = 0.0;
  dVar1 = 0.0;
  iVar6 = iVar13 * iVar15;
  dVar24 = (double)iVar18;
  if (iVar6 == 0) {
    iVar21 = (int)ROUND(dVar24 * 0.0);
    iVar17 = (int)ROUND((double)iVar19 * 0.0);
  }
  else {
    dVar1 = dVar26 / dVar24;
    dVar25 = dVar27 / (double)iVar19;
    iVar21 = iVar13;
    iVar17 = iVar15;
  }
  param_1[0x134] = 1;
  param_1[0x135] = iVar18;
  param_1[0x136] = iVar19;
  param_1[0x137] = iVar21;
  param_1[0x138] = iVar17;
  *(double *)(param_1 + 0x13a) = dVar1;
  *(double *)(param_1 + 0x13c) = dVar25;
  dVar1 = 0.0;
  dVar25 = 0.0;
  if (iVar18 * iVar19 == 0) {
    iVar18 = (int)ROUND(dVar26 * 0.0);
    iVar19 = (int)ROUND(dVar27 * 0.0);
  }
  else {
    dVar25 = dVar24 / dVar26;
    dVar1 = (double)iVar19 / dVar27;
  }
  param_1[0x156] = 1;
  param_1[0x157] = iVar13;
  param_1[0x158] = iVar15;
  param_1[0x159] = iVar18;
  param_1[0x15a] = iVar19;
  *(double *)(param_1 + 0x15c) = dVar25;
  *(double *)(param_1 + 0x15e) = dVar1;
  dVar1 = *(double *)(param_1 + 0x24);
  param_1[0x178] = 3;
  param_1[0x179] = iVar13;
  param_1[0x17a] = iVar15;
  param_1[0x17b] = (int)ROUND(dVar1 * dVar26);
  param_1[0x17c] = (int)ROUND(dVar27 * dVar1);
  auVar2._8_4_ = SUB84(dVar1,0);
  auVar2._0_8_ = dVar1;
  auVar2._12_4_ = (int)((ulonglong)dVar1 >> 0x20);
  *(undefined1 (*) [16])(param_1 + 0x17e) = auVar2;
  dVar1 = 0.0;
  dVar25 = 0.0;
  if (iVar6 == 0) {
    iVar19 = (int)ROUND((double)iVar8 * 0.0);
    iVar18 = (int)ROUND((double)iVar20 * 0.0);
  }
  else {
    dVar25 = dVar26 / (double)iVar8;
    dVar1 = dVar27 / (double)iVar20;
    iVar19 = iVar13;
    iVar18 = iVar15;
  }
  param_1[0x19a] = 1;
  param_1[0x19b] = iVar8;
  param_1[0x19c] = iVar20;
  param_1[0x19d] = iVar19;
  param_1[0x19e] = iVar18;
  *(double *)(param_1 + 0x1a0) = dVar25;
  *(double *)(param_1 + 0x1a2) = dVar1;
  dVar1 = *(double *)(param_1 + 0x2c);
  param_1[0x1bc] = 1;
  param_1[0x1bd] = iVar13;
  param_1[0x1be] = iVar15;
  param_1[0x1bf] = (int)ROUND(dVar1 * dVar26);
  param_1[0x1c0] = (int)ROUND(dVar1 * dVar27);
  auVar3._8_4_ = SUB84(dVar1,0);
  auVar3._0_8_ = dVar1;
  auVar3._12_4_ = (int)((ulonglong)dVar1 >> 0x20);
  *(undefined1 (*) [16])(param_1 + 0x1c2) = auVar3;
  dVar1 = 0.0;
  if (iVar6 == 0) {
    iVar13 = (int)ROUND((double)iVar16 * 0.0);
    iVar15 = (int)ROUND((double)iVar11 * 0.0);
  }
  else {
    dVar1 = dVar26 / (double)iVar16;
    dVar23 = dVar27 / (double)iVar11;
  }
  param_1[0x1de] = 1;
  param_1[0x1df] = iVar16;
  param_1[0x1e0] = iVar11;
  param_1[0x1e1] = iVar13;
  param_1[0x1e2] = iVar15;
  *(double *)(param_1 + 0x1e4) = dVar1;
  *(double *)(param_1 + 0x1e6) = dVar23;
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18001cf20'}
Evidence: `ev_04ae4a82164b51eac576384883ce4b293d8f6aa7c68f2149f6c58a0698e20f58`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8 *
FUN_18001cf20(undefined8 *param_1,undefined8 param_2,longlong *param_3,ulonglong param_4)

{
  longlong lVar1;
  float *pfVar2;
  int iVar3;
  undefined1 auStack_78 [36];
  undefined4 uStack_54;
  undefined8 local_48;
  longlong *local_40;
  undefined8 local_38;
  ulonglong local_30;
  float *local_28;
  undefined8 local_20;
  ulonglong local_18;
  
  iVar3 = (int)param_4;
  local_48 = 0xfffffffffffffffe;
  local_18 = DAT_18008b080 ^ (ulonglong)auStack_78;
  if (0x20 < iVar3) {
                    /* WARNING: Subroutine does not return */
    FUN_180001b90(&DAT_18007be80,param_2,param_3,param_4);
  }
  local_20 = 0;
  local_38 = 0x20;
  pfVar2 = (float *)(*param_3 + param_3[1]);
  local_30 = (longlong)iVar3 * 4 + 0x1fU & 0xffffffffffffffe0;
  lVar1 = param_3[1];
  param_3[1] = local_30 + lVar1;
  local_40 = param_3;
  local_28 = pfVar2;
  if ((ulonglong)param_3[2] < local_30 + lVar1) {
                    /* WARNING: Subroutine does not return */
    FUN_180001b90("Allocating memory exceeding the limit ...",param_2,param_3,param_4);
  }
  local_30 = 0;
  FUN_180002440(pfVar2,param_4 & 0xffffffff,param_3,param_4);
  *param_1 = CONCAT44(uStack_54,iVar3);
  param_1[1] = pfVar2;
  return param_1;
}


```

## procedure_pseudo_code {'procedure': '0x18001f8f0'}
Evidence: `ev_053ea1fefb81aaed0910b37dba2e6bba7f2bae1526e8e79cab05cc2d0dbb18b2`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined4 FUN_18001f8f0(longlong param_1,int param_2,undefined8 *param_3)

{
  int *piVar1;
  int iVar2;
  uint uVar3;
  ulonglong uVar4;
  ulonglong uVar5;
  ulonglong uVar6;
  undefined4 uVar7;
  undefined1 auStackY_1e8 [32];
  Mat local_188 [4];
  int local_184;
  int local_180;
  void *local_178;
  undefined8 local_170;
  undefined8 local_168;
  undefined8 local_160;
  longlong local_150;
  longlong local_148;
  int *local_140;
  int local_138 [4];
  undefined4 local_128;
  Mat local_118 [4];
  int local_114;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  undefined8 local_f0;
  longlong local_e0;
  longlong local_d8;
  undefined1 *local_d0;
  undefined1 local_c8 [16];
  undefined4 local_b8;
  Mat local_a8 [4];
  int local_a4;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_80;
  longlong local_70;
  longlong local_68;
  undefined1 *local_60;
  undefined1 local_58 [16];
  undefined4 local_48;
  ulonglong local_38;
  
  local_38 = DAT_18008b080 ^ (ulonglong)auStackY_1e8;
  iVar2 = *(int *)(param_1 + 0xb24);
  if (iVar2 == 0) {
    return 0x16;
  }
  *(undefined8 *)(param_1 + 0x810) = *(undefined8 *)(param_1 + 0xb28);
  memset(local_a8,0,0x68);
  FUN_1800013f0(local_a8,(Mat *)(param_1 + 0xc08));
  local_48 = *(undefined4 *)(param_1 + 0xc68);
  memset(local_118,0,0x68);
  FUN_1800013f0(local_118,(Mat *)(param_1 + 0xc70));
  local_b8 = *(undefined4 *)(param_1 + 0xcd0);
  memset(local_188,0,0x68);
  FUN_1800013f0(local_188,(Mat *)(param_1 + 0xcd8));
  local_128 = *(undefined4 *)(param_1 + 0xd38);
  uVar5 = 0;
  uVar7 = 0;
  if (*(int *)(param_1 + 0x840) < *(int *)(param_1 + 0xb20) + param_2) {
    uVar7 = 0x11;
  }
  else {
    if (*(int *)(param_1 + 0xb20) == 0) {
      if (param_2 == *(int *)(param_1 + 0x840)) {
        (*(code *)(&PTR_FUN_18007bb30)
                  [(int)((uint)(*(int *)(param_1 + 0x1c) != 1) +
                        (uint)(*(int *)(param_1 + 0x20) != 8) * 2)])
                  (param_1 + 8,*(undefined1 *)(param_1 + 0x2d),*(undefined4 *)(param_1 + 8),
                   *(undefined4 *)(param_1 + 0xc));
        if (*(char *)(param_1 + 0x124) == '\x01') {
          FUN_180005740(param_1 + 8,local_188,2);
        }
        *(int *)(param_1 + 0xb20) = iVar2;
        goto LAB_18001faf8;
      }
      if (*(char *)(param_1 + 0x2d) != '\0') {
        memset(local_178,0,(longlong)(local_138[0] * local_180));
      }
    }
    FUN_180004260((undefined4 *)(param_1 + 8),(longlong *)(param_1 + 0x808),(int *)(param_1 + 0x840)
                  ,(ulonglong)*(uint *)(param_1 + 0xb20),*(uint *)(param_1 + 0xb20) + param_2,
                  param_3,local_a8,local_118,local_188);
    *(int *)(param_1 + 0xb20) = *(int *)(param_1 + 0xb20) + param_2;
  }
LAB_18001faf8:
  if (local_150 != 0) {
    LOCK();
    piVar1 = (int *)(local_150 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(local_188);
    }
  }
  local_150 = 0;
  local_178 = (void *)0x0;
  local_160 = 0;
  local_168 = 0;
  local_170 = 0;
  uVar4 = uVar5;
  uVar6 = uVar5;
  if (0 < local_184) {
    do {
      *(undefined4 *)(uVar6 + local_148) = 0;
      uVar3 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar3;
      uVar6 = uVar6 + 4;
    } while ((int)uVar3 < local_184);
  }
  if (local_140 != local_138) {
    cv::fastFree(local_140);
  }
  if (local_e0 != 0) {
    LOCK();
    piVar1 = (int *)(local_e0 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(local_118);
    }
  }
  local_e0 = 0;
  local_108 = 0;
  local_f0 = 0;
  local_f8 = 0;
  local_100 = 0;
  uVar4 = uVar5;
  uVar6 = uVar5;
  if (0 < local_114) {
    do {
      *(undefined4 *)(uVar6 + local_d8) = 0;
      uVar3 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar3;
      uVar6 = uVar6 + 4;
    } while ((int)uVar3 < local_114);
  }
  if (local_d0 != local_c8) {
    cv::fastFree(local_d0);
  }
  if (local_70 != 0) {
    LOCK();
    piVar1 = (int *)(local_70 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(local_a8);
    }
  }
  local_70 = 0;
  local_98 = 0;
  local_80 = 0;
  local_88 = 0;
  local_90 = 0;
  uVar4 = uVar5;
  if (0 < local_a4) {
    do {
      *(undefined4 *)(uVar4 + local_68) = 0;
      uVar3 = (int)uVar5 + 1;
      uVar5 = (ulonglong)uVar3;
      uVar4 = uVar4 + 4;
    } while ((int)uVar3 < local_a4);
  }
  if (local_60 != local_58) {
    cv::fastFree(local_60);
  }
  return uVar7;
}


```

## procedure_pseudo_code {'procedure': '0x180029a60'}
Evidence: `ev_065c984578d2ee5df1c98b9ea6f1fe04560509ae6e6da5912d0ab2fdb6073f00`
```c

void FUN_180029a60(int *param_1,longlong *param_2)

{
  double dVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  char *pcVar6;
  int iVar7;
  int iVar8;
  int *piVar9;
  int iVar10;
  ulonglong uVar11;
  int *piVar12;
  int iVar13;
  int iVar14;
  int iVar15;
  uint uVar16;
  int iVar17;
  longlong lVar19;
  int iVar20;
  double dVar21;
  double dVar22;
  double dVar23;
  double dVar24;
  double dVar25;
  double dVar26;
  undefined8 local_b0;
  undefined8 local_a8;
  ulonglong uVar18;
  
  iVar20 = param_1[0xb];
  iVar17 = param_1[1];
  iVar10 = *param_1;
  uVar11 = 0;
  if (4 < param_1[10]) {
    local_b0 = 0;
    local_a8 = 0;
    pcVar6 = cv::String::allocate((String *)&local_b0,0x22);
    builtin_strncpy(pcVar6,"param->nLapPyrLevel < MAX_PYR_BUFS",0x22);
    cv::error(-0xd7,(String *)&local_b0,"sg::ForEachResizeTable",
              "x:\\doc\\olympus-dsx-r\\02_srcandbin\\01_trunk\\olyextendfocusimagesg\\olyextendfocusimagesg\\SweepGuideCommon.h"
              ,0x39e);
    cv::String::deallocate((String *)&local_b0);
  }
  piVar12 = param_1 + 0x96;
  *piVar12 = iVar10;
  param_1[0x97] = iVar17;
  iVar14 = 1;
  piVar9 = piVar12;
  if (0 < param_1[10]) {
    do {
      piVar9[2] = (*piVar9 + 1) / 2;
      piVar9[3] = (piVar9[1] + 1) / 2;
      iVar14 = iVar14 + 1;
      piVar9 = piVar9 + 2;
    } while (iVar14 <= param_1[10]);
  }
  lVar19 = (longlong)(iVar20 + -1);
  iVar20 = param_1[lVar19 * 2 + 0x96];
  iVar14 = param_1[lVar19 * 2 + 0x97];
  dVar25 = (double)iVar20;
  iVar2 = (int)ROUND(*(double *)(param_1 + 0x24) * dVar25);
  dVar26 = (double)iVar14;
  iVar3 = (int)ROUND(*(double *)(param_1 + 0x24) * dVar26);
  iVar4 = (int)ROUND(dVar25 * *(double *)(param_1 + 0x2c));
  iVar5 = (int)ROUND(dVar26 * *(double *)(param_1 + 0x2c));
  dVar1 = 0.0;
  if (0 < param_1[10]) {
    piVar9 = param_1 + 0xb0;
    uVar18 = uVar11;
    do {
      iVar8 = iVar20;
      iVar15 = iVar14;
      if ((longlong)uVar11 < lVar19) {
        iVar7 = *piVar12;
        iVar13 = piVar12[1];
        dVar21 = 0.0;
        dVar22 = 0.0;
        if (iVar20 * iVar14 == 0) {
          iVar8 = (int)ROUND((double)iVar7 * 0.0);
          iVar15 = (int)ROUND((double)iVar13 * 0.0);
        }
        else {
          dVar22 = dVar25 / (double)iVar7;
          dVar21 = dVar26 / (double)iVar13;
        }
        piVar9[-4] = 3;
LAB_180029d05:
        piVar9[-3] = iVar7;
        piVar9[-2] = iVar13;
        piVar9[-1] = iVar8;
        *piVar9 = iVar15;
        *(double *)(piVar9 + 2) = dVar22;
        *(double *)(piVar9 + 4) = dVar21;
        FUN_18001e110(param_2,piVar9 + -4);
      }
      else if (lVar19 < (longlong)uVar11) {
        iVar7 = *piVar12;
        iVar13 = piVar12[1];
        dVar21 = 0.0;
        dVar22 = 0.0;
        if (iVar20 * iVar14 == 0) {
          iVar8 = (int)ROUND((double)iVar7 * 0.0);
          iVar15 = (int)ROUND((double)iVar13 * 0.0);
        }
        else {
          dVar22 = dVar25 / (double)iVar7;
          dVar21 = dVar26 / (double)iVar13;
        }
        piVar9[-4] = 1;
        goto LAB_180029d05;
      }
      uVar16 = (int)uVar18 + 1;
      uVar18 = (ulonglong)uVar16;
      uVar11 = uVar11 + 1;
      piVar12 = piVar12 + 2;
      piVar9 = piVar9 + 0x22;
    } while ((int)uVar16 < param_1[10]);
  }
  dVar21 = 0.0;
  dVar22 = 0.0;
  iVar8 = iVar20 * iVar14;
  dVar23 = (double)iVar10;
  dVar24 = (double)iVar17;
  if (iVar8 == 0) {
    iVar15 = (int)ROUND(dVar23 * 0.0);
    iVar7 = (int)ROUND(dVar24 * 0.0);
  }
  else {
    dVar22 = dVar25 / dVar23;
    dVar21 = dVar26 / dVar24;
    iVar15 = iVar20;
    iVar7 = iVar14;
  }
  param_1[0x134] = 1;
  param_1[0x135] = iVar10;
  param_1[0x136] = iVar17;
  param_1[0x137] = iVar15;
  param_1[0x138] = iVar7;
  *(double *)(param_1 + 0x13a) = dVar22;
  *(double *)(param_1 + 0x13c) = dVar21;
  FUN_18001e110(param_2,param_1 + 0x134);
  dVar21 = 0.0;
  dVar22 = 0.0;
  if (iVar10 * iVar17 == 0) {
    iVar10 = (int)ROUND(dVar25 * 0.0);
    iVar17 = (int)ROUND(dVar26 * 0.0);
  }
  else {
    dVar22 = dVar23 / dVar25;
    dVar21 = dVar24 / dVar26;
  }
  param_1[0x156] = 1;
  param_1[0x157] = iVar20;
  param_1[0x158] = iVar14;
  param_1[0x159] = iVar10;
  param_1[0x15a] = iVar17;
  *(double *)(param_1 + 0x15c) = dVar22;
  *(double *)(param_1 + 0x15e) = dVar21;
  FUN_18001e110(param_2,param_1 + 0x156);
  dVar21 = *(double *)(param_1 + 0x24);
  param_1[0x178] = 3;
  param_1[0x179] = iVar20;
  param_1[0x17a] = iVar14;
  param_1[0x17b] = (int)ROUND(dVar21 * dVar25);
  param_1[0x17c] = (int)ROUND(dVar26 * dVar21);
  *(double *)(param_1 + 0x17e) = dVar21;
  *(double *)(param_1 + 0x180) = dVar21;
  FUN_18001e110(param_2,param_1 + 0x178);
  dVar21 = 0.0;
  dVar22 = 0.0;
  if (iVar8 == 0) {
    iVar17 = (int)ROUND((double)iVar2 * 0.0);
    iVar10 = (int)ROUND((double)iVar3 * 0.0);
  }
  else {
    dVar22 = dVar25 / (double)iVar2;
    dVar21 = dVar26 / (double)iVar3;
    iVar17 = iVar20;
    iVar10 = iVar14;
  }
  param_1[0x19a] = 1;
  param_1[0x19b] = iVar2;
  param_1[0x19c] = iVar3;
  param_1[0x19d] = iVar17;
  param_1[0x19e] = iVar10;
  *(double *)(param_1 + 0x1a0) = dVar22;
  *(double *)(param_1 + 0x1a2) = dVar21;
  FUN_18001e110(param_2,param_1 + 0x19a);
  dVar21 = *(double *)(param_1 + 0x2c);
  param_1[0x1bc] = 1;
  param_1[0x1bd] = iVar20;
  param_1[0x1be] = iVar14;
  param_1[0x1bf] = (int)ROUND(dVar21 * dVar25);
  param_1[0x1c0] = (int)ROUND(dVar21 * dVar26);
  *(double *)(param_1 + 0x1c2) = dVar21;
  *(double *)(param_1 + 0x1c4) = dVar21;
  FUN_18001e110(param_2,param_1 + 0x1bc);
  dVar21 = 0.0;
  if (iVar8 == 0) {
    iVar20 = (int)ROUND((double)iVar4 * 0.0);
    iVar14 = (int)ROUND((double)iVar5 * 0.0);
  }
  else {
    dVar21 = dVar25 / (double)iVar4;
    dVar1 = dVar26 / (double)iVar5;
  }
  param_1[0x1de] = 1;
  param_1[0x1df] = iVar4;
  param_1[0x1e0] = iVar5;
  param_1[0x1e1] = iVar20;
  param_1[0x1e2] = iVar14;
  *(double *)(param_1 + 0x1e4) = dVar21;
  *(double *)(param_1 + 0x1e6) = dVar1;
  FUN_18001e110(param_2,param_1 + 0x1de);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180004260'}
Evidence: `ev_067f9786c9792a47e534bb638157d6714ba6b8b9c4224e4cb218fa59b4332142`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180004260(undefined4 *param_1,longlong *param_2,int *param_3,undefined8 param_4,int param_5
                  ,undefined8 *param_6,undefined8 param_7,undefined8 param_8,undefined8 param_9)

{
  ulonglong uVar1;
  undefined8 *puVar2;
  int iVar3;
  ulonglong uVar4;
  longlong lVar5;
  undefined1 auStack_c8 [32];
  undefined8 *local_a8;
  undefined4 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_78;
  longlong *local_70;
  undefined8 local_68;
  ulonglong local_60;
  undefined8 *local_58;
  undefined8 local_50;
  ulonglong local_48;
  
  local_78 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStack_c8;
  lVar5 = (longlong)(int)param_4;
  local_50 = 0;
  local_68 = 0x20;
  local_a8 = (undefined8 *)(*param_2 + param_2[1]);
  uVar4 = (longlong)*param_3 * 8 + 0x1fU & 0xffffffffffffffe0;
  uVar1 = param_2[1] + uVar4;
  param_2[1] = uVar1;
  local_70 = param_2;
  local_60 = uVar4;
  local_58 = local_a8;
  if ((ulonglong)param_2[2] < uVar1) {
                    /* WARNING: Subroutine does not return */
    FUN_180001b90("Allocating memory exceeding the limit ...",param_2,param_3,param_4);
  }
  iVar3 = 0;
  puVar2 = local_a8;
  if (0 < *param_3) {
    do {
      *puVar2 = 0;
      iVar3 = iVar3 + 1;
      puVar2 = puVar2 + 1;
    } while (iVar3 < *param_3);
  }
  for (; lVar5 < param_5; lVar5 = lVar5 + 1) {
    local_a8[*(ushort *)(*(longlong *)(param_3 + 8) + lVar5 * 2)] = *param_6;
    param_6 = param_6 + 1;
  }
  local_88 = param_9;
  local_90 = param_8;
  local_98 = param_7;
  local_a0 = param_1[2];
  (*(code *)(&PTR_FUN_18007ba90)[(int)((uint)(param_1[5] != 1) + (uint)(param_1[6] != 8) * 2)])
            (param_1,*(undefined1 *)((longlong)param_1 + 0x25),*param_1,param_1[1]);
  param_2[1] = param_2[1] - uVar4;
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180011c40'}
Evidence: `ev_0f6fcdd9bf6763fa1316d394bd32a97d0cba3267a43628d1399b2d894ae6fa9f`
```c

void FUN_180011c40(float *param_1,float *param_2)

{
  float *pfVar1;
  float fVar2;
  float fVar3;
  float fVar4;
  float *pfVar5;
  float *pfVar6;
  
  pfVar5 = param_1;
  if (param_1 != param_2) {
    while (pfVar6 = pfVar5 + 2, pfVar6 != param_2) {
      fVar2 = *pfVar6;
      fVar4 = pfVar5[3];
      pfVar1 = pfVar5 + 3;
      pfVar5 = pfVar6;
      if ((fVar2 < *param_1) || ((fVar2 <= *param_1 && ((int)fVar4 < (int)param_1[1])))) {
        while (param_1 != pfVar6) {
          pfVar1[-1] = pfVar1[-3];
          *pfVar1 = pfVar1[-2];
          pfVar6 = pfVar1 + -3;
          pfVar1 = pfVar1 + -2;
        }
        *param_1 = fVar2;
        param_1[1] = fVar4;
      }
      else {
        while( true ) {
          fVar3 = pfVar6[-2];
          if ((fVar3 <= fVar2) && ((fVar3 < fVar2 || ((int)pfVar6[-1] <= (int)fVar4)))) break;
          *pfVar6 = fVar3;
          pfVar6[1] = pfVar6[-1];
          pfVar6 = pfVar6 + -2;
        }
        *pfVar6 = fVar2;
        pfVar6[1] = fVar4;
      }
    }
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180014f30'}
Evidence: `ev_11bdfa8d932df03f742669c8e7cd5152216ac452065853af7d44379cddbe0358`
```c

void FUN_180014f30(int param_1,int param_2,float param_3,longlong param_4,longlong *param_5,
                  longlong *param_6)

{
  longlong lVar1;
  int iVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  
  if (0 < param_2) {
    iVar2 = 0;
    lVar6 = (longlong)param_1;
    lVar1 = 0;
    do {
      lVar5 = 0;
      if (3 < lVar6) {
        do {
          lVar4 = lVar5 * 4;
          lVar5 = lVar5 + 4;
          lVar3 = iVar2 * (int)param_5[1] + lVar4;
          *(float *)(lVar3 + *param_5) =
               param_3 * *(float *)(param_4 +
                                   (longlong)
                                   (int)*(float *)((int)param_6[1] * lVar1 + lVar4 + *param_6) * 4)
               + *(float *)(lVar3 + *param_5);
          lVar3 = iVar2 * (int)param_5[1] + lVar4;
          *(float *)(lVar3 + 4 + *param_5) =
               param_3 * *(float *)(param_4 +
                                   (longlong)
                                   (int)*(float *)((int)param_6[1] * lVar1 + lVar4 + 4 + *param_6) *
                                   4) + *(float *)(lVar3 + 4 + *param_5);
          lVar3 = (longlong)(iVar2 * (int)param_5[1]) + lVar4 + 8;
          *(float *)(lVar3 + *param_5) =
               param_3 * *(float *)(param_4 +
                                   (longlong)
                                   (int)*(float *)((int)param_6[1] * lVar1 + lVar4 + 8 + *param_6) *
                                   4) + *(float *)(lVar3 + *param_5);
          lVar3 = (longlong)(iVar2 * (int)param_5[1]) + lVar4 + 0xc;
          *(float *)(lVar3 + *param_5) =
               param_3 * *(float *)(param_4 +
                                   (longlong)
                                   (int)*(float *)((int)param_6[1] * lVar1 + lVar4 + 0xc + *param_6)
                                   * 4) + *(float *)(lVar3 + *param_5);
        } while (lVar5 < lVar6 + -3);
      }
      for (; lVar5 < lVar6; lVar5 = lVar5 + 1) {
        lVar4 = (longlong)(iVar2 * (int)param_5[1]) + lVar5 * 4;
        *(float *)(lVar4 + *param_5) =
             param_3 * *(float *)(param_4 +
                                 (longlong)
                                 (int)*(float *)((int)param_6[1] * lVar1 + lVar5 * 4 + *param_6) * 4
                                 ) + *(float *)(lVar4 + *param_5);
      }
      iVar2 = iVar2 + 1;
      lVar1 = lVar1 + 1;
    } while (iVar2 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180026e20'}
Evidence: `ev_12796908bc2e21d5629a0791722e100eb94dbca694fc47fdff7e6c22d5341d7d`
```c

void FUN_180026e20(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5)

{
  longlong lVar1;
  float fVar2;
  int iVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  float fVar10;
  float fVar11;
  float fVar12;
  float fVar13;
  
  if (0 < param_2) {
    lVar6 = 0;
    do {
      lVar7 = 0;
      if (0 < (longlong)param_1) {
        do {
          lVar1 = lVar7 * 4;
          lVar8 = (longlong)*(int *)(lVar1 + param_5[2]);
          iVar3 = *(int *)(param_5[3] + lVar6 * 4);
          fVar10 = *(float *)(lVar1 + *param_5);
          fVar2 = *(float *)(param_5[1] + lVar6 * 4);
          lVar9 = (longlong)(iVar3 * (int)param_3[1]) + *param_3;
          fVar13 = fVar2 * fVar10;
          fVar11 = (1.0 - fVar10) * fVar2;
          lVar5 = (longlong)((iVar3 + 1) * (int)param_3[1]) + *param_3;
          lVar4 = (longlong)(*(int *)(lVar1 + param_5[2]) + 1);
          fVar12 = (1.0 - fVar2) * (1.0 - fVar10);
          fVar10 = (1.0 - fVar2) * fVar10;
          lVar7 = lVar7 + 1;
          *(float *)((int)param_4[1] * lVar6 + lVar1 + *param_4) =
               (fVar12 * *(float *)(lVar9 + 4 + lVar8 * 0xc) +
                fVar10 * *(float *)(lVar9 + 4 + lVar4 * 0xc) +
                fVar11 * *(float *)(lVar5 + 4 + lVar8 * 0xc) +
               fVar13 * *(float *)(lVar5 + 4 + lVar4 * 0xc)) * 0.587 +
               (fVar12 * *(float *)(lVar9 + 8 + lVar8 * 0xc) +
                fVar10 * *(float *)(lVar9 + 8 + lVar4 * 0xc) +
                fVar11 * *(float *)(lVar5 + 8 + lVar8 * 0xc) +
               fVar13 * *(float *)(lVar5 + 8 + lVar4 * 0xc)) * 0.299 +
               (fVar12 * *(float *)(lVar9 + lVar8 * 0xc) + fVar10 * *(float *)(lVar9 + lVar4 * 0xc)
                + fVar11 * *(float *)(lVar5 + lVar8 * 0xc) +
               fVar13 * *(float *)(lVar5 + lVar4 * 0xc)) * 0.114;
        } while (lVar7 < param_1);
      }
      lVar6 = lVar6 + 1;
    } while (lVar6 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002ca70'}
Evidence: `ev_16e4bbdd4021475e874b3d8c2568e7167f066e8207b907d0d34d157b66424f41`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_18002ca70(Mat *param_1,uint param_2)

{
  int *piVar1;
  int iVar2;
  Mat *this;
  ulonglong uVar3;
  ulonglong uVar4;
  uint uVar5;
  ulonglong uVar6;
  undefined8 uVar7;
  undefined1 auStackY_1d8 [32];
  uint local_1a8;
  int local_1a4;
  Mat *local_1a0;
  undefined8 local_198;
  uint local_190 [26];
  undefined4 local_128;
  int local_124;
  undefined4 local_120;
  undefined4 local_11c;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  longlong local_f0;
  undefined4 *local_e8;
  undefined8 *local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined4 local_c8;
  undefined4 local_b8;
  int local_b4;
  undefined4 local_b0;
  undefined4 local_ac;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  longlong local_80;
  undefined4 *local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined4 local_58;
  ulonglong local_48;
  
  local_198 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_1d8;
  memset(&local_128,0,0x68);
  local_128 = *(undefined4 *)param_1;
  local_124 = *(int *)(param_1 + 4);
  local_120 = *(undefined4 *)(param_1 + 8);
  local_11c = *(undefined4 *)(param_1 + 0xc);
  local_118 = *(undefined8 *)(param_1 + 0x10);
  local_110 = *(undefined8 *)(param_1 + 0x18);
  local_108 = *(undefined8 *)(param_1 + 0x20);
  local_100 = *(undefined8 *)(param_1 + 0x28);
  local_f8 = *(undefined8 *)(param_1 + 0x30);
  local_f0 = *(longlong *)(param_1 + 0x38);
  local_e8 = &local_120;
  uVar4 = 0;
  local_d0 = 0;
  local_d8 = 0;
  if (local_f0 != 0) {
    LOCK();
    *(int *)(local_f0 + 0x14) = *(int *)(local_f0 + 0x14) + 1;
    UNLOCK();
  }
  local_e0 = &local_d8;
  if (*(int *)(param_1 + 4) < 3) {
    local_d8 = **(undefined8 **)(param_1 + 0x48);
    local_d0 = *(undefined8 *)(*(longlong *)(param_1 + 0x48) + 8);
  }
  else {
    local_124 = 0;
    cv::Mat::copySize((Mat *)&local_128,param_1);
  }
  local_c8 = *(undefined4 *)(param_1 + 0x60);
  FUN_18002f300((longlong)&local_128,param_2);
  if (local_f0 != 0) {
    LOCK();
    piVar1 = (int *)(local_f0 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_128);
    }
  }
  local_f0 = 0;
  local_118 = 0;
  local_100 = 0;
  local_108 = 0;
  local_110 = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < local_124) {
    do {
      *(undefined4 *)(uVar3 + (longlong)local_e8) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < local_124);
  }
  if (local_e0 != &local_d8) {
    cv::fastFree(local_e0);
  }
  iVar2 = **(int **)(param_1 + 0x40);
  uVar5 = (*(int **)(param_1 + 0x40))[1];
  uVar7 = 0x68;
  memset(&local_b8,0,0x68);
  local_1a8 = uVar5;
  local_1a4 = iVar2 + param_2 * 2;
  this = (Mat *)FUN_180011dc0(local_190,&local_1a8,uVar7,
                              *(longlong *)(param_1 + 0x10) -
                              (longlong)(int)param_2 * *(longlong *)(param_1 + 0x50),
                              (longlong)(int)*(longlong *)(param_1 + 0x50));
  local_b8 = *(undefined4 *)this;
  local_b4 = *(int *)(this + 4);
  local_b0 = *(undefined4 *)(this + 8);
  local_ac = *(undefined4 *)(this + 0xc);
  local_a8 = *(undefined8 *)(this + 0x10);
  local_a0 = *(undefined8 *)(this + 0x18);
  local_98 = *(undefined8 *)(this + 0x20);
  local_90 = *(undefined8 *)(this + 0x28);
  local_88 = *(undefined8 *)(this + 0x30);
  local_80 = *(longlong *)(this + 0x38);
  local_78 = &local_b0;
  local_70 = &local_68;
  local_60 = 0;
  local_68 = 0;
  if (local_80 != 0) {
    LOCK();
    *(int *)(local_80 + 0x14) = *(int *)(local_80 + 0x14) + 1;
    UNLOCK();
  }
  local_1a0 = this;
  if (*(int *)(this + 4) < 3) {
    local_68 = **(undefined8 **)(this + 0x48);
    local_60 = *(undefined8 *)(*(longlong *)(this + 0x48) + 8);
  }
  else {
    local_b4 = 0;
    cv::Mat::copySize((Mat *)&local_b8,this);
  }
  local_58 = 0;
  if (*(longlong *)(this + 0x38) != 0) {
    LOCK();
    piVar1 = (int *)(*(longlong *)(this + 0x38) + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(this);
    }
  }
  *(undefined8 *)(this + 0x38) = 0;
  *(undefined8 *)(this + 0x10) = 0;
  *(undefined8 *)(this + 0x28) = 0;
  *(undefined8 *)(this + 0x20) = 0;
  *(undefined8 *)(this + 0x18) = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < *(int *)(this + 4)) {
    do {
      *(undefined4 *)(uVar3 + *(longlong *)(this + 0x40)) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < *(int *)(this + 4));
  }
  if (*(Mat **)(this + 0x48) != this + 0x50) {
    cv::fastFree(*(Mat **)(this + 0x48));
  }
  FUN_18002f490((longlong)&local_b8,param_2);
  if (local_80 != 0) {
    LOCK();
    piVar1 = (int *)(local_80 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_b8);
    }
  }
  local_80 = 0;
  local_a8 = 0;
  local_90 = 0;
  local_98 = 0;
  local_a0 = 0;
  uVar3 = uVar4;
  if (0 < local_b4) {
    do {
      *(undefined4 *)((longlong)local_78 + uVar3) = 0;
      uVar5 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar5;
      uVar3 = uVar3 + 4;
    } while ((int)uVar5 < local_b4);
  }
  if (local_70 != &local_68) {
    cv::fastFree(local_70);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180025a80'}
Evidence: `ev_1883003ec377119bf29a2e465a30803e7f95176c7abd83a46a594d24182597f6`
```c

void FUN_180025a80(int *param_1,longlong param_2,longlong param_3,longlong param_4)

{
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  longlong local_18;
  undefined8 uStack_10;
  
  local_28 = *(longlong *)(param_4 + 0x10);
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_4 + 0x50));
  local_18 = *(longlong *)(param_3 + 0x10);
  uStack_20 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_38 = *(longlong *)(param_2 + 0x10);
  uStack_10 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_2 + 0x50));
  FUN_18002d520(*param_1,param_1[1],param_1[0x137],param_1[0x138],
                (float)param_1[0x137] / (float)*param_1,(float)param_1[0x138] / (float)param_1[1],
                &local_38,&local_18,&local_28);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x1800026d0'}
Evidence: `ev_18eda889156df99feaeddde06af3dc31862328d84f05733a9d9eab2ab8d0ab9e`
```c

int * FUN_1800026d0(int *param_1,undefined8 *param_2,undefined8 param_3,char *param_4)

{
  undefined8 uVar1;
  int *piVar2;
  undefined8 *puVar3;
  int *piVar4;
  int iVar5;
  int iVar6;
  undefined8 *puVar7;
  double *pdVar8;
  char *pcVar9;
  undefined8 local_20;
  undefined8 local_18;
  
  pcVar9 = (char *)0x4;
  piVar2 = param_1;
  puVar3 = param_2;
  do {
    puVar7 = puVar3;
    piVar4 = piVar2;
    uVar1 = puVar7[1];
    *(undefined8 *)piVar4 = *puVar7;
    *(undefined8 *)(piVar4 + 2) = uVar1;
    uVar1 = puVar7[3];
    *(undefined8 *)(piVar4 + 4) = puVar7[2];
    *(undefined8 *)(piVar4 + 6) = uVar1;
    uVar1 = puVar7[5];
    *(undefined8 *)(piVar4 + 8) = puVar7[4];
    *(undefined8 *)(piVar4 + 10) = uVar1;
    uVar1 = puVar7[7];
    *(undefined8 *)(piVar4 + 0xc) = puVar7[6];
    *(undefined8 *)(piVar4 + 0xe) = uVar1;
    uVar1 = puVar7[9];
    *(undefined8 *)(piVar4 + 0x10) = puVar7[8];
    *(undefined8 *)(piVar4 + 0x12) = uVar1;
    uVar1 = puVar7[0xb];
    *(undefined8 *)(piVar4 + 0x14) = puVar7[10];
    *(undefined8 *)(piVar4 + 0x16) = uVar1;
    uVar1 = puVar7[0xd];
    *(undefined8 *)(piVar4 + 0x18) = puVar7[0xc];
    *(undefined8 *)(piVar4 + 0x1a) = uVar1;
    uVar1 = puVar7[0xf];
    *(undefined8 *)(piVar4 + 0x1c) = puVar7[0xe];
    *(undefined8 *)(piVar4 + 0x1e) = uVar1;
    pcVar9 = pcVar9 + -1;
    piVar2 = piVar4 + 0x20;
    puVar3 = puVar7 + 0x10;
  } while (pcVar9 != (char *)0x0);
  uVar1 = puVar7[0x11];
  *(undefined8 *)(piVar4 + 0x20) = puVar7[0x10];
  *(undefined8 *)(piVar4 + 0x22) = uVar1;
  uVar1 = puVar7[0x13];
  *(undefined8 *)(piVar4 + 0x24) = puVar7[0x12];
  *(undefined8 *)(piVar4 + 0x26) = uVar1;
  uVar1 = puVar7[0x15];
  *(undefined8 *)(piVar4 + 0x28) = puVar7[0x14];
  *(undefined8 *)(piVar4 + 0x2a) = uVar1;
  iVar6 = 0;
  param_1[0x90] = 0;
  param_1[0x95] = 0;
  param_1[0x96] = 0;
  param_1[0x97] = 0;
  param_1[0x98] = 0;
  param_1[0x99] = 0;
  param_1[0x9a] = 0;
  param_1[0x9b] = 0;
  param_1[0x9c] = 0;
  param_1[0x9d] = 0;
  param_1[0x9e] = 0;
  param_1[0x9f] = 0;
  param_1[0xad] = 0;
  param_1[0xae] = 0;
  param_1[0xaf] = 0;
  param_1[0xb0] = 0;
  param_1[0xcf] = 0;
  param_1[0xd0] = 0;
  param_1[0xd1] = 0;
  param_1[0xd2] = 0;
  param_1[0xf1] = 0;
  param_1[0xf2] = 0;
  param_1[0xf3] = 0;
  param_1[0xf4] = 0;
  param_1[0x113] = 0;
  param_1[0x114] = 0;
  param_1[0x115] = 0;
  param_1[0x116] = 0;
  param_1[0x135] = 0;
  param_1[0x136] = 0;
  param_1[0x137] = 0;
  param_1[0x138] = 0;
  param_1[0x157] = 0;
  param_1[0x158] = 0;
  param_1[0x159] = 0;
  param_1[0x15a] = 0;
  param_1[0x179] = 0;
  param_1[0x17a] = 0;
  param_1[0x17b] = 0;
  param_1[0x17c] = 0;
  param_1[0x19b] = 0;
  param_1[0x19c] = 0;
  param_1[0x19d] = 0;
  param_1[0x19e] = 0;
  param_1[0x1bd] = 0;
  param_1[0x1be] = 0;
  param_1[0x1bf] = 0;
  param_1[0x1c0] = 0;
  param_1[0x1df] = 0;
  param_1[0x1e0] = 0;
  param_1[0x1e1] = 0;
  param_1[0x1e2] = 0;
  iVar5 = (param_1[6] != 8) + 1;
  if (param_1[2] == 0) {
    param_1[2] = param_1[5] * *param_1 * iVar5;
  }
  if (param_1[3] == 0) {
    param_1[3] = param_1[5] * *param_1 * iVar5;
  }
  if (param_1[4] == 0) {
    param_1[4] = *param_1 << 2;
  }
  FUN_180006bc0(param_1);
  if ((param_1[0x90] != 0) || (param_1[0x95] != 0)) {
    local_20 = 0;
    local_18 = 0;
    pcVar9 = cv::String::allocate((String *)&local_20,0x3d);
    builtin_strncpy(pcVar9,"vEnabledBandFastEFI.size() == 0 && vEnabledBand3D.size() == 0",0x3d);
    param_4 = "SweepGuideCommon.cpp";
    pcVar9 = "sg::ParamTableBase::ParamTableBase";
    cv::error(-0xd7,(String *)&local_20,"sg::ParamTableBase::ParamTableBase","SweepGuideCommon.cpp",
              0x19d);
    cv::String::deallocate((String *)&local_20);
  }
  if (0 < *(int *)(param_2 + 5)) {
    pdVar8 = (double *)(param_2 + 0xb);
    do {
      if (0.0 < pdVar8[-4]) {
        if (3 < param_1[0x90]) {
                    /* WARNING: Subroutine does not return */
          FUN_180001b90(&DAT_18007b8d0,pdVar8,pcVar9,param_4);
        }
        param_1[(longlong)param_1[0x90] + 0x8c] = iVar6;
        param_1[0x90] = param_1[0x90] + 1;
      }
      if (0.0 < *pdVar8) {
        if (3 < param_1[0x95]) {
                    /* WARNING: Subroutine does not return */
          FUN_180001b90(&DAT_18007b8d0,pdVar8,pcVar9,param_4);
        }
        param_1[(longlong)param_1[0x95] + 0x91] = iVar6;
        param_1[0x95] = param_1[0x95] + 1;
      }
      iVar6 = iVar6 + 1;
      pdVar8 = pdVar8 + 1;
    } while (iVar6 < *(int *)(param_2 + 5));
  }
  if ((param_1[0x90] == 0) || (param_1[0x95] == 0)) {
    local_20 = 0;
    local_18 = 0;
    pcVar9 = cv::String::allocate((String *)&local_20,0x3d);
    builtin_strncpy(pcVar9,"vEnabledBandFastEFI.size() != 0 && vEnabledBand3D.size() != 0",0x3d);
    cv::error(-0xd7,(String *)&local_20,"sg::ParamTableBase::ParamTableBase","SweepGuideCommon.cpp",
              0x1a6);
    cv::String::deallocate((String *)&local_20);
  }
  return param_1;
}


```

## procedure_pseudo_code {'procedure': '0x180003eb0'}
Evidence: `ev_193af52c5cbcfbc090bb739b1fad25a4aa60c1c547d194068a3c6c231c919967`
```c

void FUN_180003eb0(char param_1,int param_2,int param_3,int param_4,longlong *param_5,
                  longlong param_6,longlong param_7,float param_8,float param_9,float param_10,
                  longlong param_11,longlong param_12,longlong param_13)

{
  longlong local_58;
  undefined8 uStack_50;
  longlong local_48;
  undefined8 uStack_40;
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  longlong lStack_20;
  longlong local_18;
  longlong lStack_10;
  
  if (param_1 != '\0') {
    local_48 = *(longlong *)(param_13 + 0x10);
    uStack_50 = CONCAT44(uStack_50._4_4_,*(undefined4 *)(param_13 + 0x50));
    local_38 = *(longlong *)(param_12 + 0x10);
    uStack_40 = uStack_50;
    uStack_50 = CONCAT44(uStack_50._4_4_,*(undefined4 *)(param_12 + 0x50));
    local_58 = *(longlong *)(param_11 + 0x10);
    uStack_30 = uStack_50;
    uStack_50 = CONCAT44(uStack_50._4_4_,*(undefined4 *)(param_11 + 0x50));
    local_28 = *param_5;
    lStack_20 = param_5[1];
    local_18 = param_5[2];
    lStack_10 = param_5[3];
    FUN_180003960(param_3,param_4,&local_28,param_2,param_6,param_7,param_8,param_9,param_10,
                  &local_58,&local_38,&local_48);
    return;
  }
  local_38 = *(longlong *)(param_12 + 0x10);
  uStack_50 = CONCAT44(uStack_50._4_4_,*(undefined4 *)(param_12 + 0x50));
  local_58 = *(longlong *)(param_11 + 0x10);
  uStack_30 = uStack_50;
  uStack_50 = CONCAT44(uStack_50._4_4_,*(undefined4 *)(param_11 + 0x50));
  uStack_40 = uStack_50;
  local_28 = *param_5;
  lStack_20 = param_5[1];
  local_18 = param_5[2];
  lStack_10 = param_5[3];
  local_48 = local_58;
  FUN_180003c30(param_3,param_4,&local_28,param_2,param_6,param_7,param_8,param_9,param_10,&local_48
                ,&local_38);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180019a20'}
Evidence: `ev_267a8373fee07df1c6367138e7ef11c577a08ba2320363563f1de346c5d85947`
```c

/* WARNING: Function: __chkstk replaced with injection: alloca_probe */
/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180019a20(longlong param_1,longlong *param_2,Mat *param_3,Mat *param_4,undefined8 param_5,
                  uint param_6,double param_7)

{
  int *piVar1;
  Mat *pMVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  uint uVar6;
  ulonglong uVar7;
  ulonglong uVar8;
  longlong lVar9;
  Mat *pMVar10;
  ulonglong uVar11;
  longlong lVar12;
  undefined1 auStackY_1178 [32];
  undefined4 local_1130;
  undefined4 local_112c;
  int local_1128;
  int local_1124;
  undefined4 local_1120;
  undefined4 local_111c;
  undefined8 local_1118;
  int local_1110;
  int local_110c;
  undefined4 local_1108 [2];
  Mat *local_1100;
  undefined8 local_10f8;
  undefined4 local_10f0 [2];
  Mat *local_10e8;
  undefined8 local_10e0;
  undefined4 local_10d8 [2];
  Mat *local_10d0;
  undefined8 local_10c8;
  undefined4 local_10c0 [2];
  Mat *local_10b8;
  undefined8 local_10b0;
  undefined4 local_10a8 [2];
  Mat *local_10a0;
  undefined8 local_1098;
  undefined4 local_1090 [2];
  undefined8 *local_1088;
  undefined8 local_1080;
  undefined8 local_1078;
  longlong *local_1070;
  undefined8 local_1068;
  longlong local_1060;
  longlong local_1058;
  undefined8 local_1050;
  undefined8 local_1048;
  int aiStack_1040 [12];
  longlong lStack_1010;
  int *local_1008;
  undefined8 *local_1000;
  undefined8 local_ff8;
  undefined8 local_ff0;
  undefined4 local_fe8;
  Mat local_fd8 [4];
  int local_fd4;
  undefined8 local_fc8;
  undefined8 local_fc0;
  undefined8 local_fb8;
  undefined8 local_fb0;
  longlong local_fa0;
  longlong local_f98;
  undefined1 *local_f90;
  undefined1 local_f88 [16];
  undefined4 local_f78;
  Mat local_f68 [4];
  int local_f64;
  undefined8 local_f58;
  undefined8 local_f50;
  undefined8 local_f48;
  undefined8 local_f40;
  longlong local_f30;
  longlong local_f28;
  undefined1 *local_f20;
  undefined1 local_f18 [16];
  undefined4 local_f08;
  Mat local_ef8 [4];
  int local_ef4;
  undefined8 local_ee8;
  undefined8 local_ee0;
  undefined8 local_ed8;
  undefined8 local_ed0;
  longlong local_ec0;
  longlong local_eb8;
  undefined1 *local_eb0;
  undefined1 local_ea8 [16];
  undefined4 local_e98;
  Mat local_e88 [4];
  int local_e84;
  undefined8 local_e78;
  undefined8 local_e70;
  undefined8 local_e68;
  undefined8 local_e60;
  longlong local_e50;
  longlong local_e48;
  undefined1 *local_e40;
  undefined1 local_e38 [16];
  undefined4 local_e28;
  Mat local_e18 [4];
  int local_e14;
  undefined8 local_e08;
  undefined8 local_e00;
  undefined8 local_df8;
  undefined8 local_df0;
  longlong local_de0;
  longlong local_dd8;
  undefined1 *local_dd0;
  undefined1 local_dc8 [16];
  undefined4 local_db8;
  Mat local_da8 [4];
  int local_da4;
  undefined8 local_d98;
  undefined8 local_d90;
  undefined8 local_d88;
  undefined8 local_d80;
  longlong local_d70;
  longlong local_d68;
  undefined1 *local_d60;
  undefined1 local_d58 [16];
  undefined4 local_d48;
  longlong local_d38;
  undefined4 local_d30;
  undefined8 local_d28;
  undefined4 local_d20;
  undefined8 local_d18;
  undefined4 local_d10;
  longlong local_d08;
  undefined4 local_d00;
  undefined8 local_cf8;
  undefined4 local_cf0;
  undefined8 local_ce8;
  undefined4 local_ce0;
  undefined8 local_cd8;
  undefined4 local_cd0;
  undefined8 local_cc8;
  undefined4 local_cc0;
  undefined8 local_cb8;
  undefined4 local_cb0;
  longlong local_ca8;
  undefined4 local_ca0;
  undefined8 local_c98;
  undefined4 local_c90;
  undefined8 local_c88;
  undefined4 local_c80;
  undefined8 local_c78;
  undefined4 local_c70;
  undefined8 local_c68;
  undefined4 local_c60;
  undefined8 local_c58;
  undefined4 local_c50;
  Mat local_c48 [16];
  longlong local_c38;
  undefined4 local_bf8;
  undefined4 local_be8;
  Mat local_be0 [16];
  undefined8 local_bd0;
  undefined4 local_b90;
  undefined4 local_b80;
  Mat local_b78 [16];
  undefined8 local_b68;
  undefined4 local_b28;
  undefined4 local_b18;
  Mat local_b10 [16];
  undefined8 local_b00;
  undefined4 local_ac0;
  undefined4 local_ab0;
  Mat local_aa8 [16];
  undefined8 local_a98;
  undefined4 local_a58;
  undefined4 local_a48;
  Mat local_a40 [16];
  undefined8 local_a30;
  undefined4 local_9f0;
  undefined4 local_9e0;
  Mat local_9d8 [96];
  undefined4 local_978;
  Mat local_970 [96];
  undefined4 local_910;
  Mat local_908 [96];
  undefined4 local_8a8;
  Mat local_8a0 [96];
  undefined4 local_840;
  Mat local_838 [96];
  undefined4 local_7d8;
  Mat local_7d0 [96];
  undefined4 local_770;
  Mat local_768 [96];
  undefined4 local_708;
  Mat local_700 [96];
  undefined4 local_6a0;
  Mat local_698 [96];
  undefined4 local_638;
  Mat local_630 [96];
  undefined4 local_5d0;
  Mat local_5c8 [96];
  undefined4 local_568;
  Mat local_560 [96];
  undefined4 local_500;
  Mat local_4f8 [96];
  Mat local_498 [96];
  Mat local_438 [96];
  Mat local_3d8 [96];
  Mat local_378 [96];
  Mat local_318 [96];
  Mat local_2b8 [624];
  ulonglong local_48;
  undefined8 uStack_40;
  
  uStack_40 = 0x180019a3e;
  local_1078 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_1178;
  iVar3 = param_6 * 2 + 1;
  memset(&local_1048,0,0x68);
  local_1048 = 0x42ff0000;
  aiStack_1040[0] = 0;
  aiStack_1040[1] = 0;
  aiStack_1040[2] = 0;
  aiStack_1040[3] = 0;
  aiStack_1040[4] = 0;
  aiStack_1040[5] = 0;
  aiStack_1040[6] = 0;
  aiStack_1040[7] = 0;
  aiStack_1040[8] = 0;
  aiStack_1040[9] = 0;
  aiStack_1040[10] = 0;
  aiStack_1040[0xb] = 0;
  lStack_1010 = 0;
  local_1008 = aiStack_1040;
  local_1000 = &local_ff8;
  local_ff0 = 0;
  local_ff8 = 0;
  local_fe8 = 0;
  FUN_1800726b8((longlong)local_2b8,0x68,6,&LAB_180006700);
  FUN_1800726b8((longlong)local_c48,0x68,6,&LAB_180006700);
  local_1050 = 0;
  local_1068 = 0x20;
  local_1060 = 0;
  local_1058 = param_2[1] + *param_2;
  lVar12 = 3;
  local_1070 = param_2;
  FUN_180025f50((longlong *)&local_1070,param_1,param_6,3,&local_1048,local_2b8,local_c48);
  iVar4 = *local_1008;
  iVar5 = local_1008[1];
  pMVar10 = param_4 + 0x138;
  local_1080 = 0;
  local_1090[0] = 0x2010000;
  local_1088 = &local_1048;
  local_10b0 = 0;
  local_10c0[0] = 0x1010000;
  local_1118 = 0;
  local_10b8 = param_3;
  cv::resize(local_10c0,local_1090,&local_1118);
  FUN_1800013f0(local_4f8,param_4 + 0xd0);
  FUN_1800013f0(local_498,param_4 + 0x68);
  FUN_1800013f0(local_438,param_4);
  FUN_1800013f0(local_3d8,param_4 + 0x208);
  FUN_1800013f0(local_378,param_4 + 0x1a0);
  FUN_1800013f0(local_318,pMVar10);
  cv::split(param_3,local_4f8);
  cv::split((Mat *)&local_1048,local_3d8);
  _eh_vector_destructor_iterator_(local_3d8,0x60,3,FUN_180001750);
  _eh_vector_destructor_iterator_(local_4f8,0x60,3,FUN_180001750);
  lVar9 = 0x270 - (longlong)pMVar10;
  do {
    local_10f8 = 0;
    local_1108[0] = 0x2010000;
    local_1100 = pMVar10 + (longlong)(param_4 + lVar9);
    local_10e0 = 0;
    local_10f0[0] = 0x1010000;
    local_1130 = 0xffffffff;
    local_112c = 0xffffffff;
    local_1128 = iVar3;
    local_1124 = iVar3;
    local_10e8 = pMVar10;
    cv::blur(local_10f0,local_1108,&local_1128,&local_1130);
    pMVar10 = pMVar10 + 0x68;
    lVar12 = lVar12 + -1;
  } while (lVar12 != 0);
  pMVar10 = param_4 + 0x138;
  FUN_1800013f0(local_9d8,pMVar10);
  local_978 = *(undefined4 *)(param_4 + 0x198);
  FUN_1800013f0(local_970,pMVar10);
  local_910 = *(undefined4 *)(param_4 + 0x198);
  FUN_1800013f0(local_908,pMVar10);
  local_8a8 = *(undefined4 *)(param_4 + 0x198);
  FUN_1800013f0(local_8a0,param_4 + 0x1a0);
  local_840 = *(undefined4 *)(param_4 + 0x200);
  FUN_1800013f0(local_838,param_4 + 0x1a0);
  local_7d8 = *(undefined4 *)(param_4 + 0x200);
  pMVar2 = param_4 + 0x208;
  FUN_1800013f0(local_7d0,pMVar2);
  local_770 = *(undefined4 *)(param_4 + 0x268);
  FUN_1800013f0(local_768,pMVar10);
  local_708 = *(undefined4 *)(param_4 + 0x198);
  FUN_1800013f0(local_700,param_4 + 0x1a0);
  local_6a0 = *(undefined4 *)(param_4 + 0x200);
  FUN_1800013f0(local_698,pMVar2);
  local_638 = *(undefined4 *)(param_4 + 0x268);
  FUN_1800013f0(local_630,param_4 + 0x1a0);
  local_5d0 = *(undefined4 *)(param_4 + 0x200);
  FUN_1800013f0(local_5c8,pMVar2);
  local_568 = *(undefined4 *)(param_4 + 0x268);
  FUN_1800013f0(local_560,pMVar2);
  local_500 = *(undefined4 *)(param_4 + 0x268);
  FUN_180017b70((longlong)local_9d8,(longlong)local_768,(longlong)local_2b8);
  _eh_vector_destructor_iterator_(local_768,0x68,6,FUN_180001750);
  _eh_vector_destructor_iterator_(local_9d8,0x68,6,FUN_180001750);
  memset(local_e18,0,0x68);
  FUN_1800013f0(local_e18,local_c48);
  local_db8 = local_be8;
  memset(local_da8,0,0x68);
  FUN_1800013f0(local_da8,local_be0);
  local_d48 = local_b80;
  memset(local_e88,0,0x68);
  FUN_1800013f0(local_e88,local_b78);
  local_e28 = local_b18;
  memset(local_f68,0,0x68);
  FUN_1800013f0(local_f68,local_b10);
  local_f08 = local_ab0;
  memset(local_fd8,0,0x68);
  FUN_1800013f0(local_fd8,local_aa8);
  local_f78 = local_a48;
  memset(local_ef8,0,0x68);
  FUN_1800013f0(local_ef8,local_a40);
  local_e98 = local_9e0;
  uVar7 = 0;
  lVar9 = 6;
  uVar11 = uVar7;
  do {
    local_10c8 = 0;
    local_10d8[0] = 0x2010000;
    local_10d0 = local_c48 + uVar11;
    local_1098 = 0;
    local_10a8[0] = 0x1010000;
    local_10a0 = local_2b8 + uVar11;
    local_1120 = 0xffffffff;
    local_111c = 0xffffffff;
    local_1110 = iVar3;
    local_110c = iVar3;
    cv::blur(local_10a8,local_10d8,&local_1110,&local_1120);
    uVar11 = uVar11 + 0x68;
    lVar9 = lVar9 + -1;
  } while (lVar9 != 0);
  local_d38 = *(longlong *)(param_4 + 0x280);
  local_d30 = *(undefined4 *)(param_4 + 0x2c0);
  local_d28 = *(undefined8 *)(param_4 + 0x2e8);
  local_d20 = *(undefined4 *)(param_4 + 0x328);
  local_d18 = *(undefined8 *)(param_4 + 0x350);
  local_d10 = (undefined4)*(undefined8 *)(param_4 + 0x390);
  local_d08 = local_c38;
  local_d00 = local_bf8;
  local_cf8 = local_bd0;
  local_cf0 = local_b90;
  local_ce8 = local_b68;
  local_ce0 = local_b28;
  local_cd8 = local_b00;
  local_cd0 = local_ac0;
  local_cc8 = local_a98;
  local_cc0 = local_a58;
  local_cb8 = local_a30;
  local_cb0 = local_9f0;
  local_ca8 = *(longlong *)(param_4 + 0x3b8);
  local_ca0 = *(undefined4 *)(param_4 + 0x3f8);
  local_c98 = *(undefined8 *)(param_4 + 0x420);
  local_c90 = *(undefined4 *)(param_4 + 0x460);
  local_c88 = *(undefined8 *)(param_4 + 0x488);
  local_c80 = *(undefined4 *)(param_4 + 0x4c8);
  local_c78 = *(undefined8 *)(param_4 + 0x4f0);
  local_c70 = *(undefined4 *)(param_4 + 0x530);
  local_c68 = *(undefined8 *)(param_4 + 0x558);
  local_c60 = *(undefined4 *)(param_4 + 0x598);
  local_c58 = *(undefined8 *)(param_4 + 0x5c0);
  local_c50 = (undefined4)*(undefined8 *)(param_4 + 0x600);
  FUN_180019740(iVar5,iVar4,(float)param_7,&local_d38,&local_d08,&local_ca8);
  if (local_ec0 != 0) {
    LOCK();
    piVar1 = (int *)(local_ec0 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate(local_ef8);
    }
  }
  local_ec0 = 0;
  local_ee8 = 0;
  local_ed0 = 0;
  local_ed8 = 0;
  local_ee0 = 0;
  uVar11 = uVar7;
  uVar8 = uVar7;
  if (0 < local_ef4) {
    do {
      *(undefined4 *)(local_eb8 + uVar8) = 0;
      uVar6 = (int)uVar11 + 1;
      uVar11 = (ulonglong)uVar6;
      uVar8 = uVar8 + 4;
    } while ((int)uVar6 < local_ef4);
  }
  if (local_eb0 != local_ea8) {
    cv::fastFree(local_eb0);
  }
  if (local_fa0 != 0) {
    LOCK();
    piVar1 = (int *)(local_fa0 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate(local_fd8);
    }
  }
  local_fa0 = 0;
  local_fc8 = 0;
  local_fb0 = 0;
  local_fb8 = 0;
  local_fc0 = 0;
  uVar11 = uVar7;
  uVar8 = uVar7;
  if (0 < local_fd4) {
    do {
      *(undefined4 *)(local_f98 + uVar8) = 0;
      uVar6 = (int)uVar11 + 1;
      uVar11 = (ulonglong)uVar6;
      uVar8 = uVar8 + 4;
    } while ((int)uVar6 < local_fd4);
  }
  if (local_f90 != local_f88) {
    cv::fastFree(local_f90);
  }
  if (local_f30 != 0) {
    LOCK();
    piVar1 = (int *)(local_f30 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate(local_f68);
    }
  }
  local_f30 = 0;
  local_f58 = 0;
  local_f40 = 0;
  local_f48 = 0;
  local_f50 = 0;
  uVar11 = uVar7;
  uVar8 = uVar7;
  if (0 < local_f64) {
    do {
      *(undefined4 *)(local_f28 + uVar8) = 0;
      uVar6 = (int)uVar11 + 1;
      uVar11 = (ulonglong)uVar6;
      uVar8 = uVar8 + 4;
    } while ((int)uVar6 < local_f64);
  }
  if (local_f20 != local_f18) {
    cv::fastFree(local_f20);
  }
  if (local_e50 != 0) {
    LOCK();
    piVar1 = (int *)(local_e50 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate(local_e88);
    }
  }
  local_e50 = 0;
  local_e78 = 0;
  local_e60 = 0;
  local_e68 = 0;
  local_e70 = 0;
  uVar11 = uVar7;
  uVar8 = uVar7;
  if (0 < local_e84) {
    do {
      *(undefined4 *)(local_e48 + uVar8) = 0;
      uVar6 = (int)uVar11 + 1;
      uVar11 = (ulonglong)uVar6;
      uVar8 = uVar8 + 4;
    } while ((int)uVar6 < local_e84);
  }
  if (local_e40 != local_e38) {
    cv::fastFree(local_e40);
  }
  if (local_d70 != 0) {
    LOCK();
    piVar1 = (int *)(local_d70 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate(local_da8);
    }
  }
  local_d70 = 0;
  local_d98 = 0;
  local_d80 = 0;
  local_d88 = 0;
  local_d90 = 0;
  uVar11 = uVar7;
  uVar8 = uVar7;
  if (0 < local_da4) {
    do {
      *(undefined4 *)(local_d68 + uVar8) = 0;
      uVar6 = (int)uVar11 + 1;
      uVar11 = (ulonglong)uVar6;
      uVar8 = uVar8 + 4;
    } while ((int)uVar6 < local_da4);
  }
  if (local_d60 != local_d58) {
    cv::fastFree(local_d60);
  }
  if (local_de0 != 0) {
    LOCK();
    piVar1 = (int *)(local_de0 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate(local_e18);
    }
  }
  local_de0 = 0;
  local_e08 = 0;
  local_df0 = 0;
  local_df8 = 0;
  local_e00 = 0;
  uVar11 = uVar7;
  uVar8 = uVar7;
  if (0 < local_e14) {
    do {
      *(undefined4 *)(local_dd8 + uVar8) = 0;
      uVar6 = (int)uVar11 + 1;
      uVar11 = (ulonglong)uVar6;
      uVar8 = uVar8 + 4;
    } while ((int)uVar6 < local_e14);
  }
  if (local_dd0 != local_dc8) {
    cv::fastFree(local_dd0);
  }
  local_1070[1] = local_1070[1] - local_1060;
  _eh_vector_destructor_iterator_(local_c48,0x68,6,FUN_180001750);
  _eh_vector_destructor_iterator_(local_2b8,0x68,6,FUN_180001750);
  if (lStack_1010 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_1010 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate((Mat *)&local_1048);
    }
  }
  lStack_1010 = 0;
  aiStack_1040[2] = 0;
  aiStack_1040[3] = 0;
  aiStack_1040[8] = 0;
  aiStack_1040[9] = 0;
  aiStack_1040[6] = 0;
  aiStack_1040[7] = 0;
  aiStack_1040[4] = 0;
  aiStack_1040[5] = 0;
  uVar11 = uVar7;
  if (0 < local_1048._4_4_) {
    do {
      *(undefined4 *)((longlong)local_1008 + uVar11) = 0;
      uVar6 = (int)uVar7 + 1;
      uVar7 = (ulonglong)uVar6;
      uVar11 = uVar11 + 4;
    } while ((int)uVar6 < local_1048._4_4_);
  }
  if (local_1000 != &local_ff8) {
    cv::fastFree(local_1000);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180018360'}
Evidence: `ev_2b1e98e93345bc9e609ffd49f0a3cecc3c5f5e9a8627a490aeabd9539805a3ca`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180018360(longlong param_1,longlong param_2,longlong param_3,longlong param_4,int param_5)

{
  int iVar1;
  int iVar2;
  undefined1 auStackY_1c8 [32];
  longlong local_188;
  undefined8 uStack_180;
  longlong local_178;
  undefined8 uStack_170;
  longlong local_168;
  undefined8 uStack_160;
  longlong local_158;
  undefined8 uStack_150;
  longlong local_148;
  undefined8 uStack_140;
  longlong local_138;
  undefined8 uStack_130;
  longlong local_128;
  undefined4 local_120;
  undefined8 local_118;
  undefined4 local_110;
  undefined8 local_108;
  undefined4 local_100;
  longlong local_f8;
  undefined4 local_f0;
  undefined8 local_e8;
  undefined4 local_e0;
  undefined8 local_d8;
  undefined4 local_d0;
  undefined8 local_c8;
  undefined4 local_c0;
  longlong local_b8;
  undefined4 local_b0;
  undefined8 local_a8;
  undefined4 local_a0;
  undefined8 local_98;
  undefined4 local_90;
  undefined8 local_88;
  undefined4 local_80;
  longlong local_78;
  undefined4 local_70;
  undefined8 local_68;
  undefined4 local_60;
  undefined8 local_58;
  undefined4 local_50;
  undefined8 local_48;
  undefined4 local_40;
  undefined8 local_38;
  undefined4 local_30;
  undefined8 local_28;
  undefined4 local_20;
  ulonglong local_18;
  
  local_18 = DAT_18008b080 ^ (ulonglong)auStackY_1c8;
  iVar1 = (*(int **)(param_1 + 0x40))[1];
  iVar2 = **(int **)(param_1 + 0x40);
  if (param_5 == 1) {
    local_158 = *(longlong *)(param_4 + 0x10);
    uStack_180 = CONCAT44(uStack_180._4_4_,*(undefined4 *)(param_4 + 0x50));
    local_138 = *(longlong *)(param_4 + 0x78);
    uStack_150 = uStack_180;
    uStack_180 = CONCAT44(uStack_180._4_4_,*(undefined4 *)(param_4 + 0xb8));
    local_148 = *(longlong *)(param_3 + 0x10);
    uStack_130 = uStack_180;
    uStack_180 = CONCAT44(uStack_180._4_4_,*(undefined4 *)(param_3 + 0x50));
    local_178 = *(longlong *)(param_2 + 0x10);
    uStack_140 = uStack_180;
    uStack_180 = CONCAT44(uStack_180._4_4_,*(undefined4 *)(param_2 + 0x50));
    local_168 = *(longlong *)(param_1 + 0x78);
    uStack_170 = uStack_180;
    uStack_180 = CONCAT44(uStack_180._4_4_,*(undefined4 *)(param_1 + 0xb8));
    local_188 = *(longlong *)(param_1 + 0x10);
    uStack_160 = uStack_180;
    uStack_180 = CONCAT44(uStack_180._4_4_,*(undefined4 *)(param_1 + 0x50));
    FUN_180017d60(iVar1,iVar2,&local_188,&local_168,&local_178,&local_148,&local_138,&local_158);
  }
  else {
    local_f8 = *(longlong *)(param_1 + 0x10);
    local_f0 = *(undefined4 *)(param_1 + 0x50);
    local_e8 = *(undefined8 *)(param_1 + 0x78);
    local_e0 = *(undefined4 *)(param_1 + 0xb8);
    local_d8 = *(undefined8 *)(param_1 + 0xe0);
    local_d0 = *(undefined4 *)(param_1 + 0x120);
    local_c8 = *(undefined8 *)(param_1 + 0x148);
    local_c0 = *(undefined4 *)(param_1 + 0x188);
    local_128 = *(longlong *)(param_2 + 0x10);
    local_120 = *(undefined4 *)(param_2 + 0x50);
    local_118 = *(undefined8 *)(param_2 + 0x78);
    local_110 = *(undefined4 *)(param_2 + 0xb8);
    local_108 = *(undefined8 *)(param_2 + 0xe0);
    local_100 = *(undefined4 *)(param_2 + 0x120);
    local_78 = *(longlong *)(param_3 + 0x10);
    local_70 = *(undefined4 *)(param_3 + 0x50);
    local_68 = *(undefined8 *)(param_3 + 0x78);
    local_60 = *(undefined4 *)(param_3 + 0xb8);
    local_58 = *(undefined8 *)(param_3 + 0xe0);
    local_50 = *(undefined4 *)(param_3 + 0x120);
    local_48 = *(undefined8 *)(param_3 + 0x148);
    local_40 = *(undefined4 *)(param_3 + 0x188);
    local_38 = *(undefined8 *)(param_3 + 0x1b0);
    local_30 = *(undefined4 *)(param_3 + 0x1f0);
    local_28 = *(undefined8 *)(param_3 + 0x218);
    local_20 = *(undefined4 *)(param_3 + 600);
    local_b8 = *(longlong *)(param_4 + 0x10);
    local_b0 = *(undefined4 *)(param_4 + 0x50);
    local_a8 = *(undefined8 *)(param_4 + 0x78);
    local_a0 = *(undefined4 *)(param_4 + 0xb8);
    local_98 = *(undefined8 *)(param_4 + 0xe0);
    local_90 = *(undefined4 *)(param_4 + 0x120);
    local_88 = *(undefined8 *)(param_4 + 0x148);
    local_80 = *(undefined4 *)(param_4 + 0x188);
    FUN_1800180c0(iVar1,iVar2,&local_f8,&local_128,&local_78,&local_b8);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002eb70'}
Evidence: `ev_356e12e550a22e33223010bdea23ee79f0eab699a67b5f2d337e56680a877d0c`
```c

float * FUN_18002eb70(float *param_1,int param_2,int param_3,longlong *param_4,longlong *param_5)

{
  float fVar1;
  float fVar2;
  float fVar3;
  float fVar4;
  float fVar5;
  int iVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  float fVar12;
  float fVar13;
  float fVar14;
  float fVar15;
  float fVar16;
  
  lVar10 = (longlong)param_2;
  lVar9 = (longlong)param_3;
  lVar8 = (longlong)*(int *)(param_4[2] + lVar10 * 4);
  iVar6 = *(int *)(param_4[3] + lVar9 * 4);
  fVar1 = *(float *)(param_5 + 2);
  fVar16 = *(float *)(*param_4 + lVar10 * 4);
  lVar7 = (longlong)((int)param_5[1] * iVar6) + *param_5;
  lVar11 = (longlong)((iVar6 + 1) * (int)param_5[1]) + *param_5;
  fVar2 = *(float *)(lVar11 + 4 + lVar8 * 4);
  fVar12 = *(float *)(param_4[1] + lVar9 * 4) * fVar16;
  fVar3 = *(float *)(lVar11 + lVar8 * 4);
  fVar15 = *(float *)(param_4[1] + lVar9 * 4);
  fVar16 = 1.0 - fVar16;
  fVar13 = fVar16 * fVar15;
  fVar4 = *(float *)(lVar7 + 4 + lVar8 * 4);
  fVar15 = 1.0 - fVar15;
  fVar5 = *(float *)(lVar7 + lVar8 * 4);
  fVar14 = fVar15 * *(float *)(*param_4 + lVar10 * 4);
  fVar15 = fVar15 * fVar16;
  *param_1 = (float)(fVar5 == fVar1 + 0.0) * fVar15 + (float)(fVar4 == fVar1 + 0.0) * fVar14 +
             (float)(fVar3 == fVar1 + 0.0) * fVar13 + (float)(fVar2 == fVar1 + 0.0) * fVar12;
  param_1[3] = (float)(fVar5 == fVar1 + 3.0) * fVar15 + (float)(fVar4 == fVar1 + 3.0) * fVar14 +
               (float)(fVar3 == fVar1 + 3.0) * fVar13 + (float)(fVar2 == fVar1 + 3.0) * fVar12;
  param_1[1] = (float)(fVar5 == fVar1 + 1.0) * fVar15 + (float)(fVar4 == fVar1 + 1.0) * fVar14 +
               (float)(fVar3 == fVar1 + 1.0) * fVar13 + (float)(fVar2 == fVar1 + 1.0) * fVar12;
  param_1[2] = (float)(fVar5 == fVar1 + 2.0) * fVar15 + (float)(fVar4 == fVar1 + 2.0) * fVar14 +
               (float)(fVar3 == fVar1 + 2.0) * fVar13 + (float)(fVar2 == fVar1 + 2.0) * fVar12;
  return param_1;
}


```

## procedure_pseudo_code {'procedure': '0x18002c920'}
Evidence: `ev_379548d6eac46cc70e7aa9e397cf47d705dd12442ce7829a863aa4e51528be7e`
```c

void FUN_18002c920(longlong param_1,int param_2)

{
  int iVar1;
  undefined4 uVar2;
  undefined4 *puVar3;
  undefined4 *puVar4;
  undefined4 *puVar5;
  longlong lVar6;
  longlong lVar7;
  int iVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  
  iVar8 = 0;
  iVar1 = **(int **)(param_1 + 0x40);
  lVar7 = (longlong)param_2;
  if (0 < iVar1) {
    lVar10 = (longlong)(*(int **)(param_1 + 0x40))[1];
    do {
      lVar6 = 0;
      lVar9 = (longlong)(*(int *)(param_1 + 0x50) * iVar8) + *(longlong *)(param_1 + 0x10);
      if (3 < lVar7) {
        lVar11 = (lVar7 - 4U >> 2) + 1;
        lVar6 = lVar11 * 4;
        puVar3 = (undefined4 *)(lVar9 + 4);
        puVar4 = (undefined4 *)(lVar9 + -8);
        puVar5 = (undefined4 *)(lVar9 + (lVar10 + -3) * 4);
        do {
          puVar4[1] = *puVar3;
          puVar3[lVar10 + -1] = puVar5[1];
          *puVar4 = puVar3[1];
          puVar3[lVar10] = *puVar5;
          puVar4[-1] = puVar3[2];
          puVar3[lVar10 + 1] = puVar5[-1];
          puVar4[-2] = puVar3[3];
          puVar3[lVar10 + 2] = puVar5[-2];
          lVar11 = lVar11 + -1;
          puVar3 = puVar3 + 4;
          puVar4 = puVar4 + -4;
          puVar5 = puVar5 + -4;
        } while (lVar11 != 0);
      }
      if (lVar6 < lVar7) {
        puVar3 = (undefined4 *)(lVar9 + ((lVar10 - lVar6) + -2) * 4);
        lVar11 = lVar7 - lVar6;
        puVar4 = (undefined4 *)(lVar9 + (lVar10 + lVar6) * 4);
        puVar5 = (undefined4 *)(lVar9 - (lVar6 * 4 + 4));
        do {
          *puVar5 = puVar4[1 - lVar10];
          uVar2 = *puVar3;
          puVar3 = puVar3 + -1;
          *puVar4 = uVar2;
          lVar11 = lVar11 + -1;
          puVar4 = puVar4 + 1;
          puVar5 = puVar5 + -1;
        } while (lVar11 != 0);
      }
      iVar8 = iVar8 + 1;
    } while (iVar8 < iVar1);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x1800256b0'}
Evidence: `ev_41f835454dc3b84203c95a34b03b1c0ef82939c87d0ebb105ff700a19f2f5b99`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_1800256b0(Mat *param_1,uint param_2)

{
  int *piVar1;
  int iVar2;
  Mat *this;
  ulonglong uVar3;
  ulonglong uVar4;
  uint uVar5;
  ulonglong uVar6;
  undefined8 uVar7;
  undefined1 auStackY_1d8 [32];
  uint local_1a8;
  int local_1a4;
  Mat *local_1a0;
  undefined8 local_198;
  uint local_190 [26];
  undefined4 local_128;
  int local_124;
  undefined4 local_120;
  undefined4 local_11c;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  longlong local_f0;
  undefined4 *local_e8;
  undefined8 *local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined4 local_c8;
  undefined4 local_b8;
  int local_b4;
  undefined4 local_b0;
  undefined4 local_ac;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  longlong local_80;
  undefined4 *local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined4 local_58;
  ulonglong local_48;
  
  local_198 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_1d8;
  memset(&local_128,0,0x68);
  local_128 = *(undefined4 *)param_1;
  local_124 = *(int *)(param_1 + 4);
  local_120 = *(undefined4 *)(param_1 + 8);
  local_11c = *(undefined4 *)(param_1 + 0xc);
  local_118 = *(undefined8 *)(param_1 + 0x10);
  local_110 = *(undefined8 *)(param_1 + 0x18);
  local_108 = *(undefined8 *)(param_1 + 0x20);
  local_100 = *(undefined8 *)(param_1 + 0x28);
  local_f8 = *(undefined8 *)(param_1 + 0x30);
  local_f0 = *(longlong *)(param_1 + 0x38);
  local_e8 = &local_120;
  uVar4 = 0;
  local_d0 = 0;
  local_d8 = 0;
  if (local_f0 != 0) {
    LOCK();
    *(int *)(local_f0 + 0x14) = *(int *)(local_f0 + 0x14) + 1;
    UNLOCK();
  }
  local_e0 = &local_d8;
  if (*(int *)(param_1 + 4) < 3) {
    local_d8 = **(undefined8 **)(param_1 + 0x48);
    local_d0 = *(undefined8 *)(*(longlong *)(param_1 + 0x48) + 8);
  }
  else {
    local_124 = 0;
    cv::Mat::copySize((Mat *)&local_128,param_1);
  }
  local_c8 = *(undefined4 *)(param_1 + 0x60);
  FUN_18002c790((longlong)&local_128,param_2);
  if (local_f0 != 0) {
    LOCK();
    piVar1 = (int *)(local_f0 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_128);
    }
  }
  local_f0 = 0;
  local_118 = 0;
  local_100 = 0;
  local_108 = 0;
  local_110 = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < local_124) {
    do {
      *(undefined4 *)(uVar3 + (longlong)local_e8) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < local_124);
  }
  if (local_e0 != &local_d8) {
    cv::fastFree(local_e0);
  }
  iVar2 = **(int **)(param_1 + 0x40);
  uVar5 = (*(int **)(param_1 + 0x40))[1];
  uVar7 = 0x68;
  memset(&local_b8,0,0x68);
  local_1a8 = uVar5;
  local_1a4 = iVar2 + param_2 * 2;
  this = (Mat *)FUN_180011dc0(local_190,&local_1a8,uVar7,
                              *(longlong *)(param_1 + 0x10) -
                              (longlong)(int)param_2 * *(longlong *)(param_1 + 0x50),
                              (longlong)(int)*(longlong *)(param_1 + 0x50));
  local_b8 = *(undefined4 *)this;
  local_b4 = *(int *)(this + 4);
  local_b0 = *(undefined4 *)(this + 8);
  local_ac = *(undefined4 *)(this + 0xc);
  local_a8 = *(undefined8 *)(this + 0x10);
  local_a0 = *(undefined8 *)(this + 0x18);
  local_98 = *(undefined8 *)(this + 0x20);
  local_90 = *(undefined8 *)(this + 0x28);
  local_88 = *(undefined8 *)(this + 0x30);
  local_80 = *(longlong *)(this + 0x38);
  local_78 = &local_b0;
  local_70 = &local_68;
  local_60 = 0;
  local_68 = 0;
  if (local_80 != 0) {
    LOCK();
    *(int *)(local_80 + 0x14) = *(int *)(local_80 + 0x14) + 1;
    UNLOCK();
  }
  local_1a0 = this;
  if (*(int *)(this + 4) < 3) {
    local_68 = **(undefined8 **)(this + 0x48);
    local_60 = *(undefined8 *)(*(longlong *)(this + 0x48) + 8);
  }
  else {
    local_b4 = 0;
    cv::Mat::copySize((Mat *)&local_b8,this);
  }
  local_58 = 0;
  if (*(longlong *)(this + 0x38) != 0) {
    LOCK();
    piVar1 = (int *)(*(longlong *)(this + 0x38) + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(this);
    }
  }
  *(undefined8 *)(this + 0x38) = 0;
  *(undefined8 *)(this + 0x10) = 0;
  *(undefined8 *)(this + 0x28) = 0;
  *(undefined8 *)(this + 0x20) = 0;
  *(undefined8 *)(this + 0x18) = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < *(int *)(this + 4)) {
    do {
      *(undefined4 *)(uVar3 + *(longlong *)(this + 0x40)) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < *(int *)(this + 4));
  }
  if (*(Mat **)(this + 0x48) != this + 0x50) {
    cv::fastFree(*(Mat **)(this + 0x48));
  }
  FUN_18002c920((longlong)&local_b8,param_2);
  if (local_80 != 0) {
    LOCK();
    piVar1 = (int *)(local_80 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_b8);
    }
  }
  local_80 = 0;
  local_a8 = 0;
  local_90 = 0;
  local_98 = 0;
  local_a0 = 0;
  uVar3 = uVar4;
  if (0 < local_b4) {
    do {
      *(undefined4 *)((longlong)local_78 + uVar3) = 0;
      uVar5 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar5;
      uVar3 = uVar3 + 4;
    } while ((int)uVar5 < local_b4);
  }
  if (local_70 != &local_68) {
    cv::fastFree(local_70);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180002a40'}
Evidence: `ev_420883bfd1cec47d848b0c5817566460374b6361df8490a242af65cd8333ea2c`
```c

undefined8 FUN_180002a40(int *param_1)

{
  uint uVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  longlong lVar6;
  double *pdVar7;
  longlong lVar8;
  longlong lVar9;
  
  iVar4 = *param_1;
  if ((iVar4 < 1) || (param_1[1] < 1)) {
    return 0x1001;
  }
  iVar3 = param_1[5];
  if ((iVar3 - 1U & 0xfffffffd) != 0) {
    return 0x1002;
  }
  iVar5 = param_1[6];
  if ((iVar5 - 8U & 0xfffffff7) != 0) {
    return 0x1003;
  }
  iVar5 = ((int)((iVar5 >> 0x1f & 7U) + iVar5) >> 3) * iVar3;
  if ((((param_1[2] != 0) && (param_1[2] < iVar4 * iVar5)) ||
      ((param_1[3] != 0 && (param_1[3] < iVar4 * iVar5)))) ||
     ((param_1[4] != 0 && (param_1[4] < iVar4 * 4)))) {
    return 0x1004;
  }
  if (0x7ffe < param_1[7] - 1U) {
    return 0x1005;
  }
  if ((0 < param_1[8]) && (param_1[8] <= param_1[7])) {
    iVar4 = param_1[10];
    lVar6 = (longlong)iVar4;
    if (3 < iVar4 - 1U) {
      return 0x1007;
    }
    if ((param_1[0xb] < 0) || (iVar4 <= param_1[0xb])) {
      return 0x1008;
    }
    uVar2 = param_1[0xc];
    if (0 < (int)uVar2) {
      uVar1 = uVar2 & 0x80000001;
      if ((int)uVar1 < 0) {
        uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
      }
      if (((uVar1 != 0) && ((int)uVar2 < 0x21)) && (uVar2 = param_1[0xd], 0 < (int)uVar2)) {
        uVar1 = uVar2 & 0x80000001;
        if ((int)uVar1 < 0) {
          uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
        }
        if (((uVar1 != 0) && ((int)uVar2 < 0x21)) && (uVar2 = param_1[0x26], 0 < (int)uVar2)) {
          uVar1 = uVar2 & 0x80000001;
          if ((int)uVar1 < 0) {
            uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
          }
          if ((uVar1 != 0) && ((int)uVar2 < 0x21)) {
            if (0 < param_1[0x1e]) {
              uVar2 = param_1[0x1e] & 0x80000001;
              if ((int)uVar2 < 0) {
                uVar2 = (uVar2 - 1 | 0xfffffffe) + 1;
              }
              if (uVar2 != 0) {
                if (param_1[0x1f] < 1) {
                  return 0x100b;
                }
                if ((*(double *)(param_1 + 0x24) <= 0.0) || (*(double *)(param_1 + 0x2c) <= 0.0)) {
                  return 0x100c;
                }
                if (param_1[0x27] < 1) {
                  return 0x100d;
                }
                if (*(double *)(param_1 + 0x28) <= 0.0 && *(double *)(param_1 + 0x28) != 0.0) {
                  return 0x100e;
                }
                if (((param_1[0x2a] - 1U & 0xfffffffd) != 0) ||
                   ((iVar3 == 1 && (param_1[0x2a] == 3)))) {
                  return 0x100f;
                }
                uVar2 = param_1[0x2e] * 2 + 1;
                if (0 < (int)uVar2) {
                  uVar1 = uVar2 & 0x80000001;
                  if ((int)uVar1 < 0) {
                    uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
                  }
                  if (((uVar1 != 0) && ((int)uVar2 < 0x21)) &&
                     (uVar2 = param_1[0x33] * 2 + 1, 0 < (int)uVar2)) {
                    uVar1 = uVar2 & 0x80000001;
                    if ((int)uVar1 < 0) {
                      uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
                    }
                    if (((uVar1 != 0) && ((int)uVar2 < 0x21)) &&
                       (uVar2 = param_1[0x36] * 2 + 1, 0 < (int)uVar2)) {
                      uVar1 = uVar2 & 0x80000001;
                      if ((int)uVar1 < 0) {
                        uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
                      }
                      if (((uVar1 != 0) && ((int)uVar2 < 0x21)) &&
                         (uVar2 = param_1[0x3b] * 2 + 1, 0 < (int)uVar2)) {
                        uVar1 = uVar2 & 0x80000001;
                        if ((int)uVar1 < 0) {
                          uVar1 = (uVar1 - 1 | 0xfffffffe) + 1;
                        }
                        if ((uVar1 != 0) && ((int)uVar2 < 0x21)) {
                          if (((param_1[0x32] - 3U & 0xfffffffd) != 0) ||
                             ((param_1[0x3a] - 3U & 0xfffffffd) != 0)) {
                            return 0x1011;
                          }
                          iVar3 = 0;
                          lVar9 = 0;
                          iVar4 = iVar3;
                          if (3 < lVar6) {
                            pdVar7 = (double *)(param_1 + 0x16);
                            lVar8 = (lVar6 - 4U >> 2) + 1;
                            lVar9 = lVar8 * 4;
                            do {
                              if (0.0 < pdVar7[-4]) {
                                iVar3 = iVar3 + 1;
                              }
                              if (0.0 < *pdVar7) {
                                iVar4 = iVar4 + 1;
                              }
                              if (0.0 < pdVar7[-3]) {
                                iVar3 = iVar3 + 1;
                              }
                              if (0.0 < pdVar7[1]) {
                                iVar4 = iVar4 + 1;
                              }
                              if (0.0 < pdVar7[-2]) {
                                iVar3 = iVar3 + 1;
                              }
                              if (0.0 < pdVar7[2]) {
                                iVar4 = iVar4 + 1;
                              }
                              if (0.0 < pdVar7[-1]) {
                                iVar3 = iVar3 + 1;
                              }
                              if (0.0 < pdVar7[3]) {
                                iVar4 = iVar4 + 1;
                              }
                              pdVar7 = pdVar7 + 4;
                              lVar8 = lVar8 + -1;
                            } while (lVar8 != 0);
                          }
                          if (lVar9 < lVar6) {
                            lVar6 = lVar6 - lVar9;
                            pdVar7 = (double *)(param_1 + lVar9 * 2 + 0x16);
                            do {
                              if (0.0 < pdVar7[-4]) {
                                iVar3 = iVar3 + 1;
                              }
                              if (0.0 < *pdVar7) {
                                iVar4 = iVar4 + 1;
                              }
                              pdVar7 = pdVar7 + 1;
                              lVar6 = lVar6 + -1;
                            } while (lVar6 != 0);
                          }
                          if ((iVar3 != 0) && (iVar4 != 0)) {
                            return 0;
                          }
                          return 0x1012;
                        }
                      }
                    }
                  }
                }
                return 0x1010;
              }
            }
            return 0x100a;
          }
        }
      }
    }
    return 0x1009;
  }
  return 0x1006;
}


```

## procedure_pseudo_code {'procedure': '0x18002c790'}
Evidence: `ev_4991b583617a4c36a47d392183ccea90eaab8ac9b526c6758440e1c5d90255b3`
```c

void FUN_18002c790(longlong param_1,uint param_2)

{
  int iVar1;
  undefined4 *puVar2;
  int iVar3;
  undefined4 *puVar4;
  longlong lVar5;
  longlong lVar6;
  int iVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  longlong lVar12;
  longlong lVar13;
  ulonglong uVar14;
  longlong lVar15;
  int iVar16;
  longlong lVar17;
  
  if (0 < (int)param_2) {
    lVar15 = (longlong)(*(int **)(param_1 + 0x40))[1];
    iVar1 = **(int **)(param_1 + 0x40) + -1;
    iVar7 = 1;
    iVar16 = -1;
    uVar14 = (ulonglong)param_2;
    do {
      iVar3 = *(int *)(param_1 + 0x50);
      lVar12 = *(longlong *)(param_1 + 0x10);
      lVar17 = 0;
      lVar13 = iVar3 * iVar16 + lVar12;
      lVar10 = (iVar1 + iVar7) * iVar3 + lVar12;
      lVar11 = iVar3 * iVar7 + lVar12;
      lVar12 = (iVar1 + iVar16) * iVar3 + lVar12;
      if (3 < lVar15) {
        lVar5 = lVar11 - lVar13;
        lVar6 = lVar12 - lVar13;
        lVar9 = lVar10 - lVar13;
        lVar8 = (lVar15 - 4U >> 2) + 1;
        lVar17 = lVar8 * 4;
        puVar4 = (undefined4 *)(lVar13 + 4);
        do {
          puVar2 = puVar4 + 4;
          puVar4[-1] = *(undefined4 *)(lVar5 + -4 + (longlong)puVar4);
          *(undefined4 *)(lVar9 + -0x14 + (longlong)puVar2) =
               *(undefined4 *)(lVar6 + -0x14 + (longlong)puVar2);
          *puVar4 = *(undefined4 *)(lVar5 + -0x10 + (longlong)puVar2);
          *(undefined4 *)(lVar9 + -0x10 + (longlong)puVar2) =
               *(undefined4 *)(lVar6 + -0x10 + (longlong)puVar2);
          puVar4[1] = *(undefined4 *)(lVar5 + -0xc + (longlong)puVar2);
          *(undefined4 *)(lVar9 + -0xc + (longlong)puVar2) =
               *(undefined4 *)(lVar6 + -0xc + (longlong)puVar2);
          puVar4[2] = *(undefined4 *)(lVar5 + -8 + (longlong)puVar2);
          *(undefined4 *)(lVar9 + -8 + (longlong)puVar2) =
               *(undefined4 *)(lVar6 + -8 + (longlong)puVar2);
          lVar8 = lVar8 + -1;
          puVar4 = puVar2;
        } while (lVar8 != 0);
      }
      if (lVar17 < lVar15) {
        lVar5 = lVar15 - lVar17;
        puVar4 = (undefined4 *)(lVar13 + lVar17 * 4);
        do {
          puVar2 = puVar4 + 1;
          *puVar4 = *(undefined4 *)((lVar11 - lVar13) + (longlong)puVar4);
          *(undefined4 *)((lVar10 - lVar13) + -4 + (longlong)puVar2) =
               *(undefined4 *)((lVar12 - lVar13) + -4 + (longlong)puVar2);
          lVar5 = lVar5 + -1;
          puVar4 = puVar2;
        } while (lVar5 != 0);
      }
      iVar7 = iVar7 + 1;
      iVar16 = iVar16 + -1;
      uVar14 = uVar14 - 1;
    } while (uVar14 != 0);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180009190'}
Evidence: `ev_4a4631d095148a63a8fdef9217cd8dbd886696090f1ec46c267ba29025938056`
```c

void FUN_180009190(undefined8 param_1,char param_2,int param_3,int param_4,longlong param_5,
                  longlong param_6,int param_7,longlong param_8,longlong param_9,longlong param_10)

{
  if (param_2 != '\0') {
    FUN_18000f040(param_3,param_4,param_5,param_6,param_7,param_8,param_9,param_10);
    return;
  }
  FUN_18000f240(param_3,param_4,param_5,param_6,param_7,param_8,param_10);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18000ea90'}
Evidence: `ev_51d4c11856b97eaf0bcbf60ffaa43aa4c72643bedcd1e62ab43a559256fbf462`
```c

void FUN_18000ea90(int param_1,int param_2,longlong param_3,longlong param_4,int param_5,
                  longlong param_6,longlong param_7,longlong param_8)

{
  longlong lVar1;
  float fVar2;
  longlong lVar3;
  ulonglong uVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  
  if (0 < param_2) {
    lVar8 = (longlong)param_1;
    lVar5 = 0;
    do {
      lVar6 = 0;
      if (3 < lVar8) {
        lVar7 = param_5 * lVar5;
        do {
          lVar1 = lVar6 * 4;
          lVar3 = lVar6 * 2;
          uVar4 = (ulonglong)
                  *(ushort *)
                   (**(longlong **)(param_6 + 0x48) * lVar5 + lVar3 + *(longlong *)(param_6 + 0x10))
          ;
          fVar2 = *(float *)(**(longlong **)(param_7 + 0x48) * lVar5 + lVar1 +
                            *(longlong *)(param_7 + 0x10));
          *(float *)(*(int *)(param_8 + 0x50) * lVar5 + lVar1 + *(longlong *)(param_8 + 0x10)) =
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + uVar4 * 2) * 8) + lVar7
                               + lVar6) * (1.0 - fVar2) +
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + 2 + uVar4 * 2) * 8) +
                                lVar7 + lVar6) * fVar2;
          uVar4 = (ulonglong)
                  *(ushort *)
                   (**(longlong **)(param_6 + 0x48) * lVar5 + lVar3 + 2 +
                   *(longlong *)(param_6 + 0x10));
          fVar2 = *(float *)(**(longlong **)(param_7 + 0x48) * lVar5 + lVar1 + 4 +
                            *(longlong *)(param_7 + 0x10));
          *(float *)(*(int *)(param_8 + 0x50) * lVar5 + lVar1 + 4 + *(longlong *)(param_8 + 0x10)) =
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + uVar4 * 2) * 8) + lVar7
                                + 1 + lVar6) * (1.0 - fVar2) +
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + 2 + uVar4 * 2) * 8) +
                                lVar7 + 1 + lVar6) * fVar2;
          uVar4 = (ulonglong)
                  *(ushort *)
                   (**(longlong **)(param_6 + 0x48) * lVar5 + *(longlong *)(param_6 + 0x10) + 4 +
                   lVar3);
          fVar2 = *(float *)(**(longlong **)(param_7 + 0x48) * lVar5 + *(longlong *)(param_7 + 0x10)
                             + 8 + lVar1);
          lVar6 = lVar6 + 4;
          *(float *)(*(int *)(param_8 + 0x50) * lVar5 + *(longlong *)(param_8 + 0x10) + 8 + lVar1) =
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + uVar4 * 2) * 8) + lVar7
                                + -2 + lVar6) * (1.0 - fVar2) +
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + 2 + uVar4 * 2) * 8) +
                                lVar7 + -2 + lVar6) * fVar2;
          uVar4 = (ulonglong)
                  *(ushort *)
                   (**(longlong **)(param_6 + 0x48) * lVar5 + *(longlong *)(param_6 + 0x10) + 6 +
                   lVar3);
          fVar2 = *(float *)(**(longlong **)(param_7 + 0x48) * lVar5 + *(longlong *)(param_7 + 0x10)
                             + 0xc + lVar1);
          *(float *)(*(int *)(param_8 + 0x50) * lVar5 + *(longlong *)(param_8 + 0x10) + 0xc + lVar1)
               = (float)*(byte *)(*(longlong *)
                                   (param_4 + (ulonglong)*(ushort *)(param_3 + uVar4 * 2) * 8) +
                                  lVar7 + -1 + lVar6) * (1.0 - fVar2) +
                 (float)*(byte *)(*(longlong *)
                                   (param_4 + (ulonglong)*(ushort *)(param_3 + 2 + uVar4 * 2) * 8) +
                                  lVar7 + -1 + lVar6) * fVar2;
        } while (lVar6 < lVar8 + -3);
      }
      if (lVar6 < lVar8) {
        do {
          lVar1 = lVar6 * 4;
          lVar7 = lVar6 * 2;
          lVar6 = lVar6 + 1;
          uVar4 = (ulonglong)
                  *(ushort *)
                   (**(longlong **)(param_6 + 0x48) * lVar5 + lVar7 + *(longlong *)(param_6 + 0x10))
          ;
          fVar2 = *(float *)(**(longlong **)(param_7 + 0x48) * lVar5 + lVar1 +
                            *(longlong *)(param_7 + 0x10));
          *(float *)(*(int *)(param_8 + 0x50) * lVar5 + lVar1 + *(longlong *)(param_8 + 0x10)) =
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + uVar4 * 2) * 8) +
                                param_5 * lVar5 + -1 + lVar6) * (1.0 - fVar2) +
               (float)*(byte *)(*(longlong *)
                                 (param_4 + (ulonglong)*(ushort *)(param_3 + 2 + uVar4 * 2) * 8) +
                                param_5 * lVar5 + -1 + lVar6) * fVar2;
        } while (lVar6 < lVar8);
      }
      lVar5 = lVar5 + 1;
    } while (lVar5 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002ba10'}
Evidence: `ev_53e919d4fa467cbf62b1df7a8070dfa6f96c6e1953f4dd6e897fe1b2bfc62551`
```c

void FUN_18002ba10(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5)

{
  longlong lVar1;
  longlong lVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  float fVar6;
  
  if (0 < param_2) {
    lVar3 = (longlong)param_1;
    lVar5 = 0;
    do {
      lVar4 = 0;
      if (3 < lVar3) {
        do {
          lVar1 = lVar4 * 4;
          lVar2 = (int)param_4[1] * lVar5 + lVar1;
          lVar4 = lVar4 + 4;
          fVar6 = (float)*(byte *)((int)param_3[1] * lVar5 + *param_3 + -4 + lVar4);
          *(float *)(lVar2 + *param_4) = fVar6 + *(float *)(lVar2 + *param_4);
          *(float *)((int)param_5[1] * lVar5 + lVar1 + *param_5) = fVar6;
          lVar2 = (int)param_4[1] * lVar5 + lVar1;
          fVar6 = (float)*(byte *)((int)param_3[1] * lVar5 + *param_3 + -3 + lVar4);
          *(float *)(lVar2 + 4 + *param_4) = fVar6 + *(float *)(lVar2 + 4 + *param_4);
          *(float *)((int)param_5[1] * lVar5 + lVar1 + 4 + *param_5) = fVar6;
          lVar2 = (int)param_4[1] * lVar5 + lVar1 + 8;
          fVar6 = (float)*(byte *)((int)param_3[1] * lVar5 + *param_3 + -2 + lVar4);
          *(float *)(lVar2 + *param_4) = fVar6 + *(float *)(lVar2 + *param_4);
          *(float *)((int)param_5[1] * lVar5 + lVar1 + 8 + *param_5) = fVar6;
          lVar2 = (int)param_4[1] * lVar5 + lVar1 + 0xc;
          fVar6 = (float)*(byte *)((int)param_3[1] * lVar5 + *param_3 + -1 + lVar4);
          *(float *)(lVar2 + *param_4) = fVar6 + *(float *)(lVar2 + *param_4);
          *(float *)((int)param_5[1] * lVar5 + lVar1 + 0xc + *param_5) = fVar6;
        } while (lVar4 < lVar3 + -3);
      }
      while (lVar4 < lVar3) {
        lVar1 = lVar4 * 4;
        lVar2 = (int)param_4[1] * lVar5 + lVar1;
        lVar4 = lVar4 + 1;
        fVar6 = (float)*(byte *)((int)param_3[1] * lVar5 + *param_3 + -1 + lVar4);
        *(float *)(lVar2 + *param_4) = fVar6 + *(float *)(lVar2 + *param_4);
        *(float *)((int)param_5[1] * lVar5 + lVar1 + *param_5) = fVar6;
      }
      lVar5 = lVar5 + 1;
    } while (lVar5 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002bc20'}
Evidence: `ev_589297c08254f89977535a2c99d2f2a03e2641547d7efd64c84fa362a717babf`
```c

void FUN_18002bc20(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5)

{
  float fVar1;
  undefined2 uVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  int iVar9;
  longlong lVar10;
  longlong lVar11;
  float fVar12;
  float fVar13;
  byte bStack_97;
  longlong local_88;
  
  if (0 < param_2) {
    lVar10 = (longlong)param_1;
    iVar9 = 0;
    lVar7 = 0;
    do {
      lVar6 = 0;
      if (3 < lVar10) {
        lVar11 = 0;
        local_88 = 0;
        do {
          lVar4 = *param_4;
          lVar5 = (int)param_3[1] * lVar7 + lVar11;
          lVar8 = lVar6 * 4;
          uVar2 = *(undefined2 *)(lVar5 + *param_3);
          bStack_97 = (byte)((ushort)uVar2 >> 8);
          fVar12 = (float)(byte)uVar2;
          lVar3 = (int)param_4[1] * iVar9 + local_88;
          fVar13 = (float)*(byte *)(lVar5 + 2 + *param_3);
          fVar1 = *(float *)(lVar3 + 8 + lVar4);
          lVar5 = (int)param_4[1] * lVar7 + local_88;
          *(ulonglong *)(lVar5 + lVar4) =
               CONCAT44((float)bStack_97 + *(float *)(lVar3 + 4 + lVar4),
                        fVar12 + *(float *)(lVar3 + lVar4));
          *(float *)(lVar5 + 8 + lVar4) = fVar13 + fVar1;
          *(float *)((int)param_5[1] * lVar7 + lVar8 + *param_5) =
               fVar13 * 0.299 + (float)bStack_97 * 0.587 + fVar12 * 0.114;
          lVar4 = *param_4;
          lVar5 = (int)param_3[1] * lVar7 + lVar11;
          uVar2 = *(undefined2 *)(lVar5 + 3 + *param_3);
          bStack_97 = (byte)((ushort)uVar2 >> 8);
          fVar12 = (float)(byte)uVar2;
          lVar3 = (int)param_4[1] * iVar9 + local_88;
          fVar13 = (float)*(byte *)(lVar5 + 5 + *param_3);
          fVar1 = *(float *)(lVar3 + 0x14 + lVar4);
          lVar5 = (int)param_4[1] * lVar7 + local_88;
          *(ulonglong *)(lVar5 + 0xc + lVar4) =
               CONCAT44((float)bStack_97 + *(float *)(lVar3 + 0x10 + lVar4),
                        fVar12 + *(float *)(lVar3 + 0xc + lVar4));
          *(float *)(lVar5 + 0x14 + lVar4) = fVar13 + fVar1;
          *(float *)((int)param_5[1] * lVar7 + lVar8 + 4 + *param_5) =
               fVar13 * 0.299 + (float)bStack_97 * 0.587 + fVar12 * 0.114;
          lVar4 = (int)param_3[1] * lVar7 + *param_3;
          uVar2 = *(undefined2 *)(lVar4 + 6 + lVar11);
          bStack_97 = (byte)((ushort)uVar2 >> 8);
          fVar12 = (float)(byte)uVar2;
          lVar3 = (longlong)((int)param_4[1] * iVar9) + *param_4;
          fVar13 = (float)*(byte *)(lVar4 + 8 + lVar11);
          fVar1 = *(float *)(lVar3 + 0x20 + local_88);
          lVar4 = (int)param_4[1] * lVar7 + *param_4;
          *(ulonglong *)(lVar4 + 0x18 + local_88) =
               CONCAT44((float)bStack_97 + *(float *)(lVar3 + 0x1c + local_88),
                        fVar12 + *(float *)(lVar3 + 0x18 + local_88));
          *(float *)(lVar4 + 0x20 + local_88) = fVar13 + fVar1;
          *(float *)((int)param_5[1] * lVar7 + *param_5 + 8 + lVar8) =
               fVar13 * 0.299 + (float)bStack_97 * 0.587 + fVar12 * 0.114;
          lVar4 = (int)param_3[1] * lVar7 + *param_3;
          uVar2 = *(undefined2 *)(lVar4 + 9 + lVar11);
          bStack_97 = (byte)((ushort)uVar2 >> 8);
          fVar12 = (float)(byte)uVar2;
          lVar3 = (longlong)((int)param_4[1] * iVar9) + *param_4;
          fVar13 = (float)*(byte *)(lVar4 + 0xb + lVar11);
          fVar1 = *(float *)(lVar3 + 0x2c + local_88);
          lVar6 = lVar6 + 4;
          lVar11 = lVar11 + 0xc;
          lVar4 = (int)param_4[1] * lVar7 + *param_4;
          *(ulonglong *)(lVar4 + 0x24 + local_88) =
               CONCAT44((float)bStack_97 + *(float *)(lVar3 + 0x28 + local_88),
                        fVar12 + *(float *)(lVar3 + 0x24 + local_88));
          *(float *)(lVar4 + 0x2c + local_88) = fVar13 + fVar1;
          local_88 = local_88 + 0x30;
          *(float *)((int)param_5[1] * lVar7 + *param_5 + 0xc + lVar8) =
               fVar13 * 0.299 + (float)bStack_97 * 0.587 + fVar12 * 0.114;
        } while (lVar6 < lVar10 + -3);
      }
      if (lVar6 < lVar10) {
        lVar11 = lVar6 * 3;
        lVar8 = lVar6 * 0xc;
        do {
          lVar11 = lVar11 + 3;
          lVar8 = lVar8 + 0xc;
          lVar4 = (int)param_3[1] * lVar7 + *param_3;
          uVar2 = *(undefined2 *)(lVar4 + -3 + lVar11);
          bStack_97 = (byte)((ushort)uVar2 >> 8);
          fVar12 = (float)(byte)uVar2;
          lVar3 = (longlong)((int)param_4[1] * iVar9) + *param_4;
          fVar13 = (float)*(byte *)(lVar4 + -1 + lVar11);
          fVar1 = *(float *)(lVar3 + -4 + lVar8);
          lVar4 = (int)param_4[1] * lVar7 + *param_4;
          *(ulonglong *)(lVar4 + -0xc + lVar8) =
               CONCAT44((float)bStack_97 + *(float *)(lVar3 + -8 + lVar8),
                        fVar12 + *(float *)(lVar3 + -0xc + lVar8));
          *(float *)(lVar4 + -4 + lVar8) = fVar13 + fVar1;
          lVar4 = lVar6 * 4;
          lVar6 = lVar6 + 1;
          *(float *)((int)param_5[1] * lVar7 + lVar4 + *param_5) =
               fVar13 * 0.299 + (float)bStack_97 * 0.587 + fVar12 * 0.114;
        } while (lVar6 < lVar10);
      }
      iVar9 = iVar9 + 1;
      lVar7 = lVar7 + 1;
    } while (iVar9 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x1800243b0'}
Evidence: `ev_68d4fabaa3980c91c79ba7a39c2f32061a646b8a29c4c603fb4b27d67671f09d`
```c

void FUN_1800243b0(longlong param_1,longlong param_2,longlong param_3)

{
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  longlong local_18;
  undefined8 uStack_10;
  
  local_28 = *(longlong *)(param_3 + 0x10);
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_18 = *(longlong *)(param_2 + 0x10);
  uStack_20 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_2 + 0x50));
  local_38 = *(longlong *)(param_1 + 0x10);
  uStack_10 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_1 + 0x50));
  FUN_18002bc20((*(int **)(param_1 + 0x40))[1],**(int **)(param_1 + 0x40),&local_38,&local_18,
                &local_28);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180019740'}
Evidence: `ev_6b8f29ece4cc32358253e214367a7d11388e522cdd0dd01071c49c4c449478f4`
```c

void FUN_180019740(int param_1,int param_2,float param_3,longlong *param_4,longlong *param_5,
                  longlong *param_6)

{
  longlong lVar1;
  longlong lVar2;
  longlong lVar3;
  float fVar4;
  float fVar5;
  float fVar6;
  float fVar7;
  float fVar8;
  float fVar9;
  float fVar10;
  float fVar11;
  float fVar12;
  float fVar13;
  
  if (0 < param_2) {
    lVar2 = 0;
    do {
      lVar3 = 0;
      if (0 < (longlong)param_1) {
        do {
          lVar1 = lVar3 * 4;
          fVar4 = *(float *)((int)param_4[1] * lVar2 + lVar1 + *param_4);
          fVar5 = *(float *)((int)param_4[3] * lVar2 + param_4[2] + lVar1);
          fVar6 = *(float *)((int)param_4[5] * lVar2 + param_4[4] + lVar1);
          fVar13 = (*(float *)((int)param_5[1] * lVar2 + lVar1 + *param_5) - fVar4 * fVar4) +
                   param_3;
          fVar12 = *(float *)((int)param_5[3] * lVar2 + lVar1 + param_5[2]) - fVar5 * fVar4;
          fVar10 = *(float *)((int)param_5[5] * lVar2 + lVar1 + param_5[4]) - fVar6 * fVar4;
          fVar11 = (*(float *)((int)param_5[7] * lVar2 + param_5[6] + lVar1) - fVar5 * fVar5) +
                   param_3;
          fVar9 = *(float *)((int)param_5[9] * lVar2 + param_5[8] + lVar1) - fVar6 * fVar5;
          fVar8 = (*(float *)((int)param_5[0xb] * lVar2 + param_5[10] + lVar1) - fVar6 * fVar6) +
                  param_3;
          fVar4 = fVar8 * fVar11 - fVar9 * fVar9;
          fVar5 = fVar9 * fVar10 - fVar8 * fVar12;
          fVar6 = fVar9 * fVar12 - fVar11 * fVar10;
          lVar3 = lVar3 + 1;
          fVar7 = fVar5 * fVar12 + fVar4 * fVar13 + fVar6 * fVar10;
          *(float *)((int)param_6[1] * lVar2 + lVar1 + *param_6) = fVar4 / fVar7;
          *(float *)((int)param_6[3] * lVar2 + lVar1 + param_6[2]) = fVar5 / fVar7;
          *(float *)((int)param_6[5] * lVar2 + lVar1 + param_6[4]) = fVar6 / fVar7;
          *(float *)((int)param_6[7] * lVar2 + lVar1 + param_6[6]) =
               (fVar8 * fVar13 - fVar10 * fVar10) / fVar7;
          *(float *)((int)param_6[9] * lVar2 + lVar1 + param_6[8]) =
               (fVar10 * fVar12 - fVar9 * fVar13) / fVar7;
          *(float *)((int)param_6[0xb] * lVar2 + param_6[10] + lVar1) =
               (fVar11 * fVar13 - fVar12 * fVar12) / fVar7;
        } while (lVar3 < param_1);
      }
      lVar2 = lVar2 + 1;
    } while (lVar2 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x1800114f0'}
Evidence: `ev_6c3757447e9cd6f9a5e8f9083a2226c2b5e5173ca447b96d346c75cfe29eee72`
```c

void FUN_1800114f0(float *param_1,float *param_2,longlong param_3,undefined1 param_4)

{
  float *pfVar1;
  longlong lVar2;
  float *local_18;
  float *local_10;
  
  lVar2 = (longlong)param_2 - (longlong)param_1;
  do {
    lVar2 = lVar2 >> 3;
    if (lVar2 < 0x21) {
LAB_1800115bb:
      if (1 < lVar2) {
        FUN_180011c40(param_1,param_2);
      }
      return;
    }
    if (param_3 < 1) {
      if (0x20 < lVar2) {
        if (8 < (longlong)((longlong)param_2 - (longlong)param_1 & 0xfffffffffffffff8U)) {
          FUN_1800119d0((longlong)param_1,(longlong)param_2);
        }
        FUN_180011af0((int *)param_1,(longlong)param_2);
        return;
      }
      goto LAB_1800115bb;
    }
    FUN_180011630((ulonglong *)&local_18,param_1,param_2);
    pfVar1 = local_10;
    param_3 = param_3 / 2 + (param_3 / 2) / 2;
    if ((longlong)((longlong)local_18 - (longlong)param_1 & 0xfffffffffffffff8U) <
        (longlong)((longlong)param_2 - (longlong)local_10 & 0xfffffffffffffff8U)) {
      FUN_1800114f0(param_1,local_18,param_3,param_4);
      param_1 = pfVar1;
    }
    else {
      FUN_1800114f0(local_10,param_2,param_3,param_4);
      param_2 = local_18;
    }
    lVar2 = (longlong)param_2 - (longlong)param_1;
  } while( true );
}


```

## procedure_assembly {'procedure': '0x180020450'}
Evidence: `ev_709a90a816dabcb85491acf94339e6128eb6f629d9732f6e2e11ee3b4ae7b8ec`
```c
0x180020450: PUSH RBP
0x180020452: PUSH RBX
0x180020453: PUSH RSI
0x180020454: PUSH RDI
0x180020455: PUSH R12
0x180020457: PUSH R13
0x180020459: PUSH R14
0x18002045b: PUSH R15
0x18002045d: LEA RBP, [RSP + -0xc68]
0x180020465: SUB RSP, 0xd68
0x18002046c: MOV qword ptr [RBP + 0x770], -0x2
0x180020477: MOVAPS xmmword ptr [RSP + 0xd50], XMM6
0x18002047f: MOV RAX, qword ptr [0x18008b080]
0x180020486: XOR RAX, RSP
0x180020489: MOV qword ptr [RBP + 0xc48], RAX
0x180020490: MOV R14, R9
0x180020493: MOV qword ptr [RBP + 0x138], R9
0x18002049a: MOV qword ptr [RSP + 0x60], R8
0x18002049f: MOV qword ptr [RSP + 0x58], RDX
0x1800204a4: MOV R13, RCX
0x1800204a7: MOV RSI, qword ptr [RBP + 0xcd0]
0x1800204ae: MOV RDI, qword ptr [RBP + 0xcd8]
0x1800204b5: MOV qword ptr [RSP + 0x50], RDI
0x1800204ba: MOV EAX, dword ptr [RCX + 0x844]
0x1800204c0: MOV dword ptr [RBP + -0x80], EAX
0x1800204c3: MOV EAX, dword ptr [RCX + 0x80]
0x1800204c9: MOV dword ptr [RSP + 0x70], EAX
0x1800204cd: MOV EBX, dword ptr [RCX + 0x1c]
0x1800204d0: MOV EAX, dword ptr [RCX + 0x4e4]
0x1800204d6: MOV dword ptr [RSP + 0x44], EAX
0x1800204da: MOV EAX, dword ptr [RCX + 0x4e8]
0x1800204e0: MOV dword ptr [RSP + 0x40], EAX
0x1800204e4: MOV RAX, qword ptr [RCX + 0x848]
0x1800204eb: MOV qword ptr [RBP + 0x4c0], RAX
0x1800204f2: MOV RAX, qword ptr [RCX + 0xb00]
0x1800204f9: MOV qword ptr [RBP + 0xc40], RAX
0x180020500: MOV RAX, qword ptr [RCX + 0xb08]
0x180020507: MOV qword ptr [RBP + 0x4b0], RAX
0x18002050e: MOV RAX, qword ptr [RCX + 0xae8]
0x180020515: MOV qword ptr [RSP + 0x68], RAX
0x18002051a: MOV RAX, qword ptr [RCX + 0xaf0]
0x180020521: MOV qword ptr [RBP + 0x140], RAX
0x180020528: MOV RAX, qword ptr [RCX + 0xaf8]
0x18002052f: MOV qword ptr [RSP + 0x48], RAX
0x180020534: XOR ECX, ECX
0x180020536: MOV qword ptr [RBP + 0xc00], RCX
0x18002053d: MOV qword ptr [RBP + 0xc08], RCX
0x180020544: MOV qword ptr [RBP + 0xc10], RCX
0x18002054b: MOV qword ptr [RBP + 0xc18], RCX
0x180020552: MOV qword ptr [RBP + 0xc20], RCX
0x180020559: LEA RCX, [R13 + 0x808]
0x180020560: MOV qword ptr [RBP + 0xc00], RCX
0x180020567: MOV dword ptr [RBP + 0xc08], 0x20
0x180020571: XOR EDX, EDX
0x180020573: MOV qword ptr [RBP + 0xc10], RDX
0x18002057a: MOV RAX, qword ptr [RCX + 0x8]
0x18002057e: ADD RAX, qword ptr [RCX]
0x180020581: MOV qword ptr [RBP + 0xc18], RAX
0x180020588: LEA RDX, [R13 + 0x8]
0x18002058c: LEA R9, [RBP + 0x130]
0x180020593: LEA R8, [RSP + 0x78]
0x180020598: LEA RCX, [RBP + 0xc00]
0x18002059f: CALL 0x18002adc0
0x1800205a4: XORPS XMM0, XMM0
0x1800205a7: MOVUPS xmmword ptr [RBP + 0xa18], XMM0
0x1800205ae: XORPS XMM1, XMM1
0x1800205b1: MOVUPS xmmword ptr [RBP + 0xa28], XMM1
0x1800205b8: MOV dword ptr [RBP + 0x690], 0xc1020006
0x1800205c2: LEA RAX, [RBP + 0xa18]
0x1800205c9: MOV qword ptr [RBP + 0x698], RAX
0x1800205d0: MOV R15D, 0x1
0x1800205d6: MOV dword ptr [RBP + 0x6a0], R15D
0x1800205dd: MOV dword ptr [RBP + 0x6a4], 0x4
0x1800205e7: CALL qword ptr [0x180078288]
0x1800205ed: MOV R8, RAX
0x1800205f0: LEA RDX, [RBP + 0x690]
0x1800205f7: MOV RCX, RSI
0x1800205fa: CALL qword ptr [0x1800782a0]
0x180020600: XORPS XMM0, XMM0
0x180020603: MOVUPS xmmword ptr [RBP + 0x9f8], XMM0
0x18002060a: XORPS XMM1, XMM1
0x18002060d: MOVUPS xmmword ptr [RBP + 0xa08], XMM1
0x180020614: MOV dword ptr [RBP + 0x6c0], 0xc1020006
0x18002061e: LEA RAX, [RBP + 0x9f8]
0x180020625: MOV qword ptr [RBP + 0x6c8], RAX
0x18002062c: MOV dword ptr [RBP + 0x6d0], R15D
0x180020633: MOV dword ptr [RBP + 0x6d4], 0x4
0x18002063d: CALL qword ptr [0x180078288]
0x180020643: MOV R8, RAX
0x180020646: LEA RDX, [RBP + 0x6c0]
0x18002064d: MOV RCX, R14
0x180020650: CALL qword ptr [0x1800782a0]
0x180020656: MOV RDX, RDI
0x180020659: LEA RCX, [RBP + 0x910]
0x180020660: CALL 0x1800013f0
0x180020665: MOV EAX, dword ptr [RDI + 0x60]
0x180020668: MOV dword ptr [RBP + 0x970], EAX
0x18002066e: MOVD XMM0, dword ptr [R13 + 0x844]
0x180020677: CVTDQ2PS XMM0, XMM0
0x18002067a: MOVSS XMM3, dword ptr [0x18007c518]
0x180020682: DIVSS XMM3, XMM0
0x180020686: MOV RAX, qword ptr [RBP + 0x950]
0x18002068d: MOV EDX, dword ptr [RAX]
0x18002068f: MOV ECX, dword ptr [RAX + 0x4]
0x180020692: MOV RAX, qword ptr [RBP + 0x920]
0x180020699: MOV qword ptr [RBP + 0xc30], RAX
0x1800206a0: MOV EAX, dword ptr [RBP + 0x960]
0x1800206a6: MOV dword ptr [RBP + 0xc38], EAX
0x1800206ac: MOVAPS XMM0, xmmword ptr [RBP + 0xc30]
0x1800206b3: CMP EBX, R15D
0x1800206b6: JNZ 0x1800206ce
0x1800206b8: MOVDQA xmmword ptr [RBP + 0x760], XMM0
0x1800206c0: LEA R8, [RBP + 0x760]
0x1800206c7: CALL 0x180013c90
0x1800206cc: JMP 0x1800206e2
0x1800206ce: MOVDQA xmmword ptr [RBP + 0x780], XMM0
0x1800206d6: LEA R8, [RBP + 0x780]
0x1800206dd: CALL 0x180013d90
0x1800206e2: LEA RCX, [RBP + 0x910]
0x1800206e9: CALL 0x180001900
0x1800206ee: LEA RAX, [RBP + 0x960]
0x1800206f5: MOV RCX, qword ptr [RBP + 0x958]
0x1800206fc: CMP RCX, RAX
0x1800206ff: JZ 0x180020707
0x180020701: CALL qword ptr [0x1800783e0]
0x180020707: LEA R15, [R13 + 0x238]
0x18002070e: MOVSXD RAX, dword ptr [R15 + 0x10]
0x180020712: LEA RCX, [R15 + RAX*0x4]
0x180020716: XOR EAX, EAX
0x180020718: MOV qword ptr [RBP + 0xc30], RAX
0x18002071f: MOV R12, RCX
0x180020722: SUB R12, R15
0x180020725: ADD R12, 0x3
0x180020729: SHR R12, 0x2
0x18002072d: CMP R15, RCX
0x180020730: CMOVA R12, RAX
0x180020734: TEST R12, R12
0x180020737: JZ 0x180020860
0x18002073d: NOP dword ptr [RAX]
0x180020740: MOV qword ptr [RBP + 0x700], 0x0
0x18002074b: MOV dword ptr [RBP + 0x6f0], 0x2010000
0x180020755: MOV qword ptr [RBP + 0x6f8], RSI
0x18002075c: MOV qword ptr [RBP + 0x6b8], 0x0
0x180020767: MOV dword ptr [RBP + 0x6a8], 0x1010000
0x180020771: MOV qword ptr [RBP + 0x6b0], RSI
0x180020778: MOV qword ptr [RBP + 0x640], 0x0
0x180020783: MOV dword ptr [RBP + 0x630], 0x1010000
0x18002078d: MOVSXD RDI, dword ptr [R15]
0x180020790: IMUL RBX, RDI, 0x68
0x180020794: MOV RAX, qword ptr [RBP + 0xc40]
0x18002079b: ADD RAX, RBX
0x18002079e: MOV qword ptr [RBP + 0x638], RAX
0x1800207a5: LEA R9, [RBP + 0x6f0]
0x1800207ac: LEA R8, [RBP + 0x6a8]
0x1800207b3: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x40]
0x1800207ba: LEA RCX, [RBP + 0x630]
0x1800207c1: CALL qword ptr [0x180078300]
0x1800207c7: XOR EAX, EAX
0x1800207c9: MOV qword ptr [RBP + 0x598], RAX
0x1800207d0: MOV dword ptr [RBP + 0x588], 0x2010000
0x1800207da: MOV qword ptr [RBP + 0x590], R14
0x1800207e1: MOV qword ptr [RBP + 0x628], RAX
0x1800207e8: MOV dword ptr [RBP + 0x618], 0x1010000
0x1800207f2: MOV qword ptr [RBP + 0x620], R14
0x1800207f9: MOV qword ptr [RBP + 0x520], RAX
0x180020800: MOV dword ptr [RBP + 0x510], 0x1010000
0x18002080a: MOV RAX, qword ptr [RBP + 0x4b0]
0x180020811: ADD RAX, RBX
0x180020814: MOV qword ptr [RBP + 0x518], RAX
0x18002081b: LEA R9, [RBP + 0x588]
0x180020822: LEA R8, [RBP + 0x618]
0x180020829: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x40]
0x180020830: LEA RCX, [RBP + 0x510]
0x180020837: CALL qword ptr [0x180078300]
0x18002083d: LEA R15, [R15 + 0x4]
0x180020841: MOV RAX, qword ptr [RBP + 0xc30]
0x180020848: INC RAX
0x18002084b: MOV qword ptr [RBP + 0xc30], RAX
0x180020852: CMP RAX, R12
0x180020855: JNZ 0x180020740
0x18002085b: MOV RDI, qword ptr [RSP + 0x50]
0x180020860: OR R12D, 0xffffffff
0x180020864: XOR R15D, R15D
0x180020867: LEA EBX, [R12 + 0x2]
0x18002086c: CMP byte ptr [R13 + 0x124], BL
0x180020873: JNZ 0x180020d14
0x180020879: MOV EAX, dword ptr [R14]
0x18002087c: MOV dword ptr [RBP + 0x3f0], EAX
0x180020882: MOV EAX, dword ptr [R14 + 0x4]
0x180020886: MOV dword ptr [RBP + 0x3f4], EAX
0x18002088c: MOV EAX, dword ptr [R14 + 0x8]
0x180020890: MOV dword ptr [RBP + 0x3f8], EAX
0x180020896: MOV EAX, dword ptr [R14 + 0xc]
0x18002089a: MOV dword ptr [RBP + 0x3fc], EAX
0x1800208a0: MOV RAX, qword ptr [R14 + 0x10]
0x1800208a4: MOV qword ptr [RBP + 0x400], RAX
0x1800208ab: MOV RAX, qword ptr [R14 + 0x18]
0x1800208af: MOV qword ptr [RBP + 0x408], RAX
0x1800208b6: MOV RAX, qword ptr [R14 + 0x20]
0x1800208ba: MOV qword ptr [RBP + 0x410], RAX
0x1800208c1: MOV RAX, qword ptr [R14 + 0x28]
0x1800208c5: MOV qword ptr [RBP + 0x418], RAX
0x1800208cc: MOV RAX, qword ptr [R14 + 0x30]
0x1800208d0: MOV qword ptr [RBP + 0x420], RAX
0x1800208d7: MOV RCX, qword ptr [R14 + 0x38]
0x1800208db: MOV qword ptr [RBP + 0x428], RCX
0x1800208e2: LEA RAX, [RBP + 0x3f8]
0x1800208e9: MOV qword ptr [RBP + 0x430], RAX
0x1800208f0: LEA RDX, [RBP + 0x440]
0x1800208f7: MOV qword ptr [RBP + 0x438], RDX
0x1800208fe: MOV qword ptr [RBP + 0x448], R15
0x180020905: MOV qword ptr [RBP + 0x440], R15
0x18002090c: TEST RCX, RCX
0x18002090f: JZ 0x18002091f
0x180020911: MOV EAX, EBX
0x180020913: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020918: MOV RDX, qword ptr [RBP + 0x438]
0x18002091f: CMP dword ptr [R14 + 0x4], 0x2
0x180020924: JG 0x180020945
0x180020926: MOV RAX, qword ptr [R14 + 0x48]
0x18002092a: MOV RCX, qword ptr [RAX]
0x18002092d: MOV qword ptr [RDX], RCX
0x180020930: MOV RAX, qword ptr [R14 + 0x48]
0x180020934: MOV RCX, qword ptr [RAX + 0x8]
0x180020938: MOV RAX, qword ptr [RBP + 0x438]
0x18002093f: MOV qword ptr [RAX + 0x8], RCX
0x180020943: JMP 0x18002095d
0x180020945: MOV dword ptr [RBP + 0x3f4], R15D
0x18002094c: MOV RDX, R14
0x18002094f: LEA RCX, [RBP + 0x3f0]
0x180020956: CALL qword ptr [0x180078290]
0x18002095c: NOP
0x18002095d: MOV R8D, EBX
0x180020960: LEA RDX, [RBP + 0x3f0]
0x180020967: LEA RCX, [R13 + 0x8]
0x18002096b: CALL 0x180005740
0x180020970: NOP
0x180020971: MOV RCX, qword ptr [RBP + 0x428]
0x180020978: TEST RCX, RCX
0x18002097b: JZ 0x180020997
0x18002097d: MOV EAX, R12D
0x180020980: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020985: CMP EAX, 0x1
0x180020988: JNZ 0x180020997
0x18002098a: LEA RCX, [RBP + 0x3f0]
0x180020991: CALL qword ptr [0x180078400]
0x180020997: MOV qword ptr [RBP + 0x428], R15
0x18002099e: MOV qword ptr [RBP + 0x400], R15
0x1800209a5: MOV qword ptr [RBP + 0x418], R15
0x1800209ac: MOV qword ptr [RBP + 0x410], R15
0x1800209b3: MOV qword ptr [RBP + 0x408], R15
0x1800209ba: MOV EDX, R15D
0x1800209bd: CMP dword ptr [RBP + 0x3f4], EDX
0x1800209c3: JLE 0x1800209e9
0x1800209c5: MOV RCX, R15
0x1800209c8: NOP dword ptr [RAX + RAX*0x1]
0x1800209d0: MOV RAX, qword ptr [RBP + 0x430]
0x1800209d7: MOV dword ptr [RAX + RCX*0x1], R15D
0x1800209db: INC EDX
0x1800209dd: LEA RCX, [RCX + 0x4]
0x1800209e1: CMP EDX, dword ptr [RBP + 0x3f4]
0x1800209e7: JL 0x1800209d0
0x1800209e9: LEA RAX, [RBP + 0x440]
0x1800209f0: MOV RCX, qword ptr [RBP + 0x438]
0x1800209f7: CMP RCX, RAX
0x1800209fa: JZ 0x180020a02
0x1800209fc: CALL qword ptr [0x1800783e0]
0x180020a02: MOV EAX, dword ptr [RSI]
0x180020a04: MOV dword ptr [RBP + 0x270], EAX
0x180020a0a: MOV EAX, dword ptr [RSI + 0x4]
0x180020a0d: MOV dword ptr [RBP + 0x274], EAX
0x180020a13: MOV EAX, dword ptr [RSI + 0x8]
0x180020a16: MOV dword ptr [RBP + 0x278], EAX
0x180020a1c: MOV EAX, dword ptr [RSI + 0xc]
0x180020a1f: MOV dword ptr [RBP + 0x27c], EAX
0x180020a25: MOV RAX, qword ptr [RSI + 0x10]
0x180020a29: MOV qword ptr [RBP + 0x280], RAX
0x180020a30: MOV RAX, qword ptr [RSI + 0x18]
0x180020a34: MOV qword ptr [RBP + 0x288], RAX
0x180020a3b: MOV RAX, qword ptr [RSI + 0x20]
0x180020a3f: MOV qword ptr [RBP + 0x290], RAX
0x180020a46: MOV RAX, qword ptr [RSI + 0x28]
0x180020a4a: MOV qword ptr [RBP + 0x298], RAX
0x180020a51: MOV RAX, qword ptr [RSI + 0x30]
0x180020a55: MOV qword ptr [RBP + 0x2a0], RAX
0x180020a5c: MOV RCX, qword ptr [RSI + 0x38]
0x180020a60: MOV qword ptr [RBP + 0x2a8], RCX
0x180020a67: LEA RAX, [RBP + 0x278]
0x180020a6e: MOV qword ptr [RBP + 0x2b0], RAX
0x180020a75: LEA RDX, [RBP + 0x2c0]
0x180020a7c: MOV qword ptr [RBP + 0x2b8], RDX
0x180020a83: MOV qword ptr [RBP + 0x2c8], R15
0x180020a8a: MOV qword ptr [RBP + 0x2c0], R15
0x180020a91: TEST RCX, RCX
0x180020a94: JZ 0x180020aa4
0x180020a96: MOV EAX, EBX
0x180020a98: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020a9d: MOV RDX, qword ptr [RBP + 0x2b8]
0x180020aa4: CMP dword ptr [RSI + 0x4], 0x2
0x180020aa8: JG 0x180020ac9
0x180020aaa: MOV RAX, qword ptr [RSI + 0x48]
0x180020aae: MOV RCX, qword ptr [RAX]
0x180020ab1: MOV qword ptr [RDX], RCX
0x180020ab4: MOV RAX, qword ptr [RSI + 0x48]
0x180020ab8: MOV RCX, qword ptr [RAX + 0x8]
0x180020abc: MOV RAX, qword ptr [RBP + 0x2b8]
0x180020ac3: MOV qword ptr [RAX + 0x8], RCX
0x180020ac7: JMP 0x180020ae1
0x180020ac9: MOV dword ptr [RBP + 0x274], R15D
0x180020ad0: MOV RDX, RSI
0x180020ad3: LEA RCX, [RBP + 0x270]
0x180020ada: CALL qword ptr [0x180078290]
0x180020ae0: NOP
0x180020ae1: LEA RDX, [R13 + 0x840]
0x180020ae8: MOV R9D, 0x9
0x180020aee: LEA R8, [RBP + 0x270]
0x180020af5: LEA RSI, [R13 + 0x8]
0x180020af9: MOV RCX, RSI
0x180020afc: CALL 0x180004b70
0x180020b01: NOP
0x180020b02: MOV RCX, qword ptr [RBP + 0x2a8]
0x180020b09: TEST RCX, RCX
0x180020b0c: JZ 0x180020b28
0x180020b0e: MOV EAX, R12D
0x180020b11: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020b16: CMP EAX, 0x1
0x180020b19: JNZ 0x180020b28
0x180020b1b: LEA RCX, [RBP + 0x270]
0x180020b22: CALL qword ptr [0x180078400]
0x180020b28: MOV qword ptr [RBP + 0x2a8], R15
0x180020b2f: MOV qword ptr [RBP + 0x280], R15
0x180020b36: MOV qword ptr [RBP + 0x298], R15
0x180020b3d: MOV qword ptr [RBP + 0x290], R15
0x180020b44: MOV qword ptr [RBP + 0x288], R15
0x180020b4b: MOV EDX, R15D
0x180020b4e: CMP dword ptr [RBP + 0x274], EDX
0x180020b54: JLE 0x180020b79
0x180020b56: MOV RCX, R15
0x180020b59: NOP dword ptr [RAX]
0x180020b60: MOV RAX, qword ptr [RBP + 0x2b0]
0x180020b67: MOV dword ptr [RAX + RCX*0x1], R15D
0x180020b6b: INC EDX
0x180020b6d: LEA RCX, [RCX + 0x4]
0x180020b71: CMP EDX, dword ptr [RBP + 0x274]
0x180020b77: JL 0x180020b60
0x180020b79: LEA RAX, [RBP + 0x2c0]
0x180020b80: MOV RCX, qword ptr [RBP + 0x2b8]
0x180020b87: CMP RCX, RAX
0x180020b8a: JZ 0x180020b92
0x180020b8c: CALL qword ptr [0x1800783e0]
0x180020b92: MOV EAX, dword ptr [RDI]
0x180020b94: MOV dword ptr [RBP + 0x150], EAX
0x180020b9a: MOV EAX, dword ptr [RDI + 0x4]
0x180020b9d: MOV dword ptr [RBP + 0x154], EAX
0x180020ba3: MOV EAX, dword ptr [RDI + 0x8]
0x180020ba6: MOV dword ptr [RBP + 0x158], EAX
0x180020bac: MOV EAX, dword ptr [RDI + 0xc]
0x180020baf: MOV dword ptr [RBP + 0x15c], EAX
0x180020bb5: MOV RAX, qword ptr [RDI + 0x10]
0x180020bb9: MOV qword ptr [RBP + 0x160], RAX
0x180020bc0: MOV RAX, qword ptr [RDI + 0x18]
0x180020bc4: MOV qword ptr [RBP + 0x168], RAX
0x180020bcb: MOV RAX, qword ptr [RDI + 0x20]
0x180020bcf: MOV qword ptr [RBP + 0x170], RAX
0x180020bd6: MOV RAX, qword ptr [RDI + 0x28]
0x180020bda: MOV qword ptr [RBP + 0x178], RAX
0x180020be1: MOV RAX, qword ptr [RDI + 0x30]
0x180020be5: MOV qword ptr [RBP + 0x180], RAX
0x180020bec: MOV RCX, qword ptr [RDI + 0x38]
0x180020bf0: MOV qword ptr [RBP + 0x188], RCX
0x180020bf7: LEA RAX, [RBP + 0x158]
0x180020bfe: MOV qword ptr [RBP + 0x190], RAX
0x180020c05: LEA RDX, [RBP + 0x1a0]
0x180020c0c: MOV qword ptr [RBP + 0x198], RDX
0x180020c13: MOV qword ptr [RBP + 0x1a8], R15
0x180020c1a: MOV qword ptr [RBP + 0x1a0], R15
0x180020c21: TEST RCX, RCX
0x180020c24: JZ 0x180020c34
0x180020c26: MOV EAX, EBX
0x180020c28: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020c2d: MOV RDX, qword ptr [RBP + 0x198]
0x180020c34: CMP dword ptr [RDI + 0x4], 0x2
0x180020c38: JG 0x180020c59
0x180020c3a: MOV RAX, qword ptr [RDI + 0x48]
0x180020c3e: MOV RCX, qword ptr [RAX]
0x180020c41: MOV qword ptr [RDX], RCX
0x180020c44: MOV RAX, qword ptr [RDI + 0x48]
0x180020c48: MOV RCX, qword ptr [RAX + 0x8]
0x180020c4c: MOV RAX, qword ptr [RBP + 0x198]
0x180020c53: MOV qword ptr [RAX + 0x8], RCX
0x180020c57: JMP 0x180020c71
0x180020c59: MOV dword ptr [RBP + 0x154], R15D
0x180020c60: MOV RDX, RDI
0x180020c63: LEA RCX, [RBP + 0x150]
0x180020c6a: CALL qword ptr [0x180078290]
0x180020c70: NOP
0x180020c71: XOR R8D, R8D
0x180020c74: LEA RDX, [RBP + 0x150]
0x180020c7b: MOV RCX, RSI
0x180020c7e: CALL 0x180005740
0x180020c83: NOP
0x180020c84: MOV RCX, qword ptr [RBP + 0x188]
0x180020c8b: TEST RCX, RCX
0x180020c8e: JZ 0x180020caa
0x180020c90: MOV EAX, R12D
0x180020c93: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020c98: CMP EAX, 0x1
0x180020c9b: JNZ 0x180020caa
0x180020c9d: LEA RCX, [RBP + 0x150]
0x180020ca4: CALL qword ptr [0x180078400]
0x180020caa: MOV qword ptr [RBP + 0x188], R15
0x180020cb1: MOV qword ptr [RBP + 0x160], R15
0x180020cb8: MOV qword ptr [RBP + 0x178], R15
0x180020cbf: MOV qword ptr [RBP + 0x170], R15
0x180020cc6: MOV qword ptr [RBP + 0x168], R15
0x180020ccd: MOV EDX, R15D
0x180020cd0: CMP dword ptr [RBP + 0x154], EDX
0x180020cd6: JLE 0x180020cf9
0x180020cd8: MOV RCX, R15
0x180020cdb: NOP dword ptr [RAX + RAX*0x1]
0x180020ce0: MOV RAX, qword ptr [RBP + 0x190]
0x180020ce7: MOV dword ptr [RAX + RCX*0x1], R15D
0x180020ceb: INC EDX
0x180020ced: LEA RCX, [RCX + 0x4]
0x180020cf1: CMP EDX, dword ptr [RBP + 0x154]
0x180020cf7: JL 0x180020ce0
0x180020cf9: LEA RAX, [RBP + 0x1a0]
0x180020d00: MOV RCX, qword ptr [RBP + 0x198]
0x180020d07: CMP RCX, RAX
0x180020d0a: JZ 0x180020d18
0x180020d0c: CALL qword ptr [0x1800783e0]
0x180020d12: JMP 0x180020d18
0x180020d14: LEA RSI, [R13 + 0x8]
0x180020d18: CMP byte ptr [R13 + 0x124], 0x1
0x180020d20: JNZ 0x180021b53
0x180020d26: XOR EDX, EDX
0x180020d28: LEA R8D, [RDX + 0x68]
0x180020d2c: LEA RCX, [RBP + 0xa40]
0x180020d33: CALL 0x180072a62
0x180020d38: XORPS XMM6, XMM6
0x180020d3b: MOVSD qword ptr [RBP + 0x720], XMM6
0x180020d43: MOVUPS xmmword ptr [RBP + 0x730], XMM6
0x180020d4a: MOVSD qword ptr [RBP + 0x728], XMM6
0x180020d52: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x180020d5a: MOVDQU xmmword ptr [RBP + 0xd0], XMM0
0x180020d62: XORPS XMM1, XMM1
0x180020d65: MOVDQU xmmword ptr [RBP + 0xe0], XMM1
0x180020d6d: XORPS XMM0, XMM0
0x180020d70: MOVDQU xmmword ptr [RBP + 0xf0], XMM0
0x180020d78: MOVDQU xmmword ptr [RBP + 0x100], XMM1
0x180020d80: LEA RAX, [RBP + 0xd8]
0x180020d87: MOV qword ptr [RBP + 0x110], RAX
0x180020d8e: LEA RAX, [RBP + 0x120]
0x180020d95: MOV qword ptr [RBP + 0x118], RAX
0x180020d9c: MOV qword ptr [RBP + 0x128], R15
0x180020da3: MOV qword ptr [RBP + 0x120], R15
0x180020daa: MOV ESI, dword ptr [RSP + 0x40]
0x180020dae: MOV dword ptr [RBP + 0xc30], ESI
0x180020db4: MOV EDI, dword ptr [RSP + 0x44]
0x180020db8: MOV dword ptr [RBP + 0xc34], EDI
0x180020dbe: MOV R9D, 0x5
0x180020dc4: LEA R8, [RBP + 0xc30]
0x180020dcb: LEA EDX, [R9 + -0x3]
0x180020dcf: LEA RCX, [RBP + 0xd0]
0x180020dd6: CALL qword ptr [0x180078298]
0x180020ddc: LEA RDX, [RBP + 0x720]
0x180020de3: LEA RCX, [RBP + 0xd0]
0x180020dea: CALL qword ptr [0x1800782b8]
0x180020df0: LEA RAX, [RBP + 0xd0]
0x180020df7: MOV qword ptr [RSP + 0x50], RAX
0x180020dfc: MOV EAX, dword ptr [RBP + 0xd0]
0x180020e02: MOV dword ptr [RBP], EAX
0x180020e05: MOV EDX, dword ptr [RBP + 0xd4]
0x180020e0b: MOV dword ptr [RBP + 0x4], EDX
0x180020e0e: MOV EAX, dword ptr [RBP + 0xd8]
0x180020e14: MOV dword ptr [RBP + 0x8], EAX
0x180020e17: MOV EAX, dword ptr [RBP + 0xdc]
0x180020e1d: MOV dword ptr [RBP + 0xc], EAX
0x180020e20: MOV RAX, qword ptr [RBP + 0xe0]
0x180020e27: MOV qword ptr [RBP + 0x10], RAX
0x180020e2b: MOV RAX, qword ptr [RBP + 0xe8]
0x180020e32: MOV qword ptr [RBP + 0x18], RAX
0x180020e36: MOV RAX, qword ptr [RBP + 0xf0]
0x180020e3d: MOV qword ptr [RBP + 0x20], RAX
0x180020e41: MOV RAX, qword ptr [RBP + 0xf8]
0x180020e48: MOV qword ptr [RBP + 0x28], RAX
0x180020e4c: MOV RAX, qword ptr [RBP + 0x100]
0x180020e53: MOV qword ptr [RBP + 0x30], RAX
0x180020e57: MOV RCX, qword ptr [RBP + 0x108]
0x180020e5e: MOV qword ptr [RBP + 0x38], RCX
0x180020e62: LEA RAX, [RBP + 0x8]
0x180020e66: MOV qword ptr [RBP + 0x40], RAX
0x180020e6a: LEA R8, [RBP + 0x50]
0x180020e6e: MOV qword ptr [RBP + 0x48], R8
0x180020e72: MOV qword ptr [RBP + 0x58], R15
0x180020e76: MOV qword ptr [RBP + 0x50], R15
0x180020e7a: TEST RCX, RCX
0x180020e7d: JZ 0x180020e90
0x180020e7f: MOV EAX, EBX
0x180020e81: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020e86: MOV R8, qword ptr [RBP + 0x48]
0x180020e8a: MOV EDX, dword ptr [RBP + 0xd4]
0x180020e90: CMP EDX, 0x2
0x180020e93: JG 0x180020eb7
0x180020e95: MOV RAX, qword ptr [RBP + 0x118]
0x180020e9c: MOV RCX, qword ptr [RAX]
0x180020e9f: MOV qword ptr [R8], RCX
0x180020ea2: MOV RAX, qword ptr [RBP + 0x118]
0x180020ea9: MOV RCX, qword ptr [RAX + 0x8]
0x180020ead: MOV RAX, qword ptr [RBP + 0x48]
0x180020eb1: MOV qword ptr [RAX + 0x8], RCX
0x180020eb5: JMP 0x180020ecc
0x180020eb7: MOV dword ptr [RBP + 0x4], R15D
0x180020ebb: LEA RDX, [RBP + 0xd0]
0x180020ec2: LEA RCX, [RBP]
0x180020ec6: CALL qword ptr [0x180078290]
0x180020ecc: MOV dword ptr [RBP + 0x60], R15D
0x180020ed0: MOV RCX, qword ptr [RBP + 0x108]
0x180020ed7: TEST RCX, RCX
0x180020eda: JZ 0x180020ef6
0x180020edc: MOV EAX, R12D
0x180020edf: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020ee4: CMP EAX, 0x1
0x180020ee7: JNZ 0x180020ef6
0x180020ee9: LEA RCX, [RBP + 0xd0]
0x180020ef0: CALL qword ptr [0x180078400]
0x180020ef6: MOV qword ptr [RBP + 0x108], R15
0x180020efd: MOV qword ptr [RBP + 0xe0], R15
0x180020f04: MOV qword ptr [RBP + 0xf8], R15
0x180020f0b: MOV qword ptr [RBP + 0xf0], R15
0x180020f12: MOV qword ptr [RBP + 0xe8], R15
0x180020f19: MOV EDX, R15D
0x180020f1c: CMP dword ptr [RBP + 0xd4], 0x0
0x180020f23: JLE 0x180020f49
0x180020f25: MOV RCX, R15
0x180020f28: NOP dword ptr [RAX + RAX*0x1]
0x180020f30: MOV RAX, qword ptr [RBP + 0x110]
0x180020f37: MOV dword ptr [RAX + RCX*0x1], R15D
0x180020f3b: INC EDX
0x180020f3d: LEA RCX, [RCX + 0x4]
0x180020f41: CMP EDX, dword ptr [RBP + 0xd4]
0x180020f47: JL 0x180020f30
0x180020f49: LEA RAX, [RBP + 0x120]
0x180020f50: MOV RCX, qword ptr [RBP + 0x118]
0x180020f57: CMP RCX, RAX
0x180020f5a: JZ 0x180020f63
0x180020f5c: CALL qword ptr [0x1800783e0]
0x180020f62: NOP
0x180020f63: MOV EAX, dword ptr [RBP]
0x180020f66: MOV dword ptr [RBP + 0xa40], EAX
0x180020f6c: MOV EDX, dword ptr [RBP + 0x4]
0x180020f6f: MOV dword ptr [RBP + 0xa44], EDX
0x180020f75: MOV EAX, dword ptr [RBP + 0x8]
0x180020f78: MOV dword ptr [RBP + 0xa48], EAX
0x180020f7e: MOV EAX, dword ptr [RBP + 0xc]
0x180020f81: MOV dword ptr [RBP + 0xa4c], EAX
0x180020f87: MOV RAX, qword ptr [RBP + 0x10]
0x180020f8b: MOV qword ptr [RBP + 0xa50], RAX
0x180020f92: MOV RAX, qword ptr [RBP + 0x18]
0x180020f96: MOV qword ptr [RBP + 0xa58], RAX
0x180020f9d: MOV RAX, qword ptr [RBP + 0x20]
0x180020fa1: MOV qword ptr [RBP + 0xa60], RAX
0x180020fa8: MOV RAX, qword ptr [RBP + 0x28]
0x180020fac: MOV qword ptr [RBP + 0xa68], RAX
0x180020fb3: MOV RAX, qword ptr [RBP + 0x30]
0x180020fb7: MOV qword ptr [RBP + 0xa70], RAX
0x180020fbe: MOV RCX, qword ptr [RBP + 0x38]
0x180020fc2: MOV qword ptr [RBP + 0xa78], RCX
0x180020fc9: LEA RAX, [RBP + 0xa48]
0x180020fd0: MOV qword ptr [RBP + 0xa80], RAX
0x180020fd7: LEA R8, [RBP + 0xa90]
0x180020fde: MOV qword ptr [RBP + 0xa88], R8
0x180020fe5: MOV qword ptr [RBP + 0xa98], R15
0x180020fec: MOV qword ptr [RBP + 0xa90], R15
0x180020ff3: TEST RCX, RCX
0x180020ff6: JZ 0x180021009
0x180020ff8: MOV EAX, EBX
0x180020ffa: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180020fff: MOV R8, qword ptr [RBP + 0xa88]
0x180021006: MOV EDX, dword ptr [RBP + 0x4]
0x180021009: CMP EDX, 0x2
0x18002100c: JG 0x18002102d
0x18002100e: MOV RAX, qword ptr [RBP + 0x48]
0x180021012: MOV RCX, qword ptr [RAX]
0x180021015: MOV qword ptr [R8], RCX
0x180021018: MOV RAX, qword ptr [RBP + 0x48]
0x18002101c: MOV RCX, qword ptr [RAX + 0x8]
0x180021020: MOV RAX, qword ptr [RBP + 0xa88]
0x180021027: MOV qword ptr [RAX + 0x8], RCX
0x18002102b: JMP 0x180021045
0x18002102d: MOV dword ptr [RBP + 0xa44], R15D
0x180021034: LEA RDX, [RBP]
0x180021038: LEA RCX, [RBP + 0xa40]
0x18002103f: CALL qword ptr [0x180078290]
0x180021045: MOV EAX, dword ptr [RBP + 0x60]
0x180021048: MOV dword ptr [RBP + 0xaa0], EAX
0x18002104e: MOV RCX, qword ptr [RBP + 0x38]
0x180021052: TEST RCX, RCX
0x180021055: JZ 0x18002106e
0x180021057: MOV EAX, R12D
0x18002105a: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18002105f: CMP EAX, 0x1
0x180021062: JNZ 0x18002106e
0x180021064: LEA RCX, [RBP]
0x180021068: CALL qword ptr [0x180078400]
0x18002106e: MOV qword ptr [RBP + 0x38], R15
0x180021072: MOV qword ptr [RBP + 0x10], R15
0x180021076: MOV qword ptr [RBP + 0x28], R15
0x18002107a: MOV qword ptr [RBP + 0x20], R15
0x18002107e: MOV qword ptr [RBP + 0x18], R15
0x180021082: MOV EDX, R15D
0x180021085: CMP dword ptr [RBP + 0x4], 0x0
0x180021089: JLE 0x1800210a3
0x18002108b: MOV RCX, R15
0x18002108e: NOP
0x180021090: MOV RAX, qword ptr [RBP + 0x40]
0x180021094: MOV dword ptr [RAX + RCX*0x1], R15D
0x180021098: INC EDX
0x18002109a: LEA RCX, [RCX + 0x4]
0x18002109e: CMP EDX, dword ptr [RBP + 0x4]
0x1800210a1: JL 0x180021090
0x1800210a3: LEA RAX, [RBP + 0x50]
0x1800210a7: MOV RCX, qword ptr [RBP + 0x48]
0x1800210ab: CMP RCX, RAX
0x1800210ae: JZ 0x1800210b6
0x1800210b0: CALL qword ptr [0x1800783e0]
0x1800210b6: XOR EDX, EDX
0x1800210b8: LEA R8D, [RDX + 0x68]
0x1800210bc: LEA RCX, [RBP + 0xb90]
0x1800210c3: CALL 0x180072a62
0x1800210c8: MOVSD qword ptr [RBP + 0x740], XMM6
0x1800210d0: MOVUPS xmmword ptr [RBP + 0x750], XMM6
0x1800210d7: MOVSD qword ptr [RBP + 0x748], XMM6
0x1800210df: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x1800210e7: MOVDQU xmmword ptr [RBP + 0x70], XMM0
0x1800210ec: XORPS XMM1, XMM1
0x1800210ef: MOVDQU xmmword ptr [RBP + 0x80], XMM1
0x1800210f7: XORPS XMM0, XMM0
0x1800210fa: MOVDQU xmmword ptr [RBP + 0x90], XMM0
0x180021102: MOVDQU xmmword ptr [RBP + 0xa0], XMM1
0x18002110a: LEA RAX, [RBP + 0x78]
0x18002110e: MOV qword ptr [RBP + 0xb0], RAX
0x180021115: LEA RAX, [RBP + 0xc0]
0x18002111c: MOV qword ptr [RBP + 0xb8], RAX
0x180021123: MOV qword ptr [RBP + 0xc8], R15
0x18002112a: MOV qword ptr [RBP + 0xc0], R15
0x180021131: MOV dword ptr [RBP + 0xc40], ESI
0x180021137: MOV dword ptr [RBP + 0xc44], EDI
0x18002113d: MOV R9D, 0x5
0x180021143: LEA R8, [RBP + 0xc40]
0x18002114a: LEA EDX, [R9 + -0x3]
0x18002114e: LEA RCX, [RBP + 0x70]
0x180021152: CALL qword ptr [0x180078298]
0x180021158: LEA RDX, [RBP + 0x740]
0x18002115f: LEA RCX, [RBP + 0x70]
0x180021163: CALL qword ptr [0x1800782b8]
0x180021169: LEA RAX, [RBP + 0x70]
0x18002116d: MOV qword ptr [RSP + 0x50], RAX
0x180021172: MOV EAX, dword ptr [RBP + 0x70]
0x180021175: MOV dword ptr [RBP + -0x70], EAX
0x180021178: MOV EDX, dword ptr [RBP + 0x74]
0x18002117b: MOV dword ptr [RBP + -0x6c], EDX
0x18002117e: MOV EAX, dword ptr [RBP + 0x78]
0x180021181: MOV dword ptr [RBP + -0x68], EAX
0x180021184: MOV EAX, dword ptr [RBP + 0x7c]
0x180021187: MOV dword ptr [RBP + -0x64], EAX
0x18002118a: MOV RAX, qword ptr [RBP + 0x80]
0x180021191: MOV qword ptr [RBP + -0x60], RAX
0x180021195: MOV RAX, qword ptr [RBP + 0x88]
0x18002119c: MOV qword ptr [RBP + -0x58], RAX
0x1800211a0: MOV RAX, qword ptr [RBP + 0x90]
0x1800211a7: MOV qword ptr [RBP + -0x50], RAX
0x1800211ab: MOV RAX, qword ptr [RBP + 0x98]
0x1800211b2: MOV qword ptr [RBP + -0x48], RAX
0x1800211b6: MOV RAX, qword ptr [RBP + 0xa0]
0x1800211bd: MOV qword ptr [RBP + -0x40], RAX
0x1800211c1: MOV RCX, qword ptr [RBP + 0xa8]
0x1800211c8: MOV qword ptr [RBP + -0x38], RCX
0x1800211cc: LEA RAX, [RBP + -0x68]
0x1800211d0: MOV qword ptr [RBP + -0x30], RAX
0x1800211d4: LEA R8, [RBP + -0x20]
0x1800211d8: MOV qword ptr [RBP + -0x28], R8
0x1800211dc: MOV qword ptr [RBP + -0x18], R15
0x1800211e0: MOV qword ptr [RBP + -0x20], R15
0x1800211e4: TEST RCX, RCX
0x1800211e7: JZ 0x1800211f7
0x1800211e9: MOV EAX, EBX
0x1800211eb: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800211f0: MOV R8, qword ptr [RBP + -0x28]
0x1800211f4: MOV EDX, dword ptr [RBP + 0x74]
0x1800211f7: CMP EDX, 0x2
0x1800211fa: JG 0x18002121e
0x1800211fc: MOV RAX, qword ptr [RBP + 0xb8]
0x180021203: MOV RCX, qword ptr [RAX]
0x180021206: MOV qword ptr [R8], RCX
0x180021209: MOV RAX, qword ptr [RBP + 0xb8]
0x180021210: MOV RCX, qword ptr [RAX + 0x8]
0x180021214: MOV RAX, qword ptr [RBP + -0x28]
0x180021218: MOV qword ptr [RAX + 0x8], RCX
0x18002121c: JMP 0x180021230
0x18002121e: MOV dword ptr [RBP + -0x6c], R15D
0x180021222: LEA RDX, [RBP + 0x70]
0x180021226: LEA RCX, [RBP + -0x70]
0x18002122a: CALL qword ptr [0x180078290]
0x180021230: MOV dword ptr [RBP + -0x10], R15D
0x180021234: MOV RCX, qword ptr [RBP + 0xa8]
0x18002123b: TEST RCX, RCX
0x18002123e: JZ 0x180021257
0x180021240: MOV EAX, R12D
0x180021243: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180021248: CMP EAX, 0x1
0x18002124b: JNZ 0x180021257
0x18002124d: LEA RCX, [RBP + 0x70]
0x180021251: CALL qword ptr [0x180078400]
0x180021257: MOV qword ptr [RBP + 0xa8], R15
0x18002125e: MOV qword ptr [RBP + 0x80], R15
0x180021265: MOV qword ptr [RBP + 0x98], R15
0x18002126c: MOV qword ptr [RBP + 0x90], R15
0x180021273: MOV qword ptr [RBP + 0x88], R15
0x18002127a: MOV EDX, R15D
0x18002127d: CMP dword ptr [RBP + 0x74], 0x0
0x180021281: JLE 0x1800212a6
0x180021283: MOV RCX, R15
0x180021286: NOP dword ptr [RAX + RAX*0x1]
0x180021290: MOV RAX, qword ptr [RBP + 0xb0]
0x180021297: MOV dword ptr [RAX + RCX*0x1], R15D
0x18002129b: INC EDX
0x18002129d: LEA RCX, [RCX + 0x4]
0x1800212a1: CMP EDX, dword ptr [RBP + 0x74]
0x1800212a4: JL 0x180021290
0x1800212a6: LEA RAX, [RBP + 0xc0]
0x1800212ad: MOV RCX, qword ptr [RBP + 0xb8]
0x1800212b4: CMP RCX, RAX
0x1800212b7: JZ 0x1800212c0
0x1800212b9: CALL qword ptr [0x1800783e0]
0x1800212bf: NOP
0x1800212c0: MOV EAX, dword ptr [RBP + -0x70]
0x1800212c3: MOV dword ptr [RBP + 0xb90], EAX
0x1800212c9: MOV EDX, dword ptr [RBP + -0x6c]
0x1800212cc: MOV dword ptr [RBP + 0xb94], EDX
0x1800212d2: MOV EAX, dword ptr [RBP + -0x68]
0x1800212d5: MOV dword ptr [RBP + 0xb98], EAX
0x1800212db: MOV EAX, dword ptr [RBP + -0x64]
0x1800212de: MOV dword ptr [RBP + 0xb9c], EAX
0x1800212e4: MOV RAX, qword ptr [RBP + -0x60]
0x1800212e8: MOV qword ptr [RBP + 0xba0], RAX
0x1800212ef: MOV RAX, qword ptr [RBP + -0x58]
0x1800212f3: MOV qword ptr [RBP + 0xba8], RAX
0x1800212fa: MOV RAX, qword ptr [RBP + -0x50]
0x1800212fe: MOV qword ptr [RBP + 0xbb0], RAX
0x180021305: MOV RAX, qword ptr [RBP + -0x48]
0x180021309: MOV qword ptr [RBP + 0xbb8], RAX
0x180021310: MOV RAX, qword ptr [RBP + -0x40]
0x180021314: MOV qword ptr [RBP + 0xbc0], RAX
0x18002131b: MOV RCX, qword ptr [RBP + -0x38]
0x18002131f: MOV qword ptr [RBP + 0xbc8], RCX
0x180021326: LEA RAX, [RBP + 0xb98]
0x18002132d: MOV qword ptr [RBP + 0xbd0], RAX
0x180021334: LEA R8, [RBP + 0xbe0]
0x18002133b: MOV qword ptr [RBP + 0xbd8], R8
0x180021342: MOV qword ptr [RBP + 0xbe8], R15
0x180021349: MOV qword ptr [RBP + 0xbe0], R15
0x180021350: TEST RCX, RCX
0x180021353: JZ 0x180021366
0x180021355: MOV EAX, EBX
0x180021357: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18002135c: MOV R8, qword ptr [RBP + 0xbd8]
0x180021363: MOV EDX, dword ptr [RBP + -0x6c]
0x180021366: CMP EDX, 0x2
0x180021369: JG 0x18002138a
0x18002136b: MOV RAX, qword ptr [RBP + -0x28]
0x18002136f: MOV RCX, qword ptr [RAX]
0x180021372: MOV qword ptr [R8], RCX
0x180021375: MOV RAX, qword ptr [RBP + -0x28]
0x180021379: MOV RCX, qword ptr [RAX + 0x8]
0x18002137d: MOV RAX, qword ptr [RBP + 0xbd8]
0x180021384: MOV qword ptr [RAX + 0x8], RCX
0x180021388: JMP 0x1800213a2
0x18002138a: MOV dword ptr [RBP + 0xb94], R15D
0x180021391: LEA RDX, [RBP + -0x70]
0x180021395: LEA RCX, [RBP + 0xb90]
0x18002139c: CALL qword ptr [0x180078290]
0x1800213a2: MOV EAX, dword ptr [RBP + -0x10]
0x1800213a5: MOV dword ptr [RBP + 0xbf0], EAX
0x1800213ab: MOV RCX, qword ptr [RBP + -0x38]
0x1800213af: TEST RCX, RCX
0x1800213b2: JZ 0x1800213cb
0x1800213b4: MOV EAX, R12D
0x1800213b7: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800213bc: CMP EAX, 0x1
0x1800213bf: JNZ 0x1800213cb
0x1800213c1: LEA RCX, [RBP + -0x70]
0x1800213c5: CALL qword ptr [0x180078400]
0x1800213cb: MOV qword ptr [RBP + -0x38], R15
0x1800213cf: MOV qword ptr [RBP + -0x60], R15
0x1800213d3: MOV qword ptr [RBP + -0x48], R15
0x1800213d7: MOV qword ptr [RBP + -0x50], R15
0x1800213db: MOV qword ptr [RBP + -0x58], R15
0x1800213df: MOV EDX, R15D
0x1800213e2: CMP dword ptr [RBP + -0x6c], 0x0
0x1800213e6: JLE 0x180021403
0x1800213e8: MOV RCX, R15
0x1800213eb: NOP dword ptr [RAX + RAX*0x1]
0x1800213f0: MOV RAX, qword ptr [RBP + -0x30]
0x1800213f4: MOV dword ptr [RAX + RCX*0x1], R15D
0x1800213f8: INC EDX
0x1800213fa: LEA RCX, [RCX + 0x4]
0x1800213fe: CMP EDX, dword ptr [RBP + -0x6c]
0x180021401: JL 0x1800213f0
0x180021403: LEA RAX, [RBP + -0x20]
0x180021407: MOV RCX, qword ptr [RBP + -0x28]
0x18002140b: CMP RCX, RAX
0x18002140e: JZ 0x180021416
0x180021410: CALL qword ptr [0x1800783e0]
0x180021416: LEA RSI, [R13 + 0x24c]
0x18002141d: MOVSXD RAX, dword ptr [RSI + 0x10]
0x180021421: LEA RCX, [RSI + RAX*0x4]
0x180021425: MOV qword ptr [RSP + 0x50], R15
0x18002142a: MOV R15, RCX
0x18002142d: SUB R15, RSI
0x180021430: ADD R15, 0x3
0x180021434: SHR R15, 0x2
0x180021438: CMP RSI, RCX
0x18002143b: MOV EBX, 0x0
0x180021440: CMOVA R15, RBX
0x180021444: TEST R15, R15
0x180021447: JZ 0x180021594
0x18002144d: MOV R14, qword ptr [RSP + 0x50]
0x180021452: MOV R12, qword ptr [RSP + 0x68]
0x180021457: NOP word ptr [RAX + RAX*0x1]
0x180021460: MOV qword ptr [RBP + 0x538], 0x0
0x18002146b: MOV dword ptr [RBP + 0x528], 0x2010000
0x180021475: LEA RAX, [RBP + 0xa40]
0x18002147c: MOV qword ptr [RBP + 0x530], RAX
0x180021483: MOV qword ptr [RBP + 0x5c8], 0x0
0x18002148e: MOV dword ptr [RBP + 0x5b8], 0x1010000
0x180021498: LEA RAX, [RBP + 0xa40]
0x18002149f: MOV qword ptr [RBP + 0x5c0], RAX
0x1800214a6: MOV qword ptr [RBP + 0x5f8], 0x0
0x1800214b1: MOV dword ptr [RBP + 0x5e8], 0x1010000
0x1800214bb: MOVSXD RDI, dword ptr [RSI]
0x1800214be: IMUL RBX, RDI, 0x68
0x1800214c2: LEA RAX, [RBX + R12*0x1]
0x1800214c6: MOV qword ptr [RBP + 0x5f0], RAX
0x1800214cd: LEA R9, [RBP + 0x528]
0x1800214d4: LEA R8, [RBP + 0x5b8]
0x1800214db: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x60]
0x1800214e2: LEA RCX, [RBP + 0x5e8]
0x1800214e9: CALL qword ptr [0x180078300]
0x1800214ef: XOR EDX, EDX
0x1800214f1: MOV qword ptr [RBP + 0x580], RDX
0x1800214f8: MOV dword ptr [RBP + 0x570], 0x2010000
0x180021502: LEA RAX, [RBP + 0xb90]
0x180021509: MOV qword ptr [RBP + 0x578], RAX
0x180021510: MOV qword ptr [RBP + 0x658], RDX
0x180021517: MOV dword ptr [RBP + 0x648], 0x1010000
0x180021521: LEA RAX, [RBP + 0xb90]
0x180021528: MOV qword ptr [RBP + 0x650], RAX
0x18002152f: MOV qword ptr [RBP + 0x5b0], RDX
0x180021536: MOV dword ptr [RBP + 0x5a0], 0x1010000
0x180021540: MOV RAX, qword ptr [RBP + 0x140]
0x180021547: ADD RAX, RBX
0x18002154a: MOV qword ptr [RBP + 0x5a8], RAX
0x180021551: LEA R9, [RBP + 0x570]
0x180021558: LEA R8, [RBP + 0x648]
0x18002155f: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x60]
0x180021566: LEA RCX, [RBP + 0x5a0]
0x18002156d: CALL qword ptr [0x180078300]
0x180021573: LEA RSI, [RSI + 0x4]
0x180021577: INC R14
0x18002157a: CMP R14, R15
0x18002157d: JNZ 0x180021460
0x180021583: MOV R14, qword ptr [RBP + 0x138]
0x18002158a: OR R12D, 0xffffffff
0x18002158e: MOV EDI, dword ptr [RSP + 0x44]
0x180021592: XOR EBX, EBX
0x180021594: XOR EDX, EDX
0x180021596: LEA R8D, [RDX + 0x68]
0x18002159a: LEA RCX, [RBP + 0xab0]
0x1800215a1: CALL 0x180072a62
0x1800215a6: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x1800215ae: MOVDQA xmmword ptr [RBP + 0xab0], XMM0
0x1800215b6: XORPS XMM1, XMM1
0x1800215b9: MOVDQA xmmword ptr [RBP + 0xac0], XMM1
0x1800215c1: XORPS XMM0, XMM0
0x1800215c4: MOVDQA xmmword ptr [RBP + 0xad0], XMM0
0x1800215cc: MOVDQA xmmword ptr [RBP + 0xae0], XMM1
0x1800215d4: LEA RAX, [RBP + 0xab8]
0x1800215db: MOV qword ptr [RBP + 0xaf0], RAX
0x1800215e2: LEA RAX, [RBP + 0xb00]
0x1800215e9: MOV qword ptr [RBP + 0xaf8], RAX
0x1800215f0: MOV qword ptr [RBP + 0xb08], RBX
0x1800215f7: MOV qword ptr [RBP + 0xb00], RBX
0x1800215fe: MOV dword ptr [RBP + 0xb10], EBX
0x180021604: LEA RAX, [RBP + 0xab0]
0x18002160b: MOV qword ptr [RSP + 0x20], RAX
0x180021610: MOV R9D, dword ptr [RSP + 0x40]
0x180021615: MOV R8D, EDI
0x180021618: MOV RDX, qword ptr [RSP + 0x68]
0x18002161d: LEA RSI, [R13 + 0x8]
0x180021621: MOV RCX, RSI
0x180021624: CALL 0x180009ec0
0x180021629: MOV EAX, dword ptr [RBP + 0xab0]
0x18002162f: MOV dword ptr [RBP + 0x330], EAX
0x180021635: MOV EDX, dword ptr [RBP + 0xab4]
0x18002163b: MOV dword ptr [RBP + 0x334], EDX
0x180021641: MOV EAX, dword ptr [RBP + 0xab8]
0x180021647: MOV dword ptr [RBP + 0x338], EAX
0x18002164d: MOV EAX, dword ptr [RBP + 0xabc]
0x180021653: MOV dword ptr [RBP + 0x33c], EAX
0x180021659: MOV RAX, qword ptr [RBP + 0xac0]
0x180021660: MOV qword ptr [RBP + 0x340], RAX
0x180021667: MOV RAX, qword ptr [RBP + 0xac8]
0x18002166e: MOV qword ptr [RBP + 0x348], RAX
0x180021675: MOV RAX, qword ptr [RBP + 0xad0]
0x18002167c: MOV qword ptr [RBP + 0x350], RAX
0x180021683: MOV RAX, qword ptr [RBP + 0xad8]
0x18002168a: MOV qword ptr [RBP + 0x358], RAX
0x180021691: MOV RAX, qword ptr [RBP + 0xae0]
0x180021698: MOV qword ptr [RBP + 0x360], RAX
0x18002169f: MOV RCX, qword ptr [RBP + 0xae8]
0x1800216a6: MOV qword ptr [RBP + 0x368], RCX
0x1800216ad: LEA RAX, [RBP + 0x338]
0x1800216b4: MOV qword ptr [RBP + 0x370], RAX
0x1800216bb: LEA R8, [RBP + 0x380]
0x1800216c2: MOV qword ptr [RBP + 0x378], R8
0x1800216c9: XOR EDI, EDI
0x1800216cb: MOV qword ptr [RBP + 0x388], RDI
0x1800216d2: MOV qword ptr [RBP + 0x380], RDI
0x1800216d9: LEA R15D, [RDI + 0x1]
0x1800216dd: TEST RCX, RCX
0x1800216e0: JZ 0x1800216f7
0x1800216e2: MOV EAX, R15D
0x1800216e5: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800216ea: MOV EDX, dword ptr [RBP + 0xab4]
0x1800216f0: MOV R8, qword ptr [RBP + 0x378]
0x1800216f7: CMP EDX, 0x2
0x1800216fa: JG 0x180021721
0x1800216fc: MOV RAX, qword ptr [RBP + 0xaf8]
0x180021703: MOV RCX, qword ptr [RAX]
0x180021706: MOV qword ptr [R8], RCX
0x180021709: MOV RAX, qword ptr [RBP + 0xaf8]
0x180021710: MOV RCX, qword ptr [RAX + 0x8]
0x180021714: MOV RAX, qword ptr [RBP + 0x378]
0x18002171b: MOV qword ptr [RAX + 0x8], RCX
0x18002171f: JMP 0x18002173c
0x180021721: MOV dword ptr [RBP + 0x334], EDI
0x180021727: LEA RDX, [RBP + 0xab0]
0x18002172e: LEA RCX, [RBP + 0x330]
0x180021735: CALL qword ptr [0x180078290]
0x18002173b: NOP
0x18002173c: XOR R9D, R9D
0x18002173f: LEA R8, [RBP + 0x330]
0x180021746: LEA RDX, [R13 + 0x840]
0x18002174d: MOV RCX, RSI
0x180021750: CALL 0x180004b70
0x180021755: NOP
0x180021756: MOV RCX, qword ptr [RBP + 0x368]
0x18002175d: TEST RCX, RCX
0x180021760: JZ 0x18002177c
0x180021762: MOV EAX, R12D
0x180021765: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18002176a: CMP EAX, 0x1
0x18002176d: JNZ 0x18002177c
0x18002176f: LEA RCX, [RBP + 0x330]
0x180021776: CALL qword ptr [0x180078400]
0x18002177c: MOV qword ptr [RBP + 0x368], RDI
0x180021783: MOV qword ptr [RBP + 0x340], RDI
0x18002178a: MOV qword ptr [RBP + 0x358], RDI
0x180021791: MOV qword ptr [RBP + 0x350], RDI
0x180021798: MOV qword ptr [RBP + 0x348], RDI
0x18002179f: MOV EDX, EDI
0x1800217a1: CMP dword ptr [RBP + 0x334], EDX
0x1800217a7: JLE 0x1800217c8
0x1800217a9: MOV RCX, RDI
0x1800217ac: NOP dword ptr [RAX]
0x1800217b0: MOV RAX, qword ptr [RBP + 0x370]
0x1800217b7: MOV dword ptr [RAX + RCX*0x1], EDI
0x1800217ba: INC EDX
0x1800217bc: LEA RCX, [RCX + 0x4]
0x1800217c0: CMP EDX, dword ptr [RBP + 0x334]
0x1800217c6: JL 0x1800217b0
0x1800217c8: LEA RAX, [RBP + 0x380]
0x1800217cf: MOV RCX, qword ptr [RBP + 0x378]
0x1800217d6: CMP RCX, RAX
0x1800217d9: JZ 0x1800217e1
0x1800217db: CALL qword ptr [0x1800783e0]
0x1800217e1: MOV EAX, dword ptr [RBP + 0xa40]
0x1800217e7: MOV dword ptr [RBP + 0x1b0], EAX
0x1800217ed: MOV EDX, dword ptr [RBP + 0xa44]
0x1800217f3: MOV dword ptr [RBP + 0x1b4], EDX
0x1800217f9: MOV EAX, dword ptr [RBP + 0xa48]
0x1800217ff: MOV dword ptr [RBP + 0x1b8], EAX
0x180021805: MOV EAX, dword ptr [RBP + 0xa4c]
0x18002180b: MOV dword ptr [RBP + 0x1bc], EAX
0x180021811: MOV RAX, qword ptr [RBP + 0xa50]
0x180021818: MOV qword ptr [RBP + 0x1c0], RAX
0x18002181f: MOV RAX, qword ptr [RBP + 0xa58]
0x180021826: MOV qword ptr [RBP + 0x1c8], RAX
0x18002182d: MOV RAX, qword ptr [RBP + 0xa60]
0x180021834: MOV qword ptr [RBP + 0x1d0], RAX
0x18002183b: MOV RAX, qword ptr [RBP + 0xa68]
0x180021842: MOV qword ptr [RBP + 0x1d8], RAX
0x180021849: MOV RAX, qword ptr [RBP + 0xa70]
0x180021850: MOV qword ptr [RBP + 0x1e0], RAX
0x180021857: MOV RCX, qword ptr [RBP + 0xa78]
0x18002185e: MOV qword ptr [RBP + 0x1e8], RCX
0x180021865: LEA RAX, [RBP + 0x1b8]
0x18002186c: MOV qword ptr [RBP + 0x1f0], RAX
0x180021873: LEA R8, [RBP + 0x200]
0x18002187a: MOV qword ptr [RBP + 0x1f8], R8
0x180021881: MOV qword ptr [RBP + 0x208], RDI
0x180021888: MOV qword ptr [RBP + 0x200], RDI
0x18002188f: TEST RCX, RCX
0x180021892: JZ 0x1800218a9
0x180021894: MOV EAX, R15D
0x180021897: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18002189c: MOV EDX, dword ptr [RBP + 0xa44]
0x1800218a2: MOV R8, qword ptr [RBP + 0x1f8]
0x1800218a9: CMP EDX, 0x2
0x1800218ac: JG 0x1800218d3
0x1800218ae: MOV RAX, qword ptr [RBP + 0xa88]
0x1800218b5: MOV RCX, qword ptr [RAX]
0x1800218b8: MOV qword ptr [R8], RCX
0x1800218bb: MOV RAX, qword ptr [RBP + 0xa88]
0x1800218c2: MOV RCX, qword ptr [RAX + 0x8]
0x1800218c6: MOV RAX, qword ptr [RBP + 0x1f8]
0x1800218cd: MOV qword ptr [RAX + 0x8], RCX
0x1800218d1: JMP 0x1800218ee
0x1800218d3: MOV dword ptr [RBP + 0x1b4], EDI
0x1800218d9: LEA RDX, [RBP + 0xa40]
0x1800218e0: LEA RCX, [RBP + 0x1b0]
0x1800218e7: CALL qword ptr [0x180078290]
0x1800218ed: NOP
0x1800218ee: MOV R9D, 0x2
0x1800218f4: LEA R8, [RBP + 0x1b0]
0x1800218fb: LEA RDX, [R13 + 0x840]
0x180021902: MOV RCX, RSI
0x180021905: CALL 0x180004b70
0x18002190a: NOP
0x18002190b: MOV RCX, qword ptr [RBP + 0x1e8]
0x180021912: TEST RCX, RCX
0x180021915: JZ 0x180021931
0x180021917: MOV EAX, R12D
0x18002191a: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18002191f: CMP EAX, 0x1
0x180021922: JNZ 0x180021931
0x180021924: LEA RCX, [RBP + 0x1b0]
0x18002192b: CALL qword ptr [0x180078400]
0x180021931: MOV qword ptr [RBP + 0x1e8], RDI
0x180021938: MOV qword ptr [RBP + 0x1c0], RDI
0x18002193f: MOV qword ptr [RBP + 0x1d8], RDI
0x180021946: MOV qword ptr [RBP + 0x1d0], RDI
0x18002194d: MOV qword ptr [RBP + 0x1c8], RDI
0x180021954: MOV EDX, EDI
0x180021956: CMP dword ptr [RBP + 0x1b4], EDX
0x18002195c: JLE 0x180021988
0x18002195e: MOV RCX, RDI
0x180021961: NOP dword ptr [RAX]
0x180021965: NOP word ptr [RAX + RAX*0x1]
0x180021970: MOV RAX, qword ptr [RBP + 0x1f0]
0x180021977: MOV dword ptr [RAX + RCX*0x1], EDI
0x18002197a: INC EDX
0x18002197c: LEA RCX, [RCX + 0x4]
0x180021980: CMP EDX, dword ptr [RBP + 0x1b4]
0x180021986: JL 0x180021970
0x180021988: LEA RAX, [RBP + 0x200]
0x18002198f: MOV RCX, qword ptr [RBP + 0x1f8]
0x180021996: CMP RCX, RAX
0x180021999: JZ 0x1800219a2
0x18002199b: CALL qword ptr [0x1800783e0]
0x1800219a1: NOP
0x1800219a2: MOV RCX, qword ptr [RBP + 0xae8]
0x1800219a9: TEST RCX, RCX
0x1800219ac: JZ 0x1800219c8
0x1800219ae: MOV EAX, R12D
0x1800219b1: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800219b6: CMP EAX, 0x1
0x1800219b9: JNZ 0x1800219c8
0x1800219bb: LEA RCX, [RBP + 0xab0]
0x1800219c2: CALL qword ptr [0x180078400]
0x1800219c8: MOV qword ptr [RBP + 0xae8], RDI
0x1800219cf: MOV qword ptr [RBP + 0xac0], RDI
0x1800219d6: MOV qword ptr [RBP + 0xad8], RDI
0x1800219dd: MOV qword ptr [RBP + 0xad0], RDI
0x1800219e4: MOV qword ptr [RBP + 0xac8], RDI
0x1800219eb: MOV EDX, EDI
0x1800219ed: CMP dword ptr [RBP + 0xab4], EDX
0x1800219f3: JLE 0x180021a18
0x1800219f5: MOV RCX, RDI
0x1800219f8: NOP dword ptr [RAX + RAX*0x1]
0x180021a00: MOV RAX, qword ptr [RBP + 0xaf0]
0x180021a07: MOV dword ptr [RAX + RCX*0x1], EDI
0x180021a0a: INC EDX
0x180021a0c: LEA RCX, [RCX + 0x4]
0x180021a10: CMP EDX, dword ptr [RBP + 0xab4]
0x180021a16: JL 0x180021a00
0x180021a18: LEA RAX, [RBP + 0xb00]
0x180021a1f: MOV RCX, qword ptr [RBP + 0xaf8]
0x180021a26: CMP RCX, RAX
0x180021a29: JZ 0x180021a32
0x180021a2b: CALL qword ptr [0x1800783e0]
0x180021a31: NOP
0x180021a32: MOV RCX, qword ptr [RBP + 0xbc8]
0x180021a39: TEST RCX, RCX
0x180021a3c: JZ 0x180021a58
0x180021a3e: MOV EAX, R12D
0x180021a41: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180021a46: CMP EAX, 0x1
0x180021a49: JNZ 0x180021a58
0x180021a4b: LEA RCX, [RBP + 0xb90]
0x180021a52: CALL qword ptr [0x180078400]
0x180021a58: MOV qword ptr [RBP + 0xbc8], RDI
0x180021a5f: MOV qword ptr [RBP + 0xba0], RDI
0x180021a66: MOV qword ptr [RBP + 0xbb8], RDI
0x180021a6d: MOV qword ptr [RBP + 0xbb0], RDI
0x180021a74: MOV qword ptr [RBP + 0xba8], RDI
0x180021a7b: MOV EDX, EDI
0x180021a7d: CMP dword ptr [RBP + 0xb94], EDX
0x180021a83: JLE 0x180021aa8
0x180021a85: MOV RCX, RDI
0x180021a88: NOP dword ptr [RAX + RAX*0x1]
0x180021a90: MOV RAX, qword ptr [RBP + 0xbd0]
0x180021a97: MOV dword ptr [RAX + RCX*0x1], EDI
0x180021a9a: INC EDX
0x180021a9c: LEA RCX, [RCX + 0x4]
0x180021aa0: CMP EDX, dword ptr [RBP + 0xb94]
0x180021aa6: JL 0x180021a90
0x180021aa8: LEA RAX, [RBP + 0xbe0]
0x180021aaf: MOV RCX, qword ptr [RBP + 0xbd8]
0x180021ab6: CMP RCX, RAX
0x180021ab9: JZ 0x180021ac2
0x180021abb: CALL qword ptr [0x1800783e0]
0x180021ac1: NOP
0x180021ac2: MOV RCX, qword ptr [RBP + 0xa78]
0x180021ac9: TEST RCX, RCX
0x180021acc: JZ 0x180021ae8
0x180021ace: MOV EAX, R12D
0x180021ad1: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180021ad6: CMP EAX, 0x1
0x180021ad9: JNZ 0x180021ae8
0x180021adb: LEA RCX, [RBP + 0xa40]
0x180021ae2: CALL qword ptr [0x180078400]
0x180021ae8: MOV qword ptr [RBP + 0xa78], RDI
0x180021aef: MOV qword ptr [RBP + 0xa50], RDI
0x180021af6: MOV qword ptr [RBP + 0xa68], RDI
0x180021afd: MOV qword ptr [RBP + 0xa60], RDI
0x180021b04: MOV qword ptr [RBP + 0xa58], RDI
0x180021b0b: MOV EDX, EDI
0x180021b0d: CMP dword ptr [RBP + 0xa44], EDX
0x180021b13: JLE 0x180021b38
0x180021b15: MOV RCX, RDI
0x180021b18: NOP dword ptr [RAX + RAX*0x1]
0x180021b20: MOV RAX, qword ptr [RBP + 0xa80]
0x180021b27: MOV dword ptr [RAX + RCX*0x1], EDI
0x180021b2a: INC EDX
0x180021b2c: LEA RCX, [RCX + 0x4]
0x180021b30: CMP EDX, dword ptr [RBP + 0xa44]
0x180021b36: JL 0x180021b20
0x180021b38: LEA RAX, [RBP + 0xa90]
0x180021b3f: MOV RCX, qword ptr [RBP + 0xa88]
0x180021b46: CMP RCX, RAX
0x180021b49: JZ 0x180021b59
0x180021b4b: CALL qword ptr [0x1800783e0]
0x180021b51: JMP 0x180021b59
0x180021b53: XOR EDI, EDI
0x180021b55: LEA R15D, [RDI + 0x1]
0x180021b59: MOV EAX, dword ptr [RSP + 0x70]
0x180021b5d: CMP dword ptr [RBP + -0x80], EAX
0x180021b60: JL 0x18002257b
0x180021b66: CMP EAX, 0x2
0x180021b69: JL 0x18002257b
0x180021b6f: MOV RAX, qword ptr [R13 + 0xb18]
0x180021b76: MOV qword ptr [RSP + 0x20], RAX
0x180021b7b: MOV R9, qword ptr [R13 + 0xb10]
0x180021b82: MOV R8, qword ptr [R13 + 0xaf8]
0x180021b89: MOV RDX, qword ptr [R13 + 0x870]
0x180021b90: MOV RCX, RSI
0x180021b93: CALL 0x180017560
0x180021b98: LEA RDI, [R13 + 0x840]
0x180021b9f: MOV RAX, qword ptr [RBP + 0x130]
0x180021ba6: MOV qword ptr [RSP + 0x30], RAX
0x180021bab: MOV RBX, qword ptr [RSP + 0x78]
0x180021bb0: MOV qword ptr [RSP + 0x28], RBX
0x180021bb5: MOV RAX, qword ptr [RSP + 0x48]
0x180021bba: MOV qword ptr [RSP + 0x20], RAX
0x180021bbf: MOV R9, qword ptr [RSP + 0x68]
0x180021bc4: MOV R8, RDI
0x180021bc7: LEA RDX, [R13 + 0x808]
0x180021bce: MOV RCX, RSI
0x180021bd1: CALL 0x180014890
0x180021bd6: XORPS XMM0, XMM0
0x180021bd9: MOVUPS xmmword ptr [RBP + 0x978], XMM0
0x180021be0: XORPS XMM1, XMM1
0x180021be3: MOVUPS xmmword ptr [RBP + 0x988], XMM1
0x180021bea: MOV dword ptr [RBP + 0x4c8], 0xc1020006
0x180021bf4: LEA RAX, [RBP + 0x978]
0x180021bfb: MOV qword ptr [RBP + 0x4d0], RAX
0x180021c02: MOV dword ptr [RBP + 0x4d8], R15D
0x180021c09: MOV dword ptr [RBP + 0x4dc], 0x4
0x180021c13: CALL qword ptr [0x180078288]
0x180021c19: MOV R8, RAX
0x180021c1c: LEA RDX, [RBP + 0x4c8]
0x180021c23: MOV RCX, qword ptr [RSP + 0x58]
0x180021c28: CALL qword ptr [0x1800782a0]
0x180021c2e: XORPS XMM0, XMM0
0x180021c31: MOVUPS xmmword ptr [RBP + 0x9d8], XMM0
0x180021c38: XORPS XMM1, XMM1
0x180021c3b: MOVUPS xmmword ptr [RBP + 0x9e8], XMM1
0x180021c42: MOV dword ptr [RBP + 0x660], 0xc1020006
0x180021c4c: LEA RAX, [RBP + 0x9d8]
0x180021c53: MOV qword ptr [RBP + 0x668], RAX
0x180021c5a: MOV dword ptr [RBP + 0x670], R15D
0x180021c61: MOV dword ptr [RBP + 0x674], 0x4
0x180021c6b: CALL qword ptr [0x180078288]
0x180021c71: MOV R8, RAX
0x180021c74: LEA RDX, [RBP + 0x660]
0x180021c7b: MOV RCX, qword ptr [RSP + 0x60]
0x180021c80: CALL qword ptr [0x1800782a0]
0x180021c86: LEA RSI, [R13 + 0x24c]
0x180021c8d: MOVSXD RAX, dword ptr [RSI + 0x10]
0x180021c91: LEA RCX, [RSI + RAX*0x4]
0x180021c95: XOR EDX, EDX
0x180021c97: MOV R15D, EDX
0x180021c9a: MOV RAX, RCX
0x180021c9d: SUB RAX, RSI
0x180021ca0: ADD RAX, 0x3
0x180021ca4: SHR RAX, 0x2
0x180021ca8: CMP RSI, RCX
0x180021cab: CMOVA RAX, RDX
0x180021caf: TEST RAX, RAX
0x180021cb2: JZ 0x180021de8
0x180021cb8: MOV R14, RAX
0x180021cbb: MOV R12, RBX
0x180021cbe: NOP
0x180021cc0: MOV qword ptr [RBP + 0x5e0], 0x0
0x180021ccb: MOV dword ptr [RBP + 0x5d0], 0x2010000
0x180021cd5: MOV RAX, qword ptr [RSP + 0x58]
0x180021cda: MOV qword ptr [RBP + 0x5d8], RAX
0x180021ce1: MOV qword ptr [RBP + 0x688], 0x0
0x180021cec: MOV dword ptr [RBP + 0x678], 0x1010000
0x180021cf6: MOV qword ptr [RBP + 0x680], RAX
0x180021cfd: MOVSXD RDI, dword ptr [RSI]
0x180021d00: IMUL RBX, RDI, 0x68
0x180021d04: LEA RAX, [RBX + R12*0x1]
0x180021d08: MOV qword ptr [RBP + 0x610], 0x0
0x180021d13: MOV dword ptr [RBP + 0x600], 0x1010000
0x180021d1d: MOV qword ptr [RBP + 0x608], RAX
0x180021d24: LEA R9, [RBP + 0x5d0]
0x180021d2b: LEA R8, [RBP + 0x678]
0x180021d32: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x60]
0x180021d39: LEA RCX, [RBP + 0x600]
0x180021d40: CALL qword ptr [0x180078300]
0x180021d46: XOR ECX, ECX
0x180021d48: MOV qword ptr [RBP + 0x6e8], RCX
0x180021d4f: MOV dword ptr [RBP + 0x6d8], 0x2010000
0x180021d59: MOV RAX, qword ptr [RSP + 0x60]
0x180021d5e: MOV qword ptr [RBP + 0x6e0], RAX
0x180021d65: MOV qword ptr [RBP + 0x550], RCX
0x180021d6c: MOV dword ptr [RBP + 0x540], 0x1010000
0x180021d76: MOV qword ptr [RBP + 0x548], RAX
0x180021d7d: MOV RAX, qword ptr [RBP + 0x130]
0x180021d84: ADD RAX, RBX
0x180021d87: MOV qword ptr [RBP + 0x568], RCX
0x180021d8e: MOV dword ptr [RBP + 0x558], 0x1010000
0x180021d98: MOV qword ptr [RBP + 0x560], RAX
0x180021d9f: LEA R9, [RBP + 0x6d8]
0x180021da6: LEA R8, [RBP + 0x540]
0x180021dad: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x60]
0x180021db4: LEA RCX, [RBP + 0x558]
0x180021dbb: CALL qword ptr [0x180078300]
0x180021dc1: LEA RSI, [RSI + 0x4]
0x180021dc5: INC R15
0x180021dc8: CMP R15, R14
0x180021dcb: JNZ 0x180021cc0
0x180021dd1: MOV R14, qword ptr [RBP + 0x138]
0x180021dd8: OR R12D, 0xffffffff
0x180021ddc: MOV RBX, qword ptr [RSP + 0x78]
0x180021de1: LEA RDI, [R13 + 0x840]
0x180021de8: CMP byte ptr [R13 + 0x124], 0x1
0x180021df0: JNZ 0x180022914
0x180021df6: XOR EDX, EDX
0x180021df8: LEA R8D, [RDX + 0x68]
0x180021dfc: LEA RCX, [RBP + 0xb20]
0x180021e03: CALL 0x180072a62
0x180021e08: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x180021e10: MOVDQA xmmword ptr [RBP + 0xb20], XMM0
0x180021e18: XORPS XMM1, XMM1
0x180021e1b: MOVDQA xmmword ptr [RBP + 0xb30], XMM1
0x180021e23: XORPS XMM0, XMM0
0x180021e26: MOVDQA xmmword ptr [RBP + 0xb40], XMM0
0x180021e2e: MOVDQA xmmword ptr [RBP + 0xb50], XMM1
0x180021e36: LEA RAX, [RBP + 0xb28]
0x180021e3d: MOV qword ptr [RBP + 0xb60], RAX
0x180021e44: LEA RAX, [RBP + 0xb70]
0x180021e4b: MOV qword ptr [RBP + 0xb68], RAX
0x180021e52: XOR EDX, EDX
0x180021e54: MOV qword ptr [RBP + 0xb78], RDX
0x180021e5b: MOV qword ptr [RBP + 0xb70], RDX
0x180021e62: MOV dword ptr [RBP + 0xb80], EDX
0x180021e68: LEA RAX, [RBP + 0xb20]
0x180021e6f: MOV qword ptr [RSP + 0x20], RAX
0x180021e74: MOV R9D, dword ptr [RSP + 0x40]
0x180021e79: MOV R15D, dword ptr [RSP + 0x44]
0x180021e7e: MOV R8D, R15D
0x180021e81: MOV RDX, RBX
0x180021e84: LEA RSI, [R13 + 0x8]
0x180021e88: MOV RCX, RSI
0x180021e8b: CALL 0x180009ec0
0x180021e90: MOV EAX, dword ptr [RBP + 0xb20]
0x180021e96: MOV dword ptr [RBP + 0x210], EAX
0x180021e9c: MOV EDX, dword ptr [RBP + 0xb24]
0x180021ea2: MOV dword ptr [RBP + 0x214], EDX
0x180021ea8: MOV EAX, dword ptr [RBP + 0xb28]
0x180021eae: MOV dword ptr [RBP + 0x218], EAX
0x180021eb4: MOV EAX, dword ptr [RBP + 0xb2c]
0x180021eba: MOV dword ptr [RBP + 0x21c], EAX
0x180021ec0: MOV RAX, qword ptr [RBP + 0xb30]
0x180021ec7: MOV qword ptr [RBP + 0x220], RAX
0x180021ece: MOV RAX, qword ptr [RBP + 0xb38]
0x180021ed5: MOV qword ptr [RBP + 0x228], RAX
0x180021edc: MOV RAX, qword ptr [RBP + 0xb40]
0x180021ee3: MOV qword ptr [RBP + 0x230], RAX
0x180021eea: MOV RAX, qword ptr [RBP + 0xb48]
0x180021ef1: MOV qword ptr [RBP + 0x238], RAX
0x180021ef8: MOV RAX, qword ptr [RBP + 0xb50]
0x180021eff: MOV qword ptr [RBP + 0x240], RAX
0x180021f06: MOV RCX, qword ptr [RBP + 0xb58]
0x180021f0d: MOV qword ptr [RBP + 0x248], RCX
0x180021f14: LEA RAX, [RBP + 0x218]
0x180021f1b: MOV qword ptr [RBP + 0x250], RAX
0x180021f22: LEA R8, [RBP + 0x260]
0x180021f29: MOV qword ptr [RBP + 0x258], R8
0x180021f30: XOR R13D, R13D
0x180021f33: MOV qword ptr [RBP + 0x268], R13
0x180021f3a: MOV qword ptr [RBP + 0x260], R13
0x180021f41: LEA EBX, [R13 + 0x1]
0x180021f45: TEST RCX, RCX
0x180021f48: JZ 0x180021f5e
0x180021f4a: MOV EAX, EBX
0x180021f4c: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180021f51: MOV EDX, dword ptr [RBP + 0xb24]
0x180021f57: MOV R8, qword ptr [RBP + 0x258]
0x180021f5e: CMP EDX, 0x2
0x180021f61: JG 0x180021f88
0x180021f63: MOV RAX, qword ptr [RBP + 0xb68]
0x180021f6a: MOV RCX, qword ptr [RAX]
0x180021f6d: MOV qword ptr [R8], RCX
0x180021f70: MOV RAX, qword ptr [RBP + 0xb68]
0x180021f77: MOV RCX, qword ptr [RAX + 0x8]
0x180021f7b: MOV RAX, qword ptr [RBP + 0x258]
0x180021f82: MOV qword ptr [RAX + 0x8], RCX
0x180021f86: JMP 0x180021fa4
0x180021f88: MOV dword ptr [RBP + 0x214], R13D
0x180021f8f: LEA RDX, [RBP + 0xb20]
0x180021f96: LEA RCX, [RBP + 0x210]
0x180021f9d: CALL qword ptr [0x180078290]
0x180021fa3: NOP
0x180021fa4: MOV R9D, EBX
0x180021fa7: LEA R8, [RBP + 0x210]
0x180021fae: MOV RDX, RDI
0x180021fb1: MOV RCX, RSI
0x180021fb4: CALL 0x180004b70
0x180021fb9: NOP
0x180021fba: MOV RCX, qword ptr [RBP + 0x248]
0x180021fc1: TEST RCX, RCX
0x180021fc4: JZ 0x180021fe0
0x180021fc6: MOV EAX, R12D
0x180021fc9: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180021fce: CMP EAX, 0x1
0x180021fd1: JNZ 0x180021fe0
0x180021fd3: LEA RCX, [RBP + 0x210]
0x180021fda: CALL qword ptr [0x180078400]
0x180021fe0: MOV qword ptr [RBP + 0x248], R13
0x180021fe7: MOV qword ptr [RBP + 0x220], R13
0x180021fee: MOV qword ptr [RBP + 0x238], R13
0x180021ff5: MOV qword ptr [RBP + 0x230], R13
0x180021ffc: MOV qword ptr [RBP + 0x228], R13
0x180022003: MOV EDX, R13D
0x180022006: CMP dword ptr [RBP + 0x214], EDX
0x18002200c: JLE 0x18002202a
0x18002200e: MOV RCX, R13
0x180022011: MOV RAX, qword ptr [RBP + 0x250]
0x180022018: MOV dword ptr [RCX + RAX*0x1], R13D
0x18002201c: INC EDX
0x18002201e: LEA RCX, [RCX + 0x4]
0x180022022: CMP EDX, dword ptr [RBP + 0x214]
0x180022028: JL 0x180022011
0x18002202a: LEA RAX, [RBP + 0x260]
0x180022031: MOV RCX, qword ptr [RBP + 0x258]
0x180022038: CMP RCX, RAX
0x18002203b: JZ 0x180022043
0x18002203d: CALL qword ptr [0x1800783e0]
0x180022043: MOV R8, qword ptr [RSP + 0x58]
0x180022048: MOV EAX, dword ptr [R8]
0x18002204b: MOV dword ptr [RBP + 0x2d0], EAX
0x180022051: MOV EAX, dword ptr [R8 + 0x4]
0x180022055: MOV dword ptr [RBP + 0x2d4], EAX
0x18002205b: MOV EAX, dword ptr [R8 + 0x8]
0x18002205f: MOV dword ptr [RBP + 0x2d8], EAX
0x180022065: MOV EAX, dword ptr [R8 + 0xc]
0x180022069: MOV dword ptr [RBP + 0x2dc], EAX
0x18002206f: MOV RAX, qword ptr [R8 + 0x10]
0x180022073: MOV qword ptr [RBP + 0x2e0], RAX
0x18002207a: MOV RAX, qword ptr [R8 + 0x18]
0x18002207e: MOV qword ptr [RBP + 0x2e8], RAX
0x180022085: MOV RAX, qword ptr [R8 + 0x20]
0x180022089: MOV qword ptr [RBP + 0x2f0], RAX
0x180022090: MOV RAX, qword ptr [R8 + 0x28]
0x180022094: MOV qword ptr [RBP + 0x2f8], RAX
0x18002209b: MOV RAX, qword ptr [R8 + 0x30]
0x18002209f: MOV qword ptr [RBP + 0x300], RAX
0x1800220a6: MOV RCX, qword ptr [R8 + 0x38]
0x1800220aa: MOV qword ptr [RBP + 0x308], RCX
0x1800220b1: LEA RAX, [RBP + 0x2d8]
0x1800220b8: MOV qword ptr [RBP + 0x310], RAX
0x1800220bf: LEA RDX, [RBP + 0x320]
0x1800220c6: MOV qword ptr [RBP + 0x318], RDX
0x1800220cd: MOV qword ptr [RBP + 0x328], R13
0x1800220d4: MOV qword ptr [RBP + 0x320], R13
0x1800220db: TEST RCX, RCX
0x1800220de: JZ 0x1800220ee
0x1800220e0: MOV EAX, EBX
0x1800220e2: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800220e7: MOV RDX, qword ptr [RBP + 0x318]
0x1800220ee: CMP dword ptr [R8 + 0x4], 0x2
0x1800220f3: JG 0x180022114
0x1800220f5: MOV RAX, qword ptr [R8 + 0x48]
0x1800220f9: MOV RCX, qword ptr [RAX]
0x1800220fc: MOV qword ptr [RDX], RCX
0x1800220ff: MOV RAX, qword ptr [R8 + 0x48]
0x180022103: MOV RCX, qword ptr [RAX + 0x8]
0x180022107: MOV RAX, qword ptr [RBP + 0x318]
0x18002210e: MOV qword ptr [RAX + 0x8], RCX
0x180022112: JMP 0x18002212c
0x180022114: MOV dword ptr [RBP + 0x2d4], R13D
0x18002211b: MOV RDX, R8
0x18002211e: LEA RCX, [RBP + 0x2d0]
0x180022125: CALL qword ptr [0x180078290]
0x18002212b: NOP
0x18002212c: MOV R9D, 0x3
0x180022132: LEA R8, [RBP + 0x2d0]
0x180022139: MOV RDX, RDI
0x18002213c: MOV RCX, RSI
0x18002213f: CALL 0x180004b70
0x180022144: NOP
0x180022145: MOV RCX, qword ptr [RBP + 0x308]
0x18002214c: TEST RCX, RCX
0x18002214f: JZ 0x18002216b
0x180022151: MOV EAX, R12D
0x180022154: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180022159: CMP EAX, 0x1
0x18002215c: JNZ 0x18002216b
0x18002215e: LEA RCX, [RBP + 0x2d0]
0x180022165: CALL qword ptr [0x180078400]
0x18002216b: MOV qword ptr [RBP + 0x308], R13
0x180022172: MOV qword ptr [RBP + 0x2e0], R13
0x180022179: MOV qword ptr [RBP + 0x2f8], R13
0x180022180: MOV qword ptr [RBP + 0x2f0], R13
0x180022187: MOV qword ptr [RBP + 0x2e8], R13
0x18002218e: MOV EDX, R13D
0x180022191: CMP dword ptr [RBP + 0x2d4], EDX
0x180022197: JLE 0x1800221b9
0x180022199: MOV RCX, R13
0x18002219c: NOP dword ptr [RAX]
0x1800221a0: MOV RAX, qword ptr [RBP + 0x310]
0x1800221a7: MOV dword ptr [RCX + RAX*0x1], R13D
0x1800221ab: INC EDX
0x1800221ad: LEA RCX, [RCX + 0x4]
0x1800221b1: CMP EDX, dword ptr [RBP + 0x2d4]
0x1800221b7: JL 0x1800221a0
0x1800221b9: LEA RAX, [RBP + 0x320]
0x1800221c0: MOV RCX, qword ptr [RBP + 0x318]
0x1800221c7: CMP RCX, RAX
0x1800221ca: JZ 0x1800221d2
0x1800221cc: CALL qword ptr [0x1800783e0]
0x1800221d2: MOV R8, qword ptr [RSP + 0x60]
0x1800221d7: MOV EAX, dword ptr [R8]
0x1800221da: MOV dword ptr [RBP + 0x390], EAX
0x1800221e0: MOV EAX, dword ptr [R8 + 0x4]
0x1800221e4: MOV dword ptr [RBP + 0x394], EAX
0x1800221ea: MOV EAX, dword ptr [R8 + 0x8]
0x1800221ee: MOV dword ptr [RBP + 0x398], EAX
0x1800221f4: MOV EAX, dword ptr [R8 + 0xc]
0x1800221f8: MOV dword ptr [RBP + 0x39c], EAX
0x1800221fe: MOV RAX, qword ptr [R8 + 0x10]
0x180022202: MOV qword ptr [RBP + 0x3a0], RAX
0x180022209: MOV RAX, qword ptr [R8 + 0x18]
0x18002220d: MOV qword ptr [RBP + 0x3a8], RAX
0x180022214: MOV RAX, qword ptr [R8 + 0x20]
0x180022218: MOV qword ptr [RBP + 0x3b0], RAX
0x18002221f: MOV RAX, qword ptr [R8 + 0x28]
0x180022223: MOV qword ptr [RBP + 0x3b8], RAX
0x18002222a: MOV RAX, qword ptr [R8 + 0x30]
0x18002222e: MOV qword ptr [RBP + 0x3c0], RAX
0x180022235: MOV RCX, qword ptr [R8 + 0x38]
0x180022239: MOV qword ptr [RBP + 0x3c8], RCX
0x180022240: LEA RAX, [RBP + 0x398]
0x180022247: MOV qword ptr [RBP + 0x3d0], RAX
0x18002224e: LEA RDX, [RBP + 0x3e0]
0x180022255: MOV qword ptr [RBP + 0x3d8], RDX
0x18002225c: MOV qword ptr [RBP + 0x3e8], R13
0x180022263: MOV qword ptr [RBP + 0x3e0], R13
0x18002226a: TEST RCX, RCX
0x18002226d: JZ 0x18002227d
0x18002226f: MOV EAX, EBX
0x180022271: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180022276: MOV RDX, qword ptr [RBP + 0x3d8]
0x18002227d: CMP dword ptr [R8 + 0x4], 0x2
0x180022282: JG 0x1800222a3
0x180022284: MOV RAX, qword ptr [R8 + 0x48]
0x180022288: MOV RCX, qword ptr [RAX]
0x18002228b: MOV qword ptr [RDX], RCX
0x18002228e: MOV RAX, qword ptr [R8 + 0x48]
0x180022292: MOV RCX, qword ptr [RAX + 0x8]
0x180022296: MOV RAX, qword ptr [RBP + 0x3d8]
0x18002229d: MOV qword ptr [RAX + 0x8], RCX
0x1800222a1: JMP 0x1800222bb
0x1800222a3: MOV dword ptr [RBP + 0x394], R13D
0x1800222aa: MOV RDX, R8
0x1800222ad: LEA RCX, [RBP + 0x390]
0x1800222b4: CALL qword ptr [0x180078290]
0x1800222ba: NOP
0x1800222bb: MOV R9D, 0x4
0x1800222c1: LEA R8, [RBP + 0x390]
0x1800222c8: MOV RDX, RDI
0x1800222cb: MOV RCX, RSI
0x1800222ce: CALL 0x180004b70
0x1800222d3: NOP
0x1800222d4: MOV RCX, qword ptr [RBP + 0x3c8]
0x1800222db: TEST RCX, RCX
0x1800222de: JZ 0x1800222fa
0x1800222e0: MOV EAX, R12D
0x1800222e3: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800222e8: CMP EAX, 0x1
0x1800222eb: JNZ 0x1800222fa
0x1800222ed: LEA RCX, [RBP + 0x390]
0x1800222f4: CALL qword ptr [0x180078400]
0x1800222fa: MOV qword ptr [RBP + 0x3c8], R13
0x180022301: MOV qword ptr [RBP + 0x3a0], R13
0x180022308: MOV qword ptr [RBP + 0x3b8], R13
0x18002230f: MOV qword ptr [RBP + 0x3b0], R13
0x180022316: MOV qword ptr [RBP + 0x3a8], R13
0x18002231d: MOV EDX, R13D
0x180022320: CMP dword ptr [RBP + 0x394], EDX
0x180022326: JLE 0x180022349
0x180022328: MOV RCX, R13
0x18002232b: NOP dword ptr [RAX + RAX*0x1]
0x180022330: MOV RAX, qword ptr [RBP + 0x3d0]
0x180022337: MOV dword ptr [RCX + RAX*0x1], R13D
0x18002233b: INC EDX
0x18002233d: LEA RCX, [RCX + 0x4]
0x180022341: CMP EDX, dword ptr [RBP + 0x394]
0x180022347: JL 0x180022330
0x180022349: LEA RAX, [RBP + 0x3e0]
0x180022350: MOV RCX, qword ptr [RBP + 0x3d8]
0x180022357: CMP RCX, RAX
0x18002235a: JZ 0x180022362
0x18002235c: CALL qword ptr [0x1800783e0]
0x180022362: MOV EAX, dword ptr [R14]
0x180022365: MOV dword ptr [RBP + 0x450], EAX
0x18002236b: MOV EAX, dword ptr [R14 + 0x4]
0x18002236f: MOV dword ptr [RBP + 0x454], EAX
0x180022375: MOV EAX, dword ptr [R14 + 0x8]
0x180022379: MOV dword ptr [RBP + 0x458], EAX
0x18002237f: MOV EAX, dword ptr [R14 + 0xc]
0x180022383: MOV dword ptr [RBP + 0x45c], EAX
0x180022389: MOV RAX, qword ptr [R14 + 0x10]
0x18002238d: MOV qword ptr [RBP + 0x460], RAX
0x180022394: MOV RAX, qword ptr [R14 + 0x18]
0x180022398: MOV qword ptr [RBP + 0x468], RAX
0x18002239f: MOV RAX, qword ptr [R14 + 0x20]
0x1800223a3: MOV qword ptr [RBP + 0x470], RAX
0x1800223aa: MOV RAX, qword ptr [R14 + 0x28]
0x1800223ae: MOV qword ptr [RBP + 0x478], RAX
0x1800223b5: MOV RAX, qword ptr [R14 + 0x30]
0x1800223b9: MOV qword ptr [RBP + 0x480], RAX
0x1800223c0: MOV RAX, qword ptr [R14 + 0x38]
0x1800223c4: MOV qword ptr [RBP + 0x488], RAX
0x1800223cb: LEA RCX, [RBP + 0x458]
0x1800223d2: MOV qword ptr [RBP + 0x490], RCX
0x1800223d9: LEA RDX, [RBP + 0x4a0]
0x1800223e0: MOV qword ptr [RBP + 0x498], RDX
0x1800223e7: MOV qword ptr [RBP + 0x4a8], R13
0x1800223ee: MOV qword ptr [RBP + 0x4a0], R13
0x1800223f5: TEST RAX, RAX
0x1800223f8: JZ 0x180022406
0x1800223fa: XADD.LOCK dword ptr [RAX + 0x14], EBX
0x1800223ff: MOV RDX, qword ptr [RBP + 0x498]
0x180022406: CMP dword ptr [R14 + 0x4], 0x2
0x18002240b: JG 0x18002242c
0x18002240d: MOV RAX, qword ptr [R14 + 0x48]
0x180022411: MOV RCX, qword ptr [RAX]
0x180022414: MOV qword ptr [RDX], RCX
0x180022417: MOV RAX, qword ptr [R14 + 0x48]
0x18002241b: MOV RCX, qword ptr [RAX + 0x8]
0x18002241f: MOV RAX, qword ptr [RBP + 0x498]
0x180022426: MOV qword ptr [RAX + 0x8], RCX
0x18002242a: JMP 0x180022444
0x18002242c: MOV dword ptr [RBP + 0x454], R13D
0x180022433: MOV RDX, R14
0x180022436: LEA RCX, [RBP + 0x450]
0x18002243d: CALL qword ptr [0x180078290]
0x180022443: NOP
0x180022444: MOV R9D, dword ptr [RSP + 0x40]
0x180022449: MOV R8D, R15D
0x18002244c: LEA RDX, [RBP + 0x450]
0x180022453: MOV RCX, RSI
0x180022456: CALL 0x180005ed0
0x18002245b: NOP
0x18002245c: MOV RCX, qword ptr [RBP + 0x488]
0x180022463: TEST RCX, RCX
0x180022466: JZ 0x180022482
0x180022468: MOV EAX, R12D
0x18002246b: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180022470: CMP EAX, 0x1
0x180022473: JNZ 0x180022482
0x180022475: LEA RCX, [RBP + 0x450]
0x18002247c: CALL qword ptr [0x180078400]
0x180022482: MOV qword ptr [RBP + 0x488], R13
0x180022489: MOV qword ptr [RBP + 0x460], R13
0x180022490: MOV qword ptr [RBP + 0x478], R13
0x180022497: MOV qword ptr [RBP + 0x470], R13
0x18002249e: MOV qword ptr [RBP + 0x468], R13
0x1800224a5: MOV ECX, R13D
0x1800224a8: CMP dword ptr [RBP + 0x454], ECX
0x1800224ae: JLE 0x1800224cc
0x1800224b0: MOV RDX, R13
0x1800224b3: MOV RAX, qword ptr [RBP + 0x490]
0x1800224ba: MOV dword ptr [RDX + RAX*0x1], R13D
0x1800224be: INC ECX
0x1800224c0: LEA RDX, [RDX + 0x4]
0x1800224c4: CMP ECX, dword ptr [RBP + 0x454]
0x1800224ca: JL 0x1800224b3
0x1800224cc: LEA RAX, [RBP + 0x4a0]
0x1800224d3: MOV RCX, qword ptr [RBP + 0x498]
0x1800224da: CMP RCX, RAX
0x1800224dd: JZ 0x1800224e6
0x1800224df: CALL qword ptr [0x1800783e0]
0x1800224e5: NOP
0x1800224e6: MOV RAX, qword ptr [RBP + 0xb58]
0x1800224ed: TEST RAX, RAX
0x1800224f0: JZ 0x18002250b
0x1800224f2: XADD.LOCK dword ptr [RAX + 0x14], R12D
0x1800224f8: CMP R12D, 0x1
0x1800224fc: JNZ 0x18002250b
0x1800224fe: LEA RCX, [RBP + 0xb20]
0x180022505: CALL qword ptr [0x180078400]
0x18002250b: MOV qword ptr [RBP + 0xb58], R13
0x180022512: MOV qword ptr [RBP + 0xb30], R13
0x180022519: MOV qword ptr [RBP + 0xb48], R13
0x180022520: MOV qword ptr [RBP + 0xb40], R13
0x180022527: MOV qword ptr [RBP + 0xb38], R13
0x18002252e: MOV ECX, R13D
0x180022531: CMP dword ptr [RBP + 0xb24], ECX
0x180022537: JLE 0x180022559
0x180022539: MOV RDX, R13
0x18002253c: NOP dword ptr [RAX]
0x180022540: MOV RAX, qword ptr [RBP + 0xb60]
0x180022547: MOV dword ptr [RDX + RAX*0x1], R13D
0x18002254b: INC ECX
0x18002254d: LEA RDX, [RDX + 0x4]
0x180022551: CMP ECX, dword ptr [RBP + 0xb24]
0x180022557: JL 0x180022540
0x180022559: LEA RAX, [RBP + 0xb70]
0x180022560: MOV RCX, qword ptr [RBP + 0xb68]
0x180022567: CMP RCX, RAX
0x18002256a: JZ 0x180022914
0x180022570: CALL qword ptr [0x1800783e0]
0x180022576: JMP 0x180022914
0x18002257b: MOV dword ptr [RSP + 0x20], EDI
0x18002257f: XOR R9D, R9D
0x180022582: XOR R8D, R8D
0x180022585: XOR EDX, EDX
0x180022587: LEA RCX, [0x18007bf60]
0x18002258e: CALL 0x180001d80
0x180022593: XORPS XMM0, XMM0
0x180022596: MOVUPS xmmword ptr [RBP + 0x9b8], XMM0
0x18002259d: XORPS XMM1, XMM1
0x1800225a0: MOVUPS xmmword ptr [RBP + 0x9c8], XMM1
0x1800225a7: MOV dword ptr [RBP + 0x4f8], 0xc1020006
0x1800225b1: LEA RAX, [RBP + 0x9b8]
0x1800225b8: MOV qword ptr [RBP + 0x500], RAX
0x1800225bf: MOV dword ptr [RBP + 0x508], R15D
0x1800225c6: MOV dword ptr [RBP + 0x50c], 0x4
0x1800225d0: CALL qword ptr [0x180078288]
0x1800225d6: MOV R8, RAX
0x1800225d9: LEA RDX, [RBP + 0x4f8]
0x1800225e0: MOV R14, qword ptr [RSP + 0x58]
0x1800225e5: MOV RCX, R14
0x1800225e8: CALL qword ptr [0x1800782a0]
0x1800225ee: XORPS XMM0, XMM0
0x1800225f1: MOVUPS xmmword ptr [RBP + 0x998], XMM0
0x1800225f8: XORPS XMM1, XMM1
0x1800225fb: MOVUPS xmmword ptr [RBP + 0x9a8], XMM1
0x180022602: MOV dword ptr [RBP + 0x4e0], 0xc1020006
0x18002260c: LEA RAX, [RBP + 0x998]
0x180022613: MOV qword ptr [RBP + 0x4e8], RAX
0x18002261a: MOV dword ptr [RBP + 0x4f0], R15D
0x180022621: MOV dword ptr [RBP + 0x4f4], 0x4
0x18002262b: CALL qword ptr [0x180078288]
0x180022631: MOV R8, RAX
0x180022634: LEA RDX, [RBP + 0x4e0]
0x18002263b: MOV R15, qword ptr [RSP + 0x60]
0x180022640: MOV RCX, R15
0x180022643: CALL qword ptr [0x1800782a0]
0x180022649: LEA RSI, [R13 + 0x24c]
0x180022650: MOVSXD RAX, dword ptr [RSI + 0x10]
0x180022654: LEA RAX, [RSI + RAX*0x4]
0x180022658: MOV qword ptr [RSP + 0x48], RAX
0x18002265d: CMP RSI, RAX
0x180022660: JZ 0x1800228f0
0x180022666: NOP dword ptr [RAX + RAX*0x1]
0x180022670: MOVSXD RDI, dword ptr [RSI]
0x180022673: IMUL RBX, RDI, 0x68
0x180022677: MOV RCX, qword ptr [RSP + 0x68]
0x18002267c: MOV RAX, qword ptr [RBX + RCX*0x1 + 0x10]
0x180022681: MOV qword ptr [RBP + 0x4b0], RAX
0x180022688: MOV RAX, qword ptr [RBX + RCX*0x1 + 0x50]
0x18002268d: MOV dword ptr [RBP + 0x4b8], EAX
0x180022693: MOVAPS XMM0, xmmword ptr [RBP + 0x4b0]
0x18002269a: MOVDQA xmmword ptr [RBP + 0x7a0], XMM0
0x1800226a2: MOV RAX, qword ptr [R14 + 0x10]
0x1800226a6: MOV qword ptr [RBP + 0x710], RAX
0x1800226ad: MOV RAX, qword ptr [R14 + 0x50]
0x1800226b1: MOV dword ptr [RBP + 0x718], EAX
0x1800226b7: MOVAPS XMM0, xmmword ptr [RBP + 0x710]
0x1800226be: MOVDQA xmmword ptr [RBP + 0x790], XMM0
0x1800226c6: MOVSD XMM2, qword ptr [R13 + RDI*0x8 + 0x60]
0x1800226cd: CVTPD2PS XMM2, XMM2
0x1800226d1: LEA RAX, [RBP + 0x7a0]
0x1800226d8: MOV qword ptr [RSP + 0x28], RAX
0x1800226dd: LEA RAX, [RBP + 0x790]
0x1800226e4: MOV qword ptr [RSP + 0x20], RAX
0x1800226e9: MOV R9, qword ptr [RBP + 0x4c0]
0x1800226f0: MOV EDX, dword ptr [RSP + 0x40]
0x1800226f4: MOV ECX, dword ptr [RSP + 0x44]
0x1800226f8: CALL 0x180014f30
0x1800226fd: MOV R8, qword ptr [RBP + 0x140]
0x180022704: ADD R8, RBX
0x180022707: MOVSD XMM1, qword ptr [R13 + RDI*0x8 + 0x60]
0x18002270e: LEA RCX, [RBP + 0x7b0]
0x180022715: CALL qword ptr [0x180078310]
0x18002271b: NOP
0x18002271c: MOV RCX, qword ptr [RAX]
0x18002271f: MOV R9, qword ptr [RCX]
0x180022722: MOV R8, R15
0x180022725: MOV RDX, RAX
0x180022728: CALL qword ptr [R9 + 0x28]
0x18002272c: NOP
0x18002272d: MOV RCX, qword ptr [RBP + 0x8b8]
0x180022734: TEST RCX, RCX
0x180022737: JZ 0x180022753
0x180022739: MOV EAX, R12D
0x18002273c: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180022741: CMP EAX, 0x1
0x180022744: JNZ 0x180022753
0x180022746: LEA RCX, [RBP + 0x880]
0x18002274d: CALL qword ptr [0x180078400]
0x180022753: XOR EBX, EBX
0x180022755: MOV qword ptr [RBP + 0x8b8], RBX
0x18002275c: MOV qword ptr [RBP + 0x890], RBX
0x180022763: MOV qword ptr [RBP + 0x8a8], RBX
0x18002276a: MOV qword ptr [RBP + 0x8a0], RBX
0x180022771: MOV qword ptr [RBP + 0x898], RBX
0x180022778: MOV EDX, EBX
0x18002277a: CMP dword ptr [RBP + 0x884], EBX
0x180022780: JLE 0x1800227a8
0x180022782: MOV ECX, EBX
0x180022784: NOP dword ptr [RAX]
0x180022788: NOP dword ptr [RAX + RAX*0x1]
0x180022790: MOV RAX, qword ptr [RBP + 0x8c0]
0x180022797: MOV dword ptr [RAX + RCX*0x1], EBX
0x18002279a: INC EDX
0x18002279c: LEA RCX, [RCX + 0x4]
0x1800227a0: CMP EDX, dword ptr [RBP + 0x884]
0x1800227a6: JL 0x180022790
0x1800227a8: LEA RAX, [RBP + 0x8d0]
0x1800227af: MOV RCX, qword ptr [RBP + 0x8c8]
0x1800227b6: CMP RCX, RAX
0x1800227b9: JZ 0x1800227c2
0x1800227bb: CALL qword ptr [0x1800783e0]
0x1800227c1: NOP
0x1800227c2: MOV RCX, qword ptr [RBP + 0x858]
0x1800227c9: TEST RCX, RCX
0x1800227cc: JZ 0x1800227e8
0x1800227ce: MOV EAX, R12D
0x1800227d1: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800227d6: CMP EAX, 0x1
0x1800227d9: JNZ 0x1800227e8
0x1800227db: LEA RCX, [RBP + 0x820]
0x1800227e2: CALL qword ptr [0x180078400]
0x1800227e8: MOV qword ptr [RBP + 0x858], RBX
0x1800227ef: MOV qword ptr [RBP + 0x830], RBX
0x1800227f6: MOV qword ptr [RBP + 0x848], RBX
0x1800227fd: MOV qword ptr [RBP + 0x840], RBX
0x180022804: MOV qword ptr [RBP + 0x838], RBX
0x18002280b: MOV EDX, EBX
0x18002280d: CMP dword ptr [RBP + 0x824], EBX
0x180022813: JLE 0x180022838
0x180022815: MOV RCX, RBX
0x180022818: NOP dword ptr [RAX + RAX*0x1]
0x180022820: MOV RAX, qword ptr [RBP + 0x860]
0x180022827: MOV dword ptr [RAX + RCX*0x1], EBX
0x18002282a: INC EDX
0x18002282c: LEA RCX, [RCX + 0x4]
0x180022830: CMP EDX, dword ptr [RBP + 0x824]
0x180022836: JL 0x180022820
0x180022838: LEA RAX, [RBP + 0x870]
0x18002283f: MOV RCX, qword ptr [RBP + 0x868]
0x180022846: CMP RCX, RAX
0x180022849: JZ 0x180022852
0x18002284b: CALL qword ptr [0x1800783e0]
0x180022851: NOP
0x180022852: MOV RCX, qword ptr [RBP + 0x7f8]
0x180022859: TEST RCX, RCX
0x18002285c: JZ 0x180022878
0x18002285e: MOV EAX, R12D
0x180022861: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180022866: CMP EAX, 0x1
0x180022869: JNZ 0x180022878
0x18002286b: LEA RCX, [RBP + 0x7c0]
0x180022872: CALL qword ptr [0x180078400]
0x180022878: MOV qword ptr [RBP + 0x7f8], RBX
0x18002287f: MOV qword ptr [RBP + 0x7d0], RBX
0x180022886: MOV qword ptr [RBP + 0x7e8], RBX
0x18002288d: MOV qword ptr [RBP + 0x7e0], RBX
0x180022894: MOV qword ptr [RBP + 0x7d8], RBX
0x18002289b: MOV EDX, EBX
0x18002289d: CMP dword ptr [RBP + 0x7c4], EBX
0x1800228a3: JLE 0x1800228c8
0x1800228a5: MOV RCX, RBX
0x1800228a8: NOP dword ptr [RAX + RAX*0x1]
0x1800228b0: MOV RAX, qword ptr [RBP + 0x800]
0x1800228b7: MOV dword ptr [RAX + RCX*0x1], EBX
0x1800228ba: INC EDX
0x1800228bc: LEA RCX, [RCX + 0x4]
0x1800228c0: CMP EDX, dword ptr [RBP + 0x7c4]
0x1800228c6: JL 0x1800228b0
0x1800228c8: LEA RAX, [RBP + 0x810]
0x1800228cf: MOV RCX, qword ptr [RBP + 0x808]
0x1800228d6: CMP RCX, RAX
0x1800228d9: JZ 0x1800228e1
0x1800228db: CALL qword ptr [0x1800783e0]
0x1800228e1: ADD RSI, 0x4
0x1800228e5: CMP RSI, qword ptr [RSP + 0x48]
0x1800228ea: JNZ 0x180022670
0x1800228f0: CMP byte ptr [R13 + 0x124], 0x1
0x1800228f8: JNZ 0x180022914
0x1800228fa: LEA RDX, [R13 + 0x840]
0x180022901: MOV R9D, 0x4
0x180022907: MOV R8, R15
0x18002290a: LEA RCX, [R13 + 0x8]
0x18002290e: CALL 0x180004b70
0x180022913: NOP
0x180022914: MOV RCX, qword ptr [RBP + 0xc00]
0x18002291b: MOV RAX, qword ptr [RBP + 0xc10]
0x180022922: SUB qword ptr [RCX + 0x8], RAX
0x180022926: MOV RCX, qword ptr [RBP + 0xc48]
0x18002292d: XOR RCX, RSP
0x180022930: CALL 0x180072a30
0x180022935: MOVAPS XMM6, xmmword ptr [RSP + 0xd50]
0x18002293d: ADD RSP, 0xd68
0x180022944: POP R15
0x180022946: POP R14
0x180022948: POP R13
0x18002294a: POP R12
0x18002294c: POP RDI
0x18002294d: POP RSI
0x18002294e: POP RBX
0x18002294f: POP RBP
0x180022950: RET
```

## procedure_pseudo_code {'procedure': '0x180003960'}
Evidence: `ev_7745a2c5375e93f75436e342f941af17aa6516a9cfa67aa640258b22606e2aeb`
```c

void FUN_180003960(int param_1,int param_2,longlong *param_3,int param_4,longlong param_5,
                  longlong param_6,float param_7,float param_8,float param_9,longlong *param_10,
                  longlong *param_11,longlong *param_12)

{
  longlong lVar1;
  float fVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  int iVar7;
  longlong lVar8;
  short sVar9;
  uint uVar10;
  longlong lVar11;
  ulonglong uVar12;
  longlong lVar13;
  longlong lVar14;
  longlong lVar15;
  float fVar16;
  float local_res10;
  
  if (0 < param_2) {
    lVar13 = 0;
    do {
      lVar15 = 0;
      if (0 < (longlong)param_1) {
        lVar3 = *param_3;
        lVar4 = param_3[1];
        lVar5 = param_3[2];
        lVar6 = param_3[3];
        do {
          fVar16 = *(float *)(lVar4 + lVar13 * 4);
          iVar7 = *(int *)(lVar6 + lVar13 * 4);
          lVar1 = lVar15 * 4;
          fVar2 = *(float *)(lVar3 + lVar1);
          lVar14 = (longlong)(iVar7 * (int)param_10[1]) + *param_10;
          lVar8 = (longlong)*(int *)(lVar5 + lVar1);
          lVar11 = (longlong)((iVar7 + 1) * (int)param_10[1]) + *param_10;
          fVar16 = (1.0 - fVar2) * (1.0 - fVar16) * *(float *)(lVar14 + lVar8 * 4) +
                   (1.0 - fVar16) * fVar2 * *(float *)(lVar14 + 4 + lVar8 * 4) +
                   (1.0 - fVar2) * fVar16 * *(float *)(lVar11 + lVar8 * 4) +
                   fVar16 * fVar2 * *(float *)(lVar11 + 4 + lVar8 * 4);
          iVar7 = (int)(((fVar16 - param_7) - param_9 * 0.5) / param_9);
          if (param_7 < fVar16) {
            if ((param_8 <= fVar16) || (param_4 <= iVar7)) {
              sVar9 = (short)param_4 + -2;
              local_res10 = 1.0;
            }
            else {
              if (iVar7 < 0) {
                iVar7 = 0;
              }
              uVar10 = *(uint *)(param_6 + (longlong)iVar7 * 4);
              uVar12 = (ulonglong)(int)uVar10;
              fVar2 = *(float *)(param_5 + uVar12 * 4);
              while (fVar2 <= fVar16) {
                uVar10 = (int)uVar12 + 1;
                uVar12 = (ulonglong)uVar10;
                fVar2 = *(float *)(param_5 + (longlong)(int)uVar10 * 4);
              }
              sVar9 = (short)uVar10 + -1;
              local_res10 = (fVar16 - *(float *)(param_5 + -4 + (longlong)(int)uVar10 * 4)) /
                            (fVar2 - *(float *)(param_5 + -4 + (longlong)(int)uVar10 * 4));
            }
          }
          else {
            sVar9 = 0;
          }
          lVar8 = lVar15 * 2;
          lVar15 = lVar15 + 1;
          *(short *)((int)param_11[1] * lVar13 + lVar8 + *param_11) = sVar9;
          *(float *)((int)param_12[1] * lVar13 + lVar1 + *param_12) = local_res10;
        } while (lVar15 < param_1);
      }
      lVar13 = lVar13 + 1;
    } while (lVar13 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180016710'}
Evidence: `ev_7d59a5dfcc534694c999d7331ded9e071df00622e85c1885e9e9691b2ace6218`
```c

void FUN_180016710(longlong param_1,Mat *param_2,undefined4 param_3,undefined4 param_4,
                  longlong param_5)

{
  int iVar1;
  undefined4 *puVar2;
  _OutputArray *p_Var3;
  int iVar4;
  int iVar5;
  undefined4 *puVar6;
  longlong lVar7;
  undefined4 local_res8;
  undefined4 local_resc;
  undefined4 uVar8;
  undefined4 local_78;
  undefined4 local_74;
  undefined4 local_70 [2];
  Mat *local_68;
  undefined8 local_60;
  undefined4 local_58 [2];
  Mat *local_50;
  undefined8 local_48;
  
  iVar1 = *(int *)(param_1 + 0x28);
  iVar4 = *(int *)(param_1 + 0x2c) + -1;
  iVar5 = 0;
  if (0 < iVar1) {
    lVar7 = param_5 - (longlong)param_2;
    do {
      local_50 = param_2 + lVar7;
      if (iVar5 < iVar4) {
        local_70[0] = 0x2010000;
        local_58[0] = 0x1010000;
        uVar8 = 3;
        puVar6 = &local_res8;
        p_Var3 = (_OutputArray *)local_70;
        puVar2 = local_58;
        local_res8 = param_3;
        local_resc = param_4;
        local_68 = local_50;
        local_50 = param_2;
LAB_1800167e3:
        local_48 = 0;
        local_60 = 0;
        cv::resize(puVar2,p_Var3,puVar6,0,0,uVar8);
      }
      else {
        local_58[0] = 0x2010000;
        p_Var3 = (_OutputArray *)local_58;
        if (iVar5 != iVar4) {
          local_70[0] = 0x1010000;
          uVar8 = 1;
          puVar6 = &local_78;
          puVar2 = local_70;
          local_78 = param_3;
          local_74 = param_4;
          local_68 = param_2;
          goto LAB_1800167e3;
        }
        local_48 = 0;
        cv::Mat::copyTo(param_2,p_Var3);
      }
      iVar5 = iVar5 + 1;
      param_2 = param_2 + 0x68;
    } while (iVar5 < iVar1);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180009100'}
Evidence: `ev_7decdccb6eb8266f0437a105ea36076b8ac62f25dede70c2041eccfc3c15f82a`
```c

void FUN_180009100(undefined8 param_1,char param_2,int param_3,int param_4,longlong param_5,
                  longlong param_6,int param_7,longlong param_8,longlong param_9,longlong param_10)

{
  if (param_2 != '\0') {
    FUN_18000ea90(param_3,param_4,param_5,param_6,param_7,param_8,param_9,param_10);
    return;
  }
  FUN_18000ee30(param_3,param_4,param_5,param_6,param_7,param_8,param_10);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180003c30'}
Evidence: `ev_7e9569df2b841c9f41825f4080023d5734903aa8d8ac3144a3d10e1469480be2`
```c

void FUN_180003c30(int param_1,int param_2,longlong *param_3,int param_4,longlong param_5,
                  longlong param_6,float param_7,float param_8,float param_9,longlong *param_10,
                  longlong *param_11)

{
  float fVar1;
  longlong lVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  int iVar6;
  longlong lVar7;
  longlong lVar8;
  ulonglong uVar9;
  longlong lVar10;
  longlong lVar11;
  longlong lVar12;
  float fVar13;
  
  if (0 < param_2) {
    lVar10 = 0;
    do {
      lVar12 = 0;
      if (0 < (longlong)param_1) {
        lVar2 = *param_3;
        lVar3 = param_3[1];
        lVar4 = param_3[2];
        lVar5 = param_3[3];
        do {
          fVar13 = *(float *)(lVar3 + lVar10 * 4);
          iVar6 = *(int *)(lVar5 + lVar10 * 4);
          fVar1 = *(float *)(lVar2 + lVar12 * 4);
          lVar11 = (longlong)(iVar6 * (int)param_10[1]) + *param_10;
          lVar7 = (longlong)*(int *)(lVar4 + lVar12 * 4);
          lVar8 = (longlong)((iVar6 + 1) * (int)param_10[1]) + *param_10;
          fVar13 = (1.0 - fVar1) * (1.0 - fVar13) * *(float *)(lVar11 + lVar7 * 4) +
                   (1.0 - fVar13) * fVar1 * *(float *)(lVar11 + 4 + lVar7 * 4) +
                   (1.0 - fVar1) * fVar13 * *(float *)(lVar8 + lVar7 * 4) +
                   fVar13 * fVar1 * *(float *)(lVar8 + 4 + lVar7 * 4);
          iVar6 = (int)(((fVar13 - param_7) - param_9 * 0.5) / param_9);
          if (param_7 < fVar13) {
            if ((param_8 <= fVar13) || (param_4 <= iVar6)) {
              uVar9 = (ulonglong)(param_4 - 1);
            }
            else {
              if (iVar6 < 0) {
                iVar6 = 0;
              }
              uVar9 = (ulonglong)*(int *)(param_6 + (longlong)iVar6 * 4);
              fVar1 = *(float *)(param_5 + uVar9 * 4);
              while (iVar6 = (int)uVar9, fVar1 <= fVar13) {
                uVar9 = (ulonglong)(iVar6 + 1U);
                fVar1 = *(float *)(param_5 + (longlong)(int)(iVar6 + 1U) * 4);
              }
              if ((fVar13 - *(float *)(param_5 + -4 + (longlong)iVar6 * 4)) /
                  (fVar1 - *(float *)(param_5 + -4 + (longlong)iVar6 * 4)) < 0.5) {
                uVar9 = (ulonglong)(iVar6 - 1);
              }
            }
          }
          else {
            uVar9 = 0;
          }
          lVar7 = lVar12 * 2;
          lVar12 = lVar12 + 1;
          *(short *)((int)param_11[1] * lVar10 + lVar7 + *param_11) = (short)uVar9;
        } while (lVar12 < param_1);
      }
      lVar10 = lVar10 + 1;
    } while (lVar10 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180013c90'}
Evidence: `ev_7f8f4c7930f22d5af3e71b01c45db9493e3cf22d3a06f0f2e7e128154a62a5d3`
```c

void FUN_180013c90(int param_1,int param_2,longlong *param_3,float param_4)

{
  longlong lVar1;
  longlong lVar2;
  longlong lVar3;
  int iVar4;
  longlong lVar5;
  
  iVar4 = 0;
  lVar5 = (longlong)param_1;
  if (0 < param_2) {
    do {
      lVar3 = 0;
      if (3 < lVar5) {
        do {
          lVar2 = lVar3 * 4 + 0xc;
          lVar3 = lVar3 + 4;
          lVar1 = (longlong)((int)param_3[1] * iVar4) + *param_3;
          *(float *)(lVar1 + -0x10 + lVar3 * 4) = param_4 * *(float *)(lVar1 + -0x10 + lVar3 * 4);
          lVar1 = (longlong)((int)param_3[1] * iVar4) + *param_3;
          *(float *)(lVar1 + -0xc + lVar3 * 4) = param_4 * *(float *)(lVar1 + -0xc + lVar3 * 4);
          lVar1 = (int)param_3[1] * iVar4 + lVar2;
          *(float *)(lVar1 + -4 + *param_3) = param_4 * *(float *)(lVar1 + -4 + *param_3);
          lVar2 = (int)param_3[1] * iVar4 + lVar2;
          *(float *)(lVar2 + *param_3) = param_4 * *(float *)(lVar2 + *param_3);
        } while (lVar3 < lVar5 + -3);
      }
      for (; lVar3 < lVar5; lVar3 = lVar3 + 1) {
        lVar2 = (longlong)((int)param_3[1] * iVar4) + lVar3 * 4;
        *(float *)(lVar2 + *param_3) = param_4 * *(float *)(lVar2 + *param_3);
      }
      iVar4 = iVar4 + 1;
    } while (iVar4 < param_2);
  }
  return;
}


```

## procedure_assembly {'procedure': '0x18001e660'}
Evidence: `ev_84fb3aaf6346047ea5be294fa624b04eef3738bb1aca0394bbaa0db53e23b55f`
```c
0x18001e660: PUSH RBX
0x18001e662: SUB RSP, 0x30
0x18001e666: MOV RAX, qword ptr [RSP + 0x60]
0x18001e66b: MOV RBX, RCX
0x18001e66e: MOV qword ptr [RSP + 0x20], RAX
0x18001e673: CALL 0x18001e6a0
0x18001e678: CMP EAX, 0x1
0x18001e67b: JZ 0x18001e68c
0x18001e67d: XOR ECX, ECX
0x18001e67f: MOV qword ptr [RBX + 0x840], RCX
0x18001e686: MOV byte ptr [RBX + 0x868], CL
0x18001e68c: ADD RSP, 0x30
0x18001e690: POP RBX
0x18001e691: RET
```

## procedure_pseudo_code {'procedure': '0x18001a680'}
Evidence: `ev_8e5a78edd46bd40bc63f0559bc18853f73d72e3cdc36eda68496730ddc311613`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_18001a680(longlong param_1,longlong *param_2,Mat *param_3,longlong param_4,longlong param_5
                  ,uint param_6,int param_7)

{
  int iVar1;
  Mat *pMVar2;
  undefined1 auStackY_598 [32];
  longlong *local_550;
  undefined8 local_548;
  longlong local_540;
  longlong local_538;
  undefined8 local_530;
  Mat local_528 [104];
  Mat local_4c0 [96];
  undefined4 local_460;
  Mat local_458 [96];
  undefined4 local_3f8;
  Mat local_3f0 [96];
  undefined4 local_390;
  Mat local_388 [96];
  undefined4 local_328;
  Mat local_320 [96];
  undefined4 local_2c0;
  Mat local_2b8 [96];
  undefined4 local_258;
  Mat local_250 [96];
  undefined4 local_1f0;
  Mat local_1e8 [416];
  ulonglong local_48;
  
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_598;
  iVar1 = param_6 * 2 + 1;
  FUN_1800726b8((longlong)local_528,0x68,4,&LAB_180006700);
  FUN_1800726b8((longlong)local_1e8,0x68,4,&LAB_180006700);
  local_530 = 0;
  local_548 = 0x20;
  local_540 = 0;
  local_538 = param_2[1] + *param_2;
  local_550 = param_2;
  FUN_180026340((longlong *)&local_550,param_1,param_6,param_7,local_528,local_1e8);
  FUN_180017970(param_4 + 0x138,(longlong)param_3,(longlong)local_4c0,param_7);
  FUN_1800013f0(local_388,param_3);
  local_328 = *(undefined4 *)(param_3 + 0x60);
  FUN_1800013f0(local_320,local_4c0);
  local_2c0 = local_460;
  FUN_1800013f0(local_2b8,local_458);
  local_258 = local_3f8;
  pMVar2 = local_250;
  FUN_1800013f0(pMVar2,local_3f0);
  local_1f0 = local_390;
  FUN_1800120e0(pMVar2,(longlong)local_388,(longlong)local_1e8,iVar1,param_7 + 1U);
  pMVar2 = local_1e8;
  FUN_180018360((longlong)pMVar2,param_4 + 0x270,param_4 + 0x3a8,(longlong)local_528,param_7);
  FUN_1800120e0(pMVar2,(longlong)local_528,(longlong)local_1e8,iVar1,param_7 + 1U);
  FUN_180018e40((longlong *)(param_1 + 0x7a0),(longlong)local_1e8,param_4,param_5,param_7);
  _eh_vector_destructor_iterator_(local_388,0x68,4,FUN_180001750);
  local_550[1] = local_550[1] - local_540;
  _eh_vector_destructor_iterator_(local_1e8,0x68,4,FUN_180001750);
  _eh_vector_destructor_iterator_(local_528,0x68,4,FUN_180001750);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180019010'}
Evidence: `ev_8ea9692476c1b1f3b59d1c675fa2364d21d3a9617f784d04af1be93fb1e32eb2`
```c

void FUN_180019010(int param_1,int param_2,longlong *param_3,longlong *param_4,float param_5,
                  longlong *param_6)

{
  longlong lVar1;
  longlong lVar2;
  longlong lVar3;
  float fVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  
  if (0 < param_2) {
    lVar5 = (longlong)param_1;
    lVar7 = 0;
    do {
      lVar6 = 0;
      if (3 < lVar5) {
        do {
          lVar3 = lVar6 * 4;
          lVar1 = lVar3 + 0xc;
          fVar4 = *(float *)((int)param_3[1] * lVar7 + *param_3 + lVar3);
          *(float *)((int)param_6[1] * lVar7 + lVar3 + *param_6) =
               1.0 / ((*(float *)((int)param_4[1] * lVar7 + lVar3 + *param_4) - fVar4 * fVar4) +
                     param_5);
          fVar4 = *(float *)((int)param_3[1] * lVar7 + *param_3 + 4 + lVar3);
          lVar2 = lVar3 + 8;
          *(float *)((int)param_6[1] * lVar7 + lVar3 + 4 + *param_6) =
               1.0 / ((*(float *)((int)param_4[1] * lVar7 + lVar3 + 4 + *param_4) - fVar4 * fVar4) +
                     param_5);
          fVar4 = *(float *)((int)param_3[1] * lVar7 + lVar2 + *param_3);
          *(float *)((int)param_6[1] * lVar7 + lVar2 + *param_6) =
               1.0 / ((*(float *)((int)param_4[1] * lVar7 + lVar2 + *param_4) - fVar4 * fVar4) +
                     param_5);
          fVar4 = *(float *)((int)param_3[1] * lVar7 + lVar1 + *param_3);
          *(float *)((int)param_6[1] * lVar7 + lVar1 + *param_6) =
               1.0 / ((*(float *)((int)param_4[1] * lVar7 + lVar1 + *param_4) - fVar4 * fVar4) +
                     param_5);
          lVar6 = lVar6 + 4;
        } while (lVar6 < lVar5 + -3);
      }
      for (; lVar6 < lVar5; lVar6 = lVar6 + 1) {
        lVar1 = lVar6 * 4;
        fVar4 = *(float *)((int)param_3[1] * lVar7 + *param_3 + lVar1);
        *(float *)((int)param_6[1] * lVar7 + lVar1 + *param_6) =
             1.0 / ((*(float *)((int)param_4[1] * lVar7 + lVar1 + *param_4) - fVar4 * fVar4) +
                   param_5);
      }
      lVar7 = lVar7 + 1;
    } while (lVar7 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002a880'}
Evidence: `ev_8fba74678d11802f9431d8e779fd911eb3df9415a64acfbb14431c4f26bb7b9e`
```c

void FUN_18002a880(longlong *param_1,longlong param_2,Mat *param_3,Mat *param_4,Mat *param_5,
                  Mat *param_6)

{
  int *piVar1;
  int iVar2;
  undefined4 uVar3;
  uint uVar4;
  undefined4 uVar5;
  Mat *pMVar6;
  uint uVar7;
  ulonglong uVar8;
  uint uVar10;
  longlong lVar11;
  int iVar12;
  uint local_res18;
  undefined4 local_res1c;
  uint local_res20;
  undefined4 local_res24;
  uint local_218;
  undefined4 local_214;
  uint local_210;
  undefined4 local_20c;
  Mat local_208 [4];
  int local_204;
  undefined8 local_1f8;
  undefined8 local_1f0;
  undefined8 local_1e8;
  undefined8 local_1e0;
  longlong local_1d0;
  longlong local_1c8;
  undefined1 *local_1c0;
  undefined1 local_1b8 [32];
  Mat local_198 [4];
  int local_194;
  undefined8 local_188;
  undefined8 local_180;
  undefined8 local_178;
  undefined8 local_170;
  longlong local_160;
  longlong local_158;
  undefined1 *local_150;
  undefined1 local_148 [32];
  Mat local_128 [4];
  int local_124;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  longlong local_f0;
  longlong local_e8;
  undefined1 *local_e0;
  undefined1 local_d8 [32];
  undefined8 local_b8;
  Mat local_a8 [4];
  int local_a4;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_80;
  longlong local_70;
  longlong local_68;
  undefined1 *local_60;
  undefined1 local_58 [32];
  ulonglong uVar9;
  
  local_b8 = 0xfffffffffffffffe;
  uVar7 = *(uint *)(param_2 + 0x4dc);
  uVar3 = *(undefined4 *)(param_2 + 0x4e0);
  uVar4 = *(uint *)(param_2 + 0x4d4);
  uVar5 = *(undefined4 *)(param_2 + 0x4d8);
  local_res18 = uVar4;
  local_res1c = uVar5;
  pMVar6 = (Mat *)FUN_18002b990(param_1,(uint *)local_208,&local_res18,0,*(int *)(param_2 + 0x14));
  if (param_3 != pMVar6) {
    FUN_180001780(param_3,pMVar6);
  }
  *(undefined4 *)(param_3 + 0x60) = *(undefined4 *)(pMVar6 + 0x60);
  if (local_1d0 != 0) {
    LOCK();
    piVar1 = (int *)(local_1d0 + 0x14);
    iVar12 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar12 == 1) {
      cv::Mat::deallocate(local_208);
    }
  }
  uVar8 = 0;
  local_1d0 = 0;
  local_1f8 = 0;
  local_1e0 = 0;
  local_1e8 = 0;
  local_1f0 = 0;
  uVar9 = uVar8;
  if (0 < local_204) {
    do {
      *(undefined4 *)(uVar8 + local_1c8) = 0;
      uVar10 = (int)uVar9 + 1;
      uVar8 = uVar8 + 4;
      uVar9 = (ulonglong)uVar10;
    } while ((int)uVar10 < local_204);
  }
  if (local_1c0 != local_1b8) {
    cv::fastFree(local_1c0);
  }
  local_res20 = uVar7;
  local_res24 = uVar3;
  pMVar6 = (Mat *)FUN_18002b990(param_1,(uint *)local_198,&local_res20,0,1);
  if (param_4 != pMVar6) {
    FUN_180001780(param_4,pMVar6);
  }
  *(undefined4 *)(param_4 + 0x60) = *(undefined4 *)(pMVar6 + 0x60);
  if (local_160 != 0) {
    LOCK();
    piVar1 = (int *)(local_160 + 0x14);
    iVar12 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar12 == 1) {
      cv::Mat::deallocate(local_198);
    }
  }
  uVar8 = 0;
  local_160 = 0;
  local_188 = 0;
  local_170 = 0;
  local_178 = 0;
  local_180 = 0;
  uVar9 = uVar8;
  if (0 < local_194) {
    do {
      *(undefined4 *)(uVar8 + local_158) = 0;
      uVar10 = (int)uVar9 + 1;
      uVar8 = uVar8 + 4;
      uVar9 = (ulonglong)uVar10;
    } while ((int)uVar10 < local_194);
  }
  if (local_150 != local_148) {
    cv::fastFree(local_150);
  }
  local_218 = uVar7;
  local_214 = uVar3;
  pMVar6 = (Mat *)FUN_18002b990(param_1,(uint *)local_128,&local_218,0,1);
  if (param_5 != pMVar6) {
    FUN_180001780(param_5,pMVar6);
  }
  *(undefined4 *)(param_5 + 0x60) = *(undefined4 *)(pMVar6 + 0x60);
  if (local_f0 != 0) {
    LOCK();
    piVar1 = (int *)(local_f0 + 0x14);
    iVar12 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar12 == 1) {
      cv::Mat::deallocate(local_128);
    }
  }
  iVar12 = 0;
  uVar8 = 0;
  local_f0 = 0;
  local_118 = 0;
  local_100 = 0;
  local_108 = 0;
  local_110 = 0;
  if (0 < local_124) {
    lVar11 = 0;
    uVar9 = uVar8;
    do {
      *(undefined4 *)(lVar11 + local_e8) = 0;
      uVar7 = (int)uVar9 + 1;
      uVar9 = (ulonglong)uVar7;
      lVar11 = lVar11 + 4;
    } while ((int)uVar7 < local_124);
  }
  if (local_e0 != local_d8) {
    cv::fastFree(local_e0);
  }
  local_210 = uVar4;
  local_20c = uVar5;
  pMVar6 = (Mat *)FUN_18002b990(param_1,(uint *)local_a8,&local_210,0,1);
  if (param_6 != pMVar6) {
    FUN_180001780(param_6,pMVar6);
  }
  *(undefined4 *)(param_6 + 0x60) = *(undefined4 *)(pMVar6 + 0x60);
  if (local_70 != 0) {
    LOCK();
    piVar1 = (int *)(local_70 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(local_a8);
    }
  }
  local_70 = 0;
  local_98 = 0;
  local_80 = 0;
  local_88 = 0;
  local_90 = 0;
  if (0 < local_a4) {
    do {
      *(undefined4 *)(uVar8 + local_68) = 0;
      iVar12 = iVar12 + 1;
      uVar8 = uVar8 + 4;
    } while (iVar12 < local_a4);
  }
  if (local_60 != local_58) {
    cv::fastFree(local_60);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180019240'}
Evidence: `ev_978793a9a83bd833bece8c6b087a34c8686ac9ed701ba9b0ae5dc4163a185051`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180019240(longlong param_1,longlong *param_2,Mat *param_3,longlong param_4,
                  undefined8 param_5,uint param_6,double param_7)

{
  int *piVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  uint uVar5;
  undefined4 *puVar6;
  longlong lVar7;
  ulonglong uVar8;
  ulonglong uVar9;
  ulonglong uVar10;
  undefined1 auStackY_2b8 [32];
  undefined8 local_278;
  undefined4 local_270 [2];
  longlong local_268;
  undefined8 local_260;
  undefined4 local_258 [2];
  longlong local_250;
  undefined8 local_248;
  undefined4 local_240 [2];
  longlong local_238;
  undefined8 local_230;
  longlong local_228;
  undefined4 uStack_220;
  undefined4 uStack_21c;
  longlong local_218;
  undefined4 uStack_210;
  undefined4 uStack_20c;
  longlong local_208;
  undefined4 uStack_200;
  undefined4 uStack_1fc;
  longlong local_1f8;
  undefined8 uStack_1f0;
  longlong local_1e8;
  undefined8 uStack_1e0;
  longlong local_1d8;
  undefined8 uStack_1d0;
  undefined8 local_1c8;
  longlong *local_1c0;
  undefined8 local_1b8;
  longlong local_1b0;
  longlong local_1a8;
  undefined8 local_1a0;
  undefined8 local_198;
  undefined8 uStack_190;
  longlong local_188;
  undefined8 uStack_180;
  undefined8 local_178;
  undefined8 uStack_170;
  undefined8 local_168;
  longlong lStack_160;
  undefined8 *local_158;
  undefined8 *local_150;
  undefined8 local_148;
  undefined8 local_140;
  undefined4 local_138;
  undefined8 local_128;
  undefined8 uStack_120;
  undefined8 local_118;
  undefined8 uStack_110;
  undefined8 local_108;
  undefined8 uStack_100;
  undefined8 local_f8;
  longlong lStack_f0;
  undefined8 *local_e8;
  undefined8 *local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined4 local_c8;
  undefined8 local_b8;
  undefined8 uStack_b0;
  undefined8 local_a8;
  undefined8 uStack_a0;
  undefined8 local_98;
  undefined8 uStack_90;
  undefined8 local_88;
  longlong lStack_80;
  undefined8 *local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined4 local_58;
  ulonglong local_48;
  
  local_1c8 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_2b8;
  iVar2 = param_6 * 2 + 1;
  lVar7 = param_4 + 0x138;
  memset(&local_b8,0,0x68);
  local_b8 = 0x42ff0000;
  uStack_b0 = 0;
  local_a8 = 0;
  uStack_a0 = 0;
  local_98 = 0;
  uStack_90 = 0;
  local_88 = 0;
  lStack_80 = 0;
  local_78 = &uStack_b0;
  local_70 = &local_68;
  local_60 = 0;
  local_68 = 0;
  local_58 = 0;
  memset(&local_128,0,0x68);
  local_128 = 0x42ff0000;
  uStack_120 = 0;
  local_118 = 0;
  uStack_110 = 0;
  local_108 = 0;
  uStack_100 = 0;
  local_f8 = 0;
  lStack_f0 = 0;
  local_e8 = &uStack_120;
  local_e0 = &local_d8;
  local_d0 = 0;
  local_d8 = 0;
  local_c8 = 0;
  memset(&local_198,0,0x68);
  local_198 = 0x42ff0000;
  uStack_190 = 0;
  local_188 = 0;
  uStack_180 = 0;
  local_178 = 0;
  uStack_170 = 0;
  local_168 = 0;
  lStack_160 = 0;
  local_158 = &uStack_190;
  local_150 = &local_148;
  local_140 = 0;
  local_148 = 0;
  local_138 = 0;
  local_1a0 = 0;
  local_1b8 = 0x20;
  local_1b0 = 0;
  local_1a8 = param_2[1] + *param_2;
  local_1c0 = param_2;
  FUN_180025f50((longlong *)&local_1c0,param_1,param_6,1,&local_b8,(Mat *)&local_128,
                (Mat *)&local_198);
  local_230 = 0;
  local_240[0] = 0x2010000;
  local_238 = param_4;
  cv::Mat::copyTo(param_3,(_OutputArray *)local_240);
  uVar9 = 0;
  local_248 = 0;
  local_258[0] = 0x2010000;
  local_260 = 0;
  local_270[0] = 0x1010000;
  local_278 = 0;
  puVar6 = local_270;
  local_268 = param_4;
  local_250 = lVar7;
  cv::resize(puVar6,local_258,&local_278);
  iVar3 = **(int **)(param_4 + 0x178);
  iVar4 = (*(int **)(param_4 + 0x178))[1];
  FUN_1800120e0(puVar6,lVar7,param_4 + 0x270,iVar2,1);
  FUN_180017970(lVar7,lVar7,(longlong)&local_128,1);
  FUN_1800120e0(lVar7,(longlong)&local_128,(longlong)&local_198,iVar2,1);
  local_218 = *(longlong *)(param_4 + 0x3b8);
  uStack_210 = (undefined4)*(undefined8 *)(param_4 + 0x3f8);
  uStack_1d0 = CONCAT44(uStack_20c,uStack_210);
  local_208 = local_188;
  uStack_200 = (undefined4)local_148;
  uStack_1e0 = CONCAT44(uStack_1fc,(undefined4)local_148);
  local_1e8 = local_188;
  local_228 = *(longlong *)(param_4 + 0x280);
  uStack_220 = (undefined4)*(undefined8 *)(param_4 + 0x2c0);
  uStack_1f0 = CONCAT44(uStack_21c,uStack_220);
  local_1f8 = local_228;
  local_1d8 = local_218;
  FUN_180019010(iVar4,iVar3,&local_1f8,&local_1e8,(float)param_7,&local_1d8);
  local_1c0[1] = local_1c0[1] - local_1b0;
  if (lStack_160 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_160 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_198);
    }
  }
  lStack_160 = 0;
  local_188 = 0;
  uStack_170 = 0;
  local_178 = 0;
  uStack_180 = 0;
  uVar8 = uVar9;
  uVar10 = uVar9;
  if (0 < local_198._4_4_) {
    do {
      *(undefined4 *)((longlong)local_158 + uVar10) = 0;
      uVar5 = (int)uVar8 + 1;
      uVar8 = (ulonglong)uVar5;
      uVar10 = uVar10 + 4;
    } while ((int)uVar5 < local_198._4_4_);
  }
  if (local_150 != &local_148) {
    cv::fastFree(local_150);
  }
  if (lStack_f0 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_f0 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_128);
    }
  }
  lStack_f0 = 0;
  local_118 = 0;
  uStack_100 = 0;
  local_108 = 0;
  uStack_110 = 0;
  uVar8 = uVar9;
  uVar10 = uVar9;
  if (0 < local_128._4_4_) {
    do {
      *(undefined4 *)((longlong)local_e8 + uVar10) = 0;
      uVar5 = (int)uVar8 + 1;
      uVar8 = (ulonglong)uVar5;
      uVar10 = uVar10 + 4;
    } while ((int)uVar5 < local_128._4_4_);
  }
  if (local_e0 != &local_d8) {
    cv::fastFree(local_e0);
  }
  if (lStack_80 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_80 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_b8);
    }
  }
  lStack_80 = 0;
  local_a8 = 0;
  uStack_90 = 0;
  local_98 = 0;
  uStack_a0 = 0;
  uVar8 = uVar9;
  if (0 < local_b8._4_4_) {
    do {
      *(undefined4 *)((longlong)local_78 + uVar8) = 0;
      uVar5 = (int)uVar9 + 1;
      uVar9 = (ulonglong)uVar5;
      uVar8 = uVar8 + 4;
    } while ((int)uVar5 < local_b8._4_4_);
  }
  if (local_70 != &local_68) {
    cv::fastFree(local_70);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002d520'}
Evidence: `ev_a12ba5d003f4f9d422ec9e685c5d7c75d99487f94da58e6b66c39010e772a3f0`
```c

void FUN_18002d520(int param_1,int param_2,int param_3,int param_4,float param_5,float param_6,
                  longlong *param_7,longlong *param_8,longlong *param_9)

{
  int iVar1;
  int iVar2;
  int iVar3;
  longlong lVar4;
  int iVar5;
  longlong lVar6;
  int iVar7;
  int local_38;
  
  if (0 < param_2) {
    local_38 = 0;
    lVar4 = 0;
    do {
      iVar7 = 0;
      lVar6 = 0;
      if (3 < param_1) {
        iVar3 = 2;
        iVar5 = (int)((float)local_38 * param_6);
        do {
          iVar2 = (int)((float)iVar7 * param_5);
          if (param_3 <= iVar2) {
            iVar2 = param_3 + -1;
          }
          iVar1 = iVar5;
          if (param_4 <= iVar5) {
            iVar1 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar1 + (longlong)iVar2 + *param_8) !=
              '\0') {
            *(float *)((int)param_9[1] * lVar4 + lVar6 * 4 + *param_9) =
                 (float)*(byte *)((int)param_7[1] * lVar4 + *param_7 + lVar6);
          }
          iVar2 = (int)((float)(iVar3 + -1) * param_5);
          if (param_3 <= iVar2) {
            iVar2 = param_3 + -1;
          }
          iVar1 = iVar5;
          if (param_4 <= iVar5) {
            iVar1 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar1 + (longlong)iVar2 + *param_8) !=
              '\0') {
            *(float *)((int)param_9[1] * lVar4 + lVar6 * 4 + 4 + *param_9) =
                 (float)*(byte *)((int)param_7[1] * lVar4 + *param_7 + 1 + lVar6);
          }
          iVar2 = (int)((float)iVar3 * param_5);
          if (param_3 <= iVar2) {
            iVar2 = param_3 + -1;
          }
          iVar1 = iVar5;
          if (param_4 <= iVar5) {
            iVar1 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar1 + (longlong)iVar2 + *param_8) !=
              '\0') {
            *(float *)((int)param_9[1] * lVar4 + lVar6 * 4 + 8 + *param_9) =
                 (float)*(byte *)((int)param_7[1] * lVar4 + *param_7 + 2 + lVar6);
          }
          iVar2 = (int)((float)(iVar3 + 1) * param_5);
          if (param_3 <= iVar2) {
            iVar2 = param_3 + -1;
          }
          iVar1 = iVar5;
          if (param_4 <= iVar5) {
            iVar1 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar1 + (longlong)iVar2 + *param_8) !=
              '\0') {
            *(float *)((int)param_9[1] * lVar4 + lVar6 * 4 + 0xc + *param_9) =
                 (float)*(byte *)((int)param_7[1] * lVar4 + *param_7 + 3 + lVar6);
          }
          iVar7 = iVar7 + 4;
          lVar6 = lVar6 + 4;
          iVar3 = iVar3 + 4;
        } while (iVar7 < param_1 + -3);
      }
      if (iVar7 < param_1) {
        do {
          iVar3 = (int)((float)iVar7 * param_5);
          if (param_3 <= iVar3) {
            iVar3 = param_3 + -1;
          }
          iVar5 = (int)((float)local_38 * param_6);
          if (param_4 <= (int)((float)local_38 * param_6)) {
            iVar5 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar5 + (longlong)iVar3 + *param_8) !=
              '\0') {
            *(float *)((int)param_9[1] * lVar4 + lVar6 * 4 + *param_9) =
                 (float)*(byte *)((int)param_7[1] * lVar4 + *param_7 + lVar6);
          }
          iVar7 = iVar7 + 1;
          lVar6 = lVar6 + 1;
        } while (iVar7 < param_1);
      }
      local_38 = local_38 + 1;
      lVar4 = lVar4 + 1;
    } while (local_38 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002f300'}
Evidence: `ev_a132a4eff8c397e36a342bff3e35925871b32686b8fefce0d2313ec97711dbaf`
```c

void FUN_18002f300(longlong param_1,uint param_2)

{
  undefined4 *puVar1;
  int iVar2;
  int iVar3;
  undefined4 *puVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  longlong lVar12;
  longlong lVar13;
  ulonglong uVar14;
  longlong lVar15;
  int iVar16;
  int iVar17;
  
  if (0 < (int)param_2) {
    iVar2 = **(int **)(param_1 + 0x40);
    lVar15 = (longlong)(*(int **)(param_1 + 0x40))[1];
    iVar16 = 1;
    iVar17 = -1;
    uVar14 = (ulonglong)param_2;
    do {
      iVar3 = *(int *)(param_1 + 0x50);
      lVar12 = *(longlong *)(param_1 + 0x10);
      lVar7 = 0;
      lVar13 = iVar3 * iVar17 + lVar12;
      lVar10 = (iVar2 + -1 + iVar16) * iVar3 + lVar12;
      lVar11 = iVar3 * iVar16 + lVar12;
      lVar12 = (iVar17 + iVar2) * iVar3 + lVar12;
      if (3 < lVar15) {
        lVar5 = lVar11 - lVar13;
        lVar6 = lVar12 - lVar13;
        lVar9 = lVar10 - lVar13;
        lVar8 = (lVar15 - 4U >> 2) + 1;
        lVar7 = lVar8 * 4;
        puVar4 = (undefined4 *)(lVar13 + 4);
        do {
          puVar1 = puVar4 + 4;
          puVar4[-1] = *(undefined4 *)(lVar5 + -4 + (longlong)puVar4);
          *(undefined4 *)(lVar9 + -0x14 + (longlong)puVar1) =
               *(undefined4 *)(lVar6 + -0x14 + (longlong)puVar1);
          *puVar4 = *(undefined4 *)(lVar5 + -0x10 + (longlong)puVar1);
          *(undefined4 *)(lVar9 + -0x10 + (longlong)puVar1) =
               *(undefined4 *)(lVar6 + -0x10 + (longlong)puVar1);
          puVar4[1] = *(undefined4 *)(lVar5 + -0xc + (longlong)puVar1);
          *(undefined4 *)(lVar9 + -0xc + (longlong)puVar1) =
               *(undefined4 *)(lVar6 + -0xc + (longlong)puVar1);
          puVar4[2] = *(undefined4 *)(lVar5 + -8 + (longlong)puVar1);
          *(undefined4 *)(lVar9 + -8 + (longlong)puVar1) =
               *(undefined4 *)(lVar6 + -8 + (longlong)puVar1);
          lVar8 = lVar8 + -1;
          puVar4 = puVar1;
        } while (lVar8 != 0);
      }
      if (lVar7 < lVar15) {
        lVar5 = lVar15 - lVar7;
        puVar4 = (undefined4 *)(lVar13 + lVar7 * 4);
        do {
          puVar1 = puVar4 + 1;
          *puVar4 = *(undefined4 *)((lVar11 - lVar13) + (longlong)puVar4);
          *(undefined4 *)((lVar10 - lVar13) + -4 + (longlong)puVar1) =
               *(undefined4 *)((lVar12 - lVar13) + -4 + (longlong)puVar1);
          lVar5 = lVar5 + -1;
          puVar4 = puVar1;
        } while (lVar5 != 0);
      }
      iVar16 = iVar16 + 1;
      iVar17 = iVar17 + -1;
      uVar14 = uVar14 - 1;
    } while (uVar14 != 0);
  }
  return;
}


```

## procedure_assembly {'procedure': '0x18001e6a0'}
Evidence: `ev_a2aed7eeceb789d780a6b587d118bd723000fd2865234db35bce33a20a222b74`
```c
0x18001e6a0: MOV RAX, RSP
0x18001e6a3: PUSH RBP
0x18001e6a4: PUSH RSI
0x18001e6a5: PUSH RDI
0x18001e6a6: PUSH R12
0x18001e6a8: PUSH R13
0x18001e6aa: PUSH R14
0x18001e6ac: PUSH R15
0x18001e6ae: LEA RBP, [RAX + -0x788]
0x18001e6b5: SUB RSP, 0x850
0x18001e6bc: MOV qword ptr [RBP + 0x30], -0x2
0x18001e6c4: MOV qword ptr [RAX + 0x10], RBX
0x18001e6c8: MOVAPS xmmword ptr [RAX + -0x48], XMM6
0x18001e6cc: MOV RAX, qword ptr [0x18008b080]
0x18001e6d3: XOR RAX, RSP
0x18001e6d6: MOV qword ptr [RBP + 0x730], RAX
0x18001e6dd: MOV R12, R9
0x18001e6e0: MOV qword ptr [RSP + 0x60], R8
0x18001e6e5: MOV qword ptr [RSP + 0x50], RDX
0x18001e6ea: MOV RDI, RCX
0x18001e6ed: MOV R13, qword ptr [RBP + 0x7b0]
0x18001e6f4: XOR EBX, EBX
0x18001e6f6: CMP byte ptr [0x18008c518], BL
0x18001e6fc: JZ 0x18001e74b
0x18001e6fe: CMP qword ptr [0x18008c538], RBX
0x18001e705: JNZ 0x18001e719
0x18001e707: MOV ECX, 0x280000
0x18001e70c: CALL qword ptr [0x180078128]
0x18001e712: MOV qword ptr [0x18008c538], RAX
0x18001e719: MOV byte ptr [0x18008c530], BL
0x18001e71f: CALL qword ptr [0x180078280]
0x18001e725: MOV qword ptr [0x18008c520], RAX
0x18001e72c: MOV qword ptr [0x18008c528], RAX
0x18001e733: MOV dword ptr [RSP + 0x20], EBX
0x18001e737: XOR R9D, R9D
0x18001e73a: XOR R8D, R8D
0x18001e73d: XOR EDX, EDX
0x18001e73f: LEA RCX, [0x18007b328]
0x18001e746: CALL 0x180001d80
0x18001e74b: LEA RSI, [RDI + 0x8]
0x18001e74f: LEA R15, [RDI + 0x840]
0x18001e756: CMP dword ptr [R15], EBX
0x18001e759: JNZ 0x18001e765
0x18001e75b: MOV EAX, 0x15
0x18001e760: JMP 0x18001f8be
0x18001e765: MOV qword ptr [RSP + 0x40], R12
0x18001e76a: MOV qword ptr [RSP + 0x48], R13
0x18001e76f: XORPS XMM6, XMM6
0x18001e772: TEST R12, R12
0x18001e775: JZ 0x18001e793
0x18001e777: MOVAPS XMM1, XMM6
0x18001e77a: MOV RCX, R13
0x18001e77d: CALL R12
0x18001e780: TEST EAX, EAX
0x18001e782: SETNZ AL
0x18001e785: TEST AL, AL
0x18001e787: JZ 0x18001e793
0x18001e789: MOV EAX, 0x19
0x18001e78e: JMP 0x18001f8be
0x18001e793: MOV EBX, dword ptr [RSI]
0x18001e795: MOV R14D, dword ptr [RSI + 0x4]
0x18001e799: XOR EDX, EDX
0x18001e79b: LEA R8D, [RDX + 0x60]
0x18001e79f: LEA RCX, [RBP + 0x6d0]
0x18001e7a6: CALL 0x180072a62
0x18001e7ab: MOVSXD RAX, dword ptr [RSI + 0x10]
0x18001e7af: MOV qword ptr [RSP + 0x28], RAX
0x18001e7b4: MOV RAX, qword ptr [RSP + 0x60]
0x18001e7b9: MOV qword ptr [RSP + 0x20], RAX
0x18001e7be: MOV R9D, 0x5
0x18001e7c4: MOV R8D, EBX
0x18001e7c7: MOV EDX, R14D
0x18001e7ca: LEA RCX, [RBP + 0x6d0]
0x18001e7d1: CALL 0x1800014c0
0x18001e7d6: NOP
0x18001e7d7: XOR EDX, EDX
0x18001e7d9: LEA R8D, [RDX + 0x60]
0x18001e7dd: LEA RCX, [RBP + 0x670]
0x18001e7e4: CALL 0x180072a62
0x18001e7e9: CMP dword ptr [RSI + 0x14], 0x1
0x18001e7ed: JNZ 0x18001e7fb
0x18001e7ef: MOV R9D, 0x2
0x18001e7f5: LEA EAX, [R9 + -0x2]
0x18001e7f9: JMP 0x18001e806
0x18001e7fb: MOV EAX, 0x10
0x18001e800: MOV R9D, 0x12
0x18001e806: CMP dword ptr [RSI + 0x18], 0x8
0x18001e80a: CMOVZ R9D, EAX
0x18001e80e: MOVSXD RAX, dword ptr [RSI + 0xc]
0x18001e812: MOV qword ptr [RSP + 0x28], RAX
0x18001e817: MOV RAX, qword ptr [RSP + 0x50]
0x18001e81c: MOV qword ptr [RSP + 0x20], RAX
0x18001e821: MOV R8D, EBX
0x18001e824: MOV EDX, R14D
0x18001e827: LEA RCX, [RBP + 0x670]
0x18001e82e: CALL 0x1800014c0
0x18001e833: NOP
0x18001e834: LEA R14, [RDI + 0x808]
0x18001e83b: MOV R8, R15
0x18001e83e: MOV RDX, R14
0x18001e841: MOV RCX, RSI
0x18001e844: CALL 0x180002e50
0x18001e849: CMP dword ptr [RDI + 0x844], 0x1
0x18001e850: JNZ 0x18001e8eb
0x18001e856: XOR ESI, ESI
0x18001e858: MOV qword ptr [RBP + -0x38], RSI
0x18001e85c: MOV dword ptr [RBP + -0x48], 0x2010000
0x18001e863: LEA RAX, [RBP + 0x670]
0x18001e86a: MOV qword ptr [RBP + -0x40], RAX
0x18001e86e: LEA RCX, [RDI + 0x878]
0x18001e875: LEA RDX, [RBP + -0x48]
0x18001e879: CALL qword ptr [0x1800782b0]
0x18001e87f: MOV RAX, qword ptr [RDI + 0x848]
0x18001e886: MOVSS XMM0, dword ptr [RAX]
0x18001e88a: CVTPS2PD XMM0, XMM0
0x18001e88d: UNPCKLPD XMM0, XMM0
0x18001e891: MOVUPS xmmword ptr [RBP], XMM0
0x18001e895: MOVUPS xmmword ptr [RBP + 0x10], XMM0
0x18001e899: MOV dword ptr [RBP + -0x60], 0xc1020006
0x18001e8a0: LEA RAX, [RBP]
0x18001e8a4: MOV qword ptr [RBP + -0x58], RAX
0x18001e8a8: MOV EBX, 0x1
0x18001e8ad: MOV dword ptr [RBP + -0x50], EBX
0x18001e8b0: MOV dword ptr [RBP + -0x4c], 0x4
0x18001e8b7: CALL qword ptr [0x180078288]
0x18001e8bd: MOV R8, RAX
0x18001e8c0: LEA RDX, [RBP + -0x60]
0x18001e8c4: LEA RCX, [RBP + 0x6d0]
0x18001e8cb: CALL qword ptr [0x1800782a0]
0x18001e8d1: TEST R12, R12
0x18001e8d4: JZ 0x18001e8e4
0x18001e8d6: MOVSD XMM1, qword ptr [0x18007c5a8]
0x18001e8de: MOV RCX, R13
0x18001e8e1: CALL R12
0x18001e8e4: MOV EBX, ESI
0x18001e8e6: JMP 0x18001f871
0x18001e8eb: MOV RAX, qword ptr [RDI + 0xd40]
0x18001e8f2: MOV qword ptr [RDI + 0x810], RAX
0x18001e8f9: XOR EDX, EDX
0x18001e8fb: LEA R8D, [RDX + 0x68]
0x18001e8ff: LEA RCX, [RBP + 0x590]
0x18001e906: CALL 0x180072a62
0x18001e90b: LEA RDX, [RDI + 0x8e0]
0x18001e912: LEA RCX, [RBP + 0x590]
0x18001e919: CALL 0x1800013f0
0x18001e91e: MOV EAX, dword ptr [RDI + 0x940]
0x18001e924: MOV dword ptr [RBP + 0x5f0], EAX
0x18001e92a: XOR EDX, EDX
0x18001e92c: LEA R8D, [RDX + 0x68]
0x18001e930: LEA RCX, [RBP + 0x600]
0x18001e937: CALL 0x180072a62
0x18001e93c: LEA RDX, [RDI + 0x948]
0x18001e943: LEA RCX, [RBP + 0x600]
0x18001e94a: CALL 0x1800013f0
0x18001e94f: MOV EAX, dword ptr [RDI + 0x9a8]
0x18001e955: MOV dword ptr [RBP + 0x660], EAX
0x18001e95b: XOR EDX, EDX
0x18001e95d: LEA R8D, [RDX + 0x68]
0x18001e961: LEA RCX, [RBP + 0x520]
0x18001e968: CALL 0x180072a62
0x18001e96d: LEA RDX, [RDI + 0x9b0]
0x18001e974: LEA RCX, [RBP + 0x520]
0x18001e97b: CALL 0x1800013f0
0x18001e980: MOV EAX, dword ptr [RDI + 0xa10]
0x18001e986: MOV dword ptr [RBP + 0x580], EAX
0x18001e98c: XOR EDX, EDX
0x18001e98e: LEA R8D, [RDX + 0x68]
0x18001e992: LEA RCX, [RBP + 0x4b0]
0x18001e999: CALL 0x180072a62
0x18001e99e: LEA RDX, [RDI + 0xa18]
0x18001e9a5: LEA RCX, [RBP + 0x4b0]
0x18001e9ac: CALL 0x1800013f0
0x18001e9b1: MOV EAX, dword ptr [RDI + 0xa78]
0x18001e9b7: MOV dword ptr [RBP + 0x510], EAX
0x18001e9bd: XOR EDX, EDX
0x18001e9bf: LEA R8D, [RDX + 0x68]
0x18001e9c3: LEA RCX, [RBP + 0x280]
0x18001e9ca: CALL 0x180072a62
0x18001e9cf: LEA RDX, [RDI + 0xa80]
0x18001e9d6: LEA RCX, [RBP + 0x280]
0x18001e9dd: CALL 0x1800013f0
0x18001e9e2: MOV EAX, dword ptr [RDI + 0xae0]
0x18001e9e8: MOV dword ptr [RBP + 0x2e0], EAX
0x18001e9ee: LEA RAX, [RBP + 0x280]
0x18001e9f5: MOV qword ptr [RSP + 0x28], RAX
0x18001e9fa: LEA RAX, [RBP + 0x4b0]
0x18001ea01: MOV qword ptr [RSP + 0x20], RAX
0x18001ea06: LEA R9, [RBP + 0x520]
0x18001ea0d: LEA R8, [RBP + 0x600]
0x18001ea14: LEA RDX, [RBP + 0x590]
0x18001ea1b: MOV RCX, RDI
0x18001ea1e: CALL 0x180020450
0x18001ea23: XOR EAX, EAX
0x18001ea25: MOV dword ptr [RSP + 0x20], EAX
0x18001ea29: XOR R9D, R9D
0x18001ea2c: XOR R8D, R8D
0x18001ea2f: XOR EDX, EDX
0x18001ea31: LEA RCX, [0x18007bed0]
0x18001ea38: CALL 0x180001d80
0x18001ea3d: TEST R12, R12
0x18001ea40: JZ 0x18001ea63
0x18001ea42: MOVSD XMM1, qword ptr [0x18007c558]
0x18001ea4a: MOV RCX, R13
0x18001ea4d: CALL R12
0x18001ea50: TEST EAX, EAX
0x18001ea52: SETNZ AL
0x18001ea55: TEST AL, AL
0x18001ea57: JZ 0x18001ea63
0x18001ea59: MOV EBX, 0x19
0x18001ea5e: JMP 0x18001f7b3
0x18001ea63: XOR EDX, EDX
0x18001ea65: LEA R8D, [RDX + 0x68]
0x18001ea69: LEA RCX, [RBP + 0x150]
0x18001ea70: CALL 0x180072a62
0x18001ea75: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x18001ea7d: MOVDQA xmmword ptr [RBP + 0x150], XMM0
0x18001ea85: XORPS XMM1, XMM1
0x18001ea88: MOVDQA xmmword ptr [RBP + 0x160], XMM1
0x18001ea90: XORPS XMM0, XMM0
0x18001ea93: MOVDQA xmmword ptr [RBP + 0x170], XMM0
0x18001ea9b: MOVDQA xmmword ptr [RBP + 0x180], XMM1
0x18001eaa3: LEA RAX, [RBP + 0x158]
0x18001eaaa: MOV qword ptr [RBP + 0x190], RAX
0x18001eab1: LEA RAX, [RBP + 0x1a0]
0x18001eab8: MOV qword ptr [RBP + 0x198], RAX
0x18001eabf: XOR R13D, R13D
0x18001eac2: MOV qword ptr [RBP + 0x1a8], R13
0x18001eac9: MOV qword ptr [RBP + 0x1a0], R13
0x18001ead0: MOV dword ptr [RBP + 0x1b0], R13D
0x18001ead7: XOR EDX, EDX
0x18001ead9: LEA R8D, [R13 + 0x68]
0x18001eadd: LEA RCX, [RBP + 0x210]
0x18001eae4: CALL 0x180072a62
0x18001eae9: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x18001eaf1: MOVDQA xmmword ptr [RBP + 0x210], XMM0
0x18001eaf9: XORPS XMM1, XMM1
0x18001eafc: MOVDQA xmmword ptr [RBP + 0x220], XMM1
0x18001eb04: XORPS XMM0, XMM0
0x18001eb07: MOVDQA xmmword ptr [RBP + 0x230], XMM0
0x18001eb0f: MOVDQA xmmword ptr [RBP + 0x240], XMM1
0x18001eb17: LEA RAX, [RBP + 0x218]
0x18001eb1e: MOV qword ptr [RBP + 0x250], RAX
0x18001eb25: LEA RAX, [RBP + 0x260]
0x18001eb2c: MOV qword ptr [RBP + 0x258], RAX
0x18001eb33: MOV qword ptr [RBP + 0x268], R13
0x18001eb3a: MOV qword ptr [RBP + 0x260], R13
0x18001eb41: MOV dword ptr [RBP + 0x270], R13D
0x18001eb48: XOR EDX, EDX
0x18001eb4a: LEA R8D, [R13 + 0x68]
0x18001eb4e: LEA RCX, [RBP + 0x3d0]
0x18001eb55: CALL 0x180072a62
0x18001eb5a: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x18001eb62: MOVDQA xmmword ptr [RBP + 0x3d0], XMM0
0x18001eb6a: XORPS XMM1, XMM1
0x18001eb6d: MOVDQA xmmword ptr [RBP + 0x3e0], XMM1
0x18001eb75: XORPS XMM0, XMM0
0x18001eb78: MOVDQA xmmword ptr [RBP + 0x3f0], XMM0
0x18001eb80: MOVDQA xmmword ptr [RBP + 0x400], XMM1
0x18001eb88: LEA RAX, [RBP + 0x3d8]
0x18001eb8f: MOV qword ptr [RBP + 0x410], RAX
0x18001eb96: LEA RAX, [RBP + 0x420]
0x18001eb9d: MOV qword ptr [RBP + 0x418], RAX
0x18001eba4: MOV qword ptr [RBP + 0x428], R13
0x18001ebab: MOV qword ptr [RBP + 0x420], R13
0x18001ebb2: MOV dword ptr [RBP + 0x430], R13D
0x18001ebb9: XOR EDX, EDX
0x18001ebbb: LEA R8D, [R13 + 0x68]
0x18001ebbf: LEA RCX, [RBP + 0x2f0]
0x18001ebc6: CALL 0x180072a62
0x18001ebcb: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x18001ebd3: MOVDQA xmmword ptr [RBP + 0x2f0], XMM0
0x18001ebdb: XORPS XMM1, XMM1
0x18001ebde: MOVDQA xmmword ptr [RBP + 0x300], XMM1
0x18001ebe6: XORPS XMM0, XMM0
0x18001ebe9: MOVDQA xmmword ptr [RBP + 0x310], XMM0
0x18001ebf1: MOVDQA xmmword ptr [RBP + 0x320], XMM1
0x18001ebf9: LEA RAX, [RBP + 0x2f8]
0x18001ec00: MOV qword ptr [RBP + 0x330], RAX
0x18001ec07: LEA RAX, [RBP + 0x340]
0x18001ec0e: MOV qword ptr [RBP + 0x338], RAX
0x18001ec15: MOV qword ptr [RBP + 0x348], R13
0x18001ec1c: MOV qword ptr [RBP + 0x340], R13
0x18001ec23: MOV dword ptr [RBP + 0x350], R13D
0x18001ec2a: MOV RAX, qword ptr [RDI + 0xd48]
0x18001ec31: MOV qword ptr [RDI + 0x810], RAX
0x18001ec38: XOR EAX, EAX
0x18001ec3a: MOV qword ptr [RBP + 0x1c0], RAX
0x18001ec41: MOV qword ptr [RBP + 0x1c8], RAX
0x18001ec48: MOV qword ptr [RBP + 0x1d0], RAX
0x18001ec4f: MOV qword ptr [RBP + 0x1d8], RAX
0x18001ec56: MOV qword ptr [RBP + 0x1e0], RAX
0x18001ec5d: MOV qword ptr [RBP + 0x1c0], R14
0x18001ec64: MOV dword ptr [RBP + 0x1c8], 0x20
0x18001ec6e: MOV qword ptr [RBP + 0x1d0], R13
0x18001ec75: MOV RAX, qword ptr [R14 + 0x8]
0x18001ec79: ADD RAX, qword ptr [R14]
0x18001ec7c: MOV qword ptr [RBP + 0x1d8], RAX
0x18001ec83: LEA RAX, [RBP + 0x2f0]
0x18001ec8a: MOV qword ptr [RSP + 0x28], RAX
0x18001ec8f: LEA RAX, [RBP + 0x3d0]
0x18001ec96: MOV qword ptr [RSP + 0x20], RAX
0x18001ec9b: LEA R9, [RBP + 0x210]
0x18001eca2: LEA R8, [RBP + 0x150]
0x18001eca9: MOV RDX, RSI
0x18001ecac: LEA RCX, [RBP + 0x1c0]
0x18001ecb3: CALL 0x18002a880
0x18001ecb8: MOV dword ptr [RSP + 0x30], 0x3
0x18001ecc0: LEA RAX, [RBP + 0x150]
0x18001ecc7: MOV qword ptr [RSP + 0x28], RAX
0x18001eccc: LEA RAX, [RBP + 0x4b0]
0x18001ecd3: MOV qword ptr [RSP + 0x20], RAX
0x18001ecd8: LEA R9, [RBP + 0x280]
0x18001ecdf: LEA R8, [RBP + 0x520]
0x18001ece6: MOV RDX, R14
0x18001ece9: MOV RCX, RSI
0x18001ecec: CALL 0x180003550
0x18001ecf1: MOV dword ptr [RSP + 0x20], R13D
0x18001ecf6: XOR R9D, R9D
0x18001ecf9: XOR R8D, R8D
0x18001ecfc: XOR EDX, EDX
0x18001ecfe: LEA RCX, [0x18007bee8]
0x18001ed05: CALL 0x180001d80
0x18001ed0a: MOV qword ptr [RBP + -0x80], R13
0x18001ed0e: MOV dword ptr [RSP + 0x70], 0x2010000
0x18001ed16: LEA RAX, [RBP + 0x210]
0x18001ed1d: MOV qword ptr [RSP + 0x78], RAX
0x18001ed22: LEA RDX, [RSP + 0x70]
0x18001ed27: LEA RCX, [RBP + 0x590]
0x18001ed2e: CALL qword ptr [0x1800782b0]
0x18001ed34: MOV dword ptr [RSP + 0x28], R13D
0x18001ed39: LEA RAX, [RBP + 0x600]
0x18001ed40: MOV qword ptr [RSP + 0x20], RAX
0x18001ed45: LEA R9, [RBP + 0x210]
0x18001ed4c: MOV R8, R15
0x18001ed4f: MOV RDX, R14
0x18001ed52: MOV RCX, RSI
0x18001ed55: CALL 0x180012b70
0x18001ed5a: MOV dword ptr [RSP + 0x20], R13D
0x18001ed5f: XOR R9D, R9D
0x18001ed62: XOR R8D, R8D
0x18001ed65: XOR EDX, EDX
0x18001ed67: LEA RCX, [0x18007be50]
0x18001ed6e: CALL 0x180001d80
0x18001ed73: MOV dword ptr [RSP + 0x30], R13D
0x18001ed78: LEA RAX, [RBP + 0x150]
0x18001ed7f: MOV qword ptr [RSP + 0x28], RAX
0x18001ed84: LEA RAX, [RBP + 0x210]
0x18001ed8b: MOV qword ptr [RSP + 0x20], RAX
0x18001ed90: LEA R9, [RSP + 0x40]
0x18001ed95: MOV R8, R15
0x18001ed98: MOV RDX, R14
0x18001ed9b: MOV RCX, RSI
0x18001ed9e: CALL 0x18001aa20
0x18001eda3: TEST AL, AL
0x18001eda5: JNZ 0x18001f700
0x18001edab: MOV dword ptr [RSP + 0x20], R13D
0x18001edb0: LEA R9, [RBP + 0x210]
0x18001edb7: MOV R8, R15
0x18001edba: MOV RDX, R14
0x18001edbd: MOV RCX, RSI
0x18001edc0: CALL 0x18001c480
0x18001edc5: MOV R12D, EAX
0x18001edc8: MOV dword ptr [RSP + 0x20], R13D
0x18001edcd: XOR R9D, R9D
0x18001edd0: XOR R8D, R8D
0x18001edd3: XOR EDX, EDX
0x18001edd5: LEA RCX, [0x18007be68]
0x18001eddc: CALL 0x180001d80
0x18001ede1: TEST R12D, R12D
0x18001ede4: JNZ 0x18001f706
0x18001edea: MOV qword ptr [RBP + -0x8], R13
0x18001edee: MOV dword ptr [RBP + -0x18], 0x2010000
0x18001edf5: LEA RAX, [RBP + 0x3d0]
0x18001edfc: MOV qword ptr [RBP + -0x10], RAX
0x18001ee00: LEA RDX, [RBP + -0x18]
0x18001ee04: LEA RCX, [RBP + 0x590]
0x18001ee0b: CALL qword ptr [0x1800782b0]
0x18001ee11: MOV EBX, 0x1
0x18001ee16: MOV dword ptr [RSP + 0x28], EBX
0x18001ee1a: LEA RAX, [RBP + 0x600]
0x18001ee21: MOV qword ptr [RSP + 0x20], RAX
0x18001ee26: LEA R9, [RBP + 0x3d0]
0x18001ee2d: MOV R8, R15
0x18001ee30: MOV RDX, R14
0x18001ee33: MOV RCX, RSI
0x18001ee36: CALL 0x180012b70
0x18001ee3b: MOV dword ptr [RSP + 0x20], R13D
0x18001ee40: XOR R9D, R9D
0x18001ee43: XOR R8D, R8D
0x18001ee46: XOR EDX, EDX
0x18001ee48: LEA RCX, [0x18007be50]
0x18001ee4f: CALL 0x180001d80
0x18001ee54: MOV dword ptr [RSP + 0x30], EBX
0x18001ee58: LEA RAX, [RBP + 0x150]
0x18001ee5f: MOV qword ptr [RSP + 0x28], RAX
0x18001ee64: LEA RAX, [RBP + 0x3d0]
0x18001ee6b: MOV qword ptr [RSP + 0x20], RAX
0x18001ee70: LEA R9, [RSP + 0x40]
0x18001ee75: MOV R8, R15
0x18001ee78: MOV RDX, R14
0x18001ee7b: MOV RCX, RSI
0x18001ee7e: CALL 0x18001aa20
0x18001ee83: TEST AL, AL
0x18001ee85: JNZ 0x18001f700
0x18001ee8b: MOV dword ptr [RSP + 0x20], EBX
0x18001ee8f: LEA R9, [RBP + 0x3d0]
0x18001ee96: MOV R8, R15
0x18001ee99: MOV RDX, R14
0x18001ee9c: MOV RCX, RSI
0x18001ee9f: CALL 0x18001c480
0x18001eea4: MOV R12D, EAX
0x18001eea7: MOV dword ptr [RSP + 0x20], R13D
0x18001eeac: XOR R9D, R9D
0x18001eeaf: XOR R8D, R8D
0x18001eeb2: XOR EDX, EDX
0x18001eeb4: LEA RCX, [0x18007be68]
0x18001eebb: CALL 0x180001d80
0x18001eec0: TEST R12D, R12D
0x18001eec3: JNZ 0x18001f706
0x18001eec9: XOR EDX, EDX
0x18001eecb: LEA R8D, [R13 + 0x68]
0x18001eecf: LEA RCX, [RBP + 0x440]
0x18001eed6: CALL 0x180072a62
0x18001eedb: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x18001eee3: MOVDQA xmmword ptr [RBP + 0x440], XMM0
0x18001eeeb: XORPS XMM1, XMM1
0x18001eeee: MOVDQA xmmword ptr [RBP + 0x450], XMM1
0x18001eef6: XORPS XMM0, XMM0
0x18001eef9: MOVDQA xmmword ptr [RBP + 0x460], XMM0
0x18001ef01: MOVDQA xmmword ptr [RBP + 0x470], XMM1
0x18001ef09: LEA RAX, [RBP + 0x448]
0x18001ef10: MOV qword ptr [RBP + 0x480], RAX
0x18001ef17: LEA RAX, [RBP + 0x490]
0x18001ef1e: MOV qword ptr [RBP + 0x488], RAX
0x18001ef25: MOV qword ptr [RBP + 0x498], R13
0x18001ef2c: MOV qword ptr [RBP + 0x490], R13
0x18001ef33: MOV dword ptr [RBP + 0x4a0], R13D
0x18001ef3a: XOR EDX, EDX
0x18001ef3c: LEA R8D, [R13 + 0x68]
0x18001ef40: LEA RCX, [RBP + 0x360]
0x18001ef47: CALL 0x180072a62
0x18001ef4c: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x18001ef54: MOVDQA xmmword ptr [RBP + 0x360], XMM0
0x18001ef5c: XORPS XMM1, XMM1
0x18001ef5f: MOVDQA xmmword ptr [RBP + 0x370], XMM1
0x18001ef67: XORPS XMM0, XMM0
0x18001ef6a: MOVDQA xmmword ptr [RBP + 0x380], XMM0
0x18001ef72: MOVDQA xmmword ptr [RBP + 0x390], XMM1
0x18001ef7a: LEA RAX, [RBP + 0x368]
0x18001ef81: MOV qword ptr [RBP + 0x3a0], RAX
0x18001ef88: LEA RAX, [RBP + 0x3b0]
0x18001ef8f: MOV qword ptr [RBP + 0x3a8], RAX
0x18001ef96: MOV qword ptr [RBP + 0x3b8], R13
0x18001ef9d: MOV qword ptr [RBP + 0x3b0], R13
0x18001efa4: MOV dword ptr [RBP + 0x3c0], R13D
0x18001efab: XOR EDX, EDX
0x18001efad: LEA R8D, [R13 + 0x68]
0x18001efb1: LEA RCX, [RBP + 0x70]
0x18001efb5: CALL 0x180072a62
0x18001efba: MOV EAX, dword ptr [RBP + 0x280]
0x18001efc0: MOV dword ptr [RBP + 0x70], EAX
0x18001efc3: MOV EDX, dword ptr [RBP + 0x284]
0x18001efc9: MOV dword ptr [RBP + 0x74], EDX
0x18001efcc: MOV EAX, dword ptr [RBP + 0x288]
0x18001efd2: MOV dword ptr [RBP + 0x78], EAX
0x18001efd5: MOV EAX, dword ptr [RBP + 0x28c]
0x18001efdb: MOV dword ptr [RBP + 0x7c], EAX
0x18001efde: MOV RAX, qword ptr [RBP + 0x290]
0x18001efe5: MOV qword ptr [RBP + 0x80], RAX
0x18001efec: MOV RAX, qword ptr [RBP + 0x298]
0x18001eff3: MOV qword ptr [RBP + 0x88], RAX
0x18001effa: MOV RAX, qword ptr [RBP + 0x2a0]
0x18001f001: MOV qword ptr [RBP + 0x90], RAX
0x18001f008: MOV RAX, qword ptr [RBP + 0x2a8]
0x18001f00f: MOV qword ptr [RBP + 0x98], RAX
0x18001f016: MOV RAX, qword ptr [RBP + 0x2b0]
0x18001f01d: MOV qword ptr [RBP + 0xa0], RAX
0x18001f024: MOV RCX, qword ptr [RBP + 0x2b8]
0x18001f02b: MOV qword ptr [RBP + 0xa8], RCX
0x18001f032: LEA RAX, [RBP + 0x78]
0x18001f036: MOV qword ptr [RBP + 0xb0], RAX
0x18001f03d: LEA R8, [RBP + 0xc0]
0x18001f044: MOV qword ptr [RBP + 0xb8], R8
0x18001f04b: MOV qword ptr [RBP + 0xc8], R13
0x18001f052: MOV qword ptr [RBP + 0xc0], R13
0x18001f059: TEST RCX, RCX
0x18001f05c: JZ 0x18001f072
0x18001f05e: MOV EAX, EBX
0x18001f060: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18001f065: MOV EDX, dword ptr [RBP + 0x284]
0x18001f06b: MOV R8, qword ptr [RBP + 0xb8]
0x18001f072: CMP EDX, 0x2
0x18001f075: JG 0x18001f09c
0x18001f077: MOV RAX, qword ptr [RBP + 0x2c8]
0x18001f07e: MOV RCX, qword ptr [RAX]
0x18001f081: MOV qword ptr [R8], RCX
0x18001f084: MOV RAX, qword ptr [RBP + 0x2c8]
0x18001f08b: MOV RCX, qword ptr [RAX + 0x8]
0x18001f08f: MOV RAX, qword ptr [RBP + 0xb8]
0x18001f096: MOV qword ptr [RAX + 0x8], RCX
0x18001f09a: JMP 0x18001f0b1
0x18001f09c: MOV dword ptr [RBP + 0x74], R13D
0x18001f0a0: LEA RDX, [RBP + 0x280]
0x18001f0a7: LEA RCX, [RBP + 0x70]
0x18001f0ab: CALL qword ptr [0x180078290]
0x18001f0b1: MOV EAX, dword ptr [RBP + 0x2e0]
0x18001f0b7: MOV dword ptr [RBP + 0xd0], EAX
0x18001f0bd: XOR EAX, EAX
0x18001f0bf: MOV qword ptr [RBP + 0x1e8], RAX
0x18001f0c6: MOV qword ptr [RBP + 0x1f0], RAX
0x18001f0cd: MOV qword ptr [RBP + 0x1f8], RAX
0x18001f0d4: MOV qword ptr [RBP + 0x200], RAX
0x18001f0db: MOV qword ptr [RBP + 0x208], RAX
0x18001f0e2: MOV qword ptr [RBP + 0x1e8], R14
0x18001f0e9: MOV dword ptr [RBP + 0x1f0], 0x20
0x18001f0f3: MOV qword ptr [RBP + 0x1f8], R13
0x18001f0fa: MOV RAX, qword ptr [R14 + 0x8]
0x18001f0fe: ADD RAX, qword ptr [R14]
0x18001f101: MOV qword ptr [RBP + 0x200], RAX
0x18001f108: LEA R9, [RBP + 0x360]
0x18001f10f: LEA R8, [RBP + 0x440]
0x18001f116: MOV RDX, RSI
0x18001f119: LEA RCX, [RBP + 0x1e8]
0x18001f120: CALL 0x18002abf0
0x18001f125: LEA RAX, [RBP + 0x360]
0x18001f12c: MOV qword ptr [RSP + 0x28], RAX
0x18001f131: LEA RAX, [RBP + 0x440]
0x18001f138: MOV qword ptr [RSP + 0x20], RAX
0x18001f13d: LEA R9, [RBP + 0x3d0]
0x18001f144: MOV R8, R15
0x18001f147: MOV RDX, R14
0x18001f14a: MOV RCX, RSI
0x18001f14d: CALL 0x180004050
0x18001f152: MOV EAX, dword ptr [R15]
0x18001f155: OR R12D, 0xffffffff
0x18001f159: CMP EAX, dword ptr [RDI + 0x28]
0x18001f15c: JG 0x18001f51b
0x18001f162: XOR EDX, EDX
0x18001f164: LEA R8D, [R12 + 0x69]
0x18001f169: LEA RCX, [RBP + 0xe0]
0x18001f170: CALL 0x180072a62
0x18001f175: MOV EAX, dword ptr [RBP + 0x150]
0x18001f17b: MOV dword ptr [RBP + 0xe0], EAX
0x18001f181: MOV ECX, dword ptr [RBP + 0x154]
0x18001f187: MOV dword ptr [RBP + 0xe4], ECX
0x18001f18d: MOV EAX, dword ptr [RBP + 0x158]
0x18001f193: MOV dword ptr [RBP + 0xe8], EAX
0x18001f199: MOV EAX, dword ptr [RBP + 0x15c]
0x18001f19f: MOV dword ptr [RBP + 0xec], EAX
0x18001f1a5: MOV RAX, qword ptr [RBP + 0x160]
0x18001f1ac: MOV qword ptr [RBP + 0xf0], RAX
0x18001f1b3: MOV RAX, qword ptr [RBP + 0x168]
0x18001f1ba: MOV qword ptr [RBP + 0xf8], RAX
0x18001f1c1: MOV RAX, qword ptr [RBP + 0x170]
0x18001f1c8: MOV qword ptr [RBP + 0x100], RAX
0x18001f1cf: MOV RAX, qword ptr [RBP + 0x178]
0x18001f1d6: MOV qword ptr [RBP + 0x108], RAX
0x18001f1dd: MOV RAX, qword ptr [RBP + 0x180]
0x18001f1e4: MOV qword ptr [RBP + 0x110], RAX
0x18001f1eb: MOV RAX, qword ptr [RBP + 0x188]
0x18001f1f2: MOV qword ptr [RBP + 0x118], RAX
0x18001f1f9: LEA RDX, [RBP + 0xe8]
0x18001f200: MOV qword ptr [RBP + 0x120], RDX
0x18001f207: LEA RDX, [RBP + 0x130]
0x18001f20e: MOV qword ptr [RBP + 0x128], RDX
0x18001f215: MOV qword ptr [RBP + 0x138], R13
0x18001f21c: MOV qword ptr [RBP + 0x130], R13
0x18001f223: TEST RAX, RAX
0x18001f226: JZ 0x18001f23a
0x18001f228: XADD.LOCK dword ptr [RAX + 0x14], EBX
0x18001f22d: MOV ECX, dword ptr [RBP + 0x154]
0x18001f233: MOV RDX, qword ptr [RBP + 0x128]
0x18001f23a: CMP ECX, 0x2
0x18001f23d: JG 0x18001f264
0x18001f23f: MOV RAX, qword ptr [RBP + 0x198]
0x18001f246: MOV RCX, qword ptr [RAX]
0x18001f249: MOV qword ptr [RDX], RCX
0x18001f24c: MOV RAX, qword ptr [RBP + 0x198]
0x18001f253: MOV RCX, qword ptr [RAX + 0x8]
0x18001f257: MOV RAX, qword ptr [RBP + 0x128]
0x18001f25e: MOV qword ptr [RAX + 0x8], RCX
0x18001f262: JMP 0x18001f27f
0x18001f264: MOV dword ptr [RBP + 0xe4], R13D
0x18001f26b: LEA RDX, [RBP + 0x150]
0x18001f272: LEA RCX, [RBP + 0xe0]
0x18001f279: CALL qword ptr [0x180078290]
0x18001f27f: MOV EAX, dword ptr [RBP + 0x1b0]
0x18001f285: MOV dword ptr [RBP + 0x140], EAX
0x18001f28b: MOV EDX, dword ptr [RSI + 0x4d8]
0x18001f291: MOV EAX, R13D
0x18001f294: CMP dword ptr [RSI + 0x18], 0x8
0x18001f298: SETNZ AL
0x18001f29b: INC EAX
0x18001f29d: IMUL EAX, dword ptr [RSI + 0x4d4]
0x18001f2a4: IMUL EAX, dword ptr [RSI + 0x14]
0x18001f2a8: CDQE
0x18001f2aa: IMUL EDX, EAX
0x18001f2ad: LEA R9, [RDI + 0x878]
0x18001f2b4: LEA RAX, [RBP + 0x70]
0x18001f2b8: MOV qword ptr [RSP + 0x38], RAX
0x18001f2bd: LEA RAX, [RBP + 0x360]
0x18001f2c4: MOV qword ptr [RSP + 0x30], RAX
0x18001f2c9: LEA RAX, [RBP + 0x440]
0x18001f2d0: MOV qword ptr [RSP + 0x28], RAX
0x18001f2d5: MOV dword ptr [RSP + 0x20], EDX
0x18001f2d9: MOV R8, R15
0x18001f2dc: MOV RDX, R14
0x18001f2df: MOV RCX, RSI
0x18001f2e2: CALL 0x1800043f0
0x18001f2e7: MOV dword ptr [RSP + 0x20], R13D
0x18001f2ec: XOR R9D, R9D
0x18001f2ef: XOR R8D, R8D
0x18001f2f2: XOR EDX, EDX
0x18001f2f4: LEA RCX, [0x18007bef8]
0x18001f2fb: CALL 0x180001d80
0x18001f300: MOV RDI, qword ptr [RSP + 0x48]
0x18001f305: MOV RBX, qword ptr [RSP + 0x40]
0x18001f30a: TEST RBX, RBX
0x18001f30d: JZ 0x18001f32f
0x18001f30f: MOVSD XMM1, qword ptr [0x18007c598]
0x18001f317: MOV RCX, RDI
0x18001f31a: CALL RBX
0x18001f31c: TEST EAX, EAX
0x18001f31e: SETNZ AL
0x18001f321: TEST AL, AL
0x18001f323: JZ 0x18001f32f
0x18001f325: MOV EBX, 0x19
0x18001f32a: JMP 0x18001f487
0x18001f32f: MOV dword ptr [RSP + 0x30], 0x4
0x18001f337: LEA RAX, [RBP + 0xe0]
0x18001f33e: MOV qword ptr [RSP + 0x28], RAX
0x18001f343: LEA RAX, [RBP + 0x4b0]
0x18001f34a: MOV qword ptr [RSP + 0x20], RAX
0x18001f34f: LEA R9, [RBP + 0x70]
0x18001f353: LEA R8, [RBP + 0x520]
0x18001f35a: MOV RDX, R14
0x18001f35d: MOV RCX, RSI
0x18001f360: CALL 0x180003550
0x18001f365: MOV dword ptr [RSP + 0x20], R13D
0x18001f36a: XOR R9D, R9D
0x18001f36d: XOR R8D, R8D
0x18001f370: XOR EDX, EDX
0x18001f372: LEA RCX, [0x18007bee8]
0x18001f379: CALL 0x180001d80
0x18001f37e: MOV qword ptr [RBP + -0x68], R13
0x18001f382: MOV dword ptr [RBP + -0x78], 0x2010000
0x18001f389: LEA RAX, [RBP + 0x670]
0x18001f390: MOV qword ptr [RBP + -0x70], RAX
0x18001f394: CMP dword ptr [RSI + 0x18], 0x8
0x18001f398: MOV R8D, 0x2
0x18001f39e: CMOVZ R8D, R13D
0x18001f3a2: MOVSD qword ptr [RSP + 0x20], XMM6
0x18001f3a8: MOVSD XMM3, qword ptr [0x18007c538]
0x18001f3b0: LEA RDX, [RBP + -0x78]
0x18001f3b4: LEA RCX, [RBP + 0xe0]
0x18001f3bb: CALL qword ptr [0x1800782a8]
0x18001f3c1: MOV RAX, qword ptr [RBP + 0x300]
0x18001f3c8: MOV qword ptr [RSP + 0x50], RAX
0x18001f3cd: MOV EAX, dword ptr [RBP + 0x340]
0x18001f3d3: MOV dword ptr [RSP + 0x58], EAX
0x18001f3d7: MOVAPS XMM0, xmmword ptr [RSP + 0x50]
0x18001f3dc: MOVDQA xmmword ptr [RBP + 0x20], XMM0
0x18001f3e1: MOV RAX, qword ptr [RBP + 0x220]
0x18001f3e8: MOV qword ptr [RSP + 0x60], RAX
0x18001f3ed: MOV EAX, dword ptr [RBP + 0x260]
0x18001f3f3: MOV dword ptr [RSP + 0x68], EAX
0x18001f3f7: MOVAPS XMM0, xmmword ptr [RSP + 0x60]
0x18001f3fc: MOVDQA xmmword ptr [RBP + 0x40], XMM0
0x18001f401: MOVUPS XMM0, xmmword ptr [RSI + 0x580]
0x18001f408: MOVAPS xmmword ptr [RBP + 0x50], XMM0
0x18001f40c: MOVUPS XMM1, xmmword ptr [RSI + 0x590]
0x18001f413: MOVAPS xmmword ptr [RBP + 0x60], XMM1
0x18001f417: LEA RAX, [RBP + 0x20]
0x18001f41b: MOV qword ptr [RSP + 0x20], RAX
0x18001f420: LEA R9, [RBP + 0x40]
0x18001f424: LEA R8, [RBP + 0x50]
0x18001f428: MOV EDX, dword ptr [RSI + 0x4]
0x18001f42b: MOV ECX, dword ptr [RSI]
0x18001f42d: CALL 0x18001ba70
0x18001f432: MOV qword ptr [RBP + -0x20], R13
0x18001f436: MOV dword ptr [RBP + -0x30], 0x2010000
0x18001f43d: LEA RAX, [RBP + 0x6d0]
0x18001f444: MOV qword ptr [RBP + -0x28], RAX
0x18001f448: LEA RDX, [RBP + -0x30]
0x18001f44c: LEA RCX, [RBP + 0x2f0]
0x18001f453: CALL qword ptr [0x1800782b0]
0x18001f459: MOV dword ptr [RSP + 0x20], R13D
0x18001f45e: XOR R9D, R9D
0x18001f461: XOR R8D, R8D
0x18001f464: XOR EDX, EDX
0x18001f466: LEA RCX, [0x18007bf08]
0x18001f46d: CALL 0x180001d80
0x18001f472: TEST RBX, RBX
0x18001f475: JZ 0x18001f484
0x18001f477: MOVSD XMM1, qword ptr [0x18007c5a8]
0x18001f47f: MOV RCX, RDI
0x18001f482: CALL RBX
0x18001f484: MOV EBX, R13D
0x18001f487: MOV RCX, qword ptr [RBP + 0x118]
0x18001f48e: TEST RCX, RCX
0x18001f491: JZ 0x18001f4ad
0x18001f493: MOV EAX, R12D
0x18001f496: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18001f49b: CMP EAX, 0x1
0x18001f49e: JNZ 0x18001f4ad
0x18001f4a0: LEA RCX, [RBP + 0xe0]
0x18001f4a7: CALL qword ptr [0x180078400]
0x18001f4ad: MOV qword ptr [RBP + 0x118], R13
0x18001f4b4: MOV qword ptr [RBP + 0xf0], R13
0x18001f4bb: MOV qword ptr [RBP + 0x108], R13
0x18001f4c2: MOV qword ptr [RBP + 0x100], R13
0x18001f4c9: MOV qword ptr [RBP + 0xf8], R13
0x18001f4d0: MOV ECX, R13D
0x18001f4d3: CMP dword ptr [RBP + 0xe4], ECX
0x18001f4d9: JLE 0x18001f4f9
0x18001f4db: MOV RDX, R13
0x18001f4de: NOP
0x18001f4e0: MOV RAX, qword ptr [RBP + 0x120]
0x18001f4e7: MOV dword ptr [RDX + RAX*0x1], R13D
0x18001f4eb: INC ECX
0x18001f4ed: LEA RDX, [RDX + 0x4]
0x18001f4f1: CMP ECX, dword ptr [RBP + 0xe4]
0x18001f4f7: JL 0x18001f4e0
0x18001f4f9: LEA RAX, [RBP + 0x130]
0x18001f500: MOV RCX, qword ptr [RBP + 0x128]
0x18001f507: CMP RCX, RAX
0x18001f50a: JZ 0x18001f621
0x18001f510: CALL qword ptr [0x1800783e0]
0x18001f516: JMP 0x18001f621
0x18001f51b: MOV dword ptr [RDI + 0xb20], R13D
0x18001f522: MOV dword ptr [RDI + 0xb24], EAX
0x18001f528: MOV RAX, qword ptr [RDI + 0x810]
0x18001f52f: MOV qword ptr [RDI + 0xb28], RAX
0x18001f536: LEA RSI, [RDI + 0xb38]
0x18001f53d: LEA RAX, [RBP + 0x210]
0x18001f544: CMP RSI, RAX
0x18001f547: JZ 0x18001f558
0x18001f549: LEA RDX, [RBP + 0x210]
0x18001f550: MOV RCX, RSI
0x18001f553: CALL 0x180001780
0x18001f558: MOV EAX, dword ptr [RBP + 0x270]
0x18001f55e: MOV dword ptr [RSI + 0x60], EAX
0x18001f561: LEA RSI, [RDI + 0xba0]
0x18001f568: LEA RAX, [RBP + 0x2f0]
0x18001f56f: CMP RSI, RAX
0x18001f572: JZ 0x18001f583
0x18001f574: LEA RDX, [RBP + 0x2f0]
0x18001f57b: MOV RCX, RSI
0x18001f57e: CALL 0x180001780
0x18001f583: MOV EAX, dword ptr [RBP + 0x350]
0x18001f589: MOV dword ptr [RSI + 0x60], EAX
0x18001f58c: LEA RSI, [RDI + 0xc08]
0x18001f593: LEA RAX, [RBP + 0x440]
0x18001f59a: CMP RSI, RAX
0x18001f59d: JZ 0x18001f5ae
0x18001f59f: LEA RDX, [RBP + 0x440]
0x18001f5a6: MOV RCX, RSI
0x18001f5a9: CALL 0x180001780
0x18001f5ae: MOV EAX, dword ptr [RBP + 0x4a0]
0x18001f5b4: MOV dword ptr [RSI + 0x60], EAX
0x18001f5b7: LEA RSI, [RDI + 0xc70]
0x18001f5be: LEA RAX, [RBP + 0x360]
0x18001f5c5: CMP RSI, RAX
0x18001f5c8: JZ 0x18001f5d9
0x18001f5ca: LEA RDX, [RBP + 0x360]
0x18001f5d1: MOV RCX, RSI
0x18001f5d4: CALL 0x180001780
0x18001f5d9: MOV EAX, dword ptr [RBP + 0x3c0]
0x18001f5df: MOV dword ptr [RSI + 0x60], EAX
0x18001f5e2: ADD RDI, 0xcd8
0x18001f5e9: LEA RAX, [RBP + 0x70]
0x18001f5ed: CMP RDI, RAX
0x18001f5f0: JZ 0x18001f5fe
0x18001f5f2: LEA RDX, [RBP + 0x70]
0x18001f5f6: MOV RCX, RDI
0x18001f5f9: CALL 0x180001780
0x18001f5fe: MOV EAX, dword ptr [RBP + 0xd0]
0x18001f604: MOV dword ptr [RDI + 0x60], EAX
0x18001f607: MOV RAX, qword ptr [RSP + 0x40]
0x18001f60c: TEST RAX, RAX
0x18001f60f: JZ 0x18001f621
0x18001f611: MOVSD XMM1, qword ptr [0x18007c5a8]
0x18001f619: MOV RCX, qword ptr [RSP + 0x48]
0x18001f61e: CALL RAX
0x18001f620: NOP
0x18001f621: MOV RCX, qword ptr [RBP + 0x1e8]
0x18001f628: MOV RAX, qword ptr [RBP + 0x1f8]
0x18001f62f: SUB qword ptr [RCX + 0x8], RAX
0x18001f633: MOV RAX, qword ptr [RBP + 0xa8]
0x18001f63a: TEST RAX, RAX
0x18001f63d: JZ 0x18001f655
0x18001f63f: XADD.LOCK dword ptr [RAX + 0x14], R12D
0x18001f645: CMP R12D, 0x1
0x18001f649: JNZ 0x18001f655
0x18001f64b: LEA RCX, [RBP + 0x70]
0x18001f64f: CALL qword ptr [0x180078400]
0x18001f655: MOV qword ptr [RBP + 0xa8], R13
0x18001f65c: MOV qword ptr [RBP + 0x80], R13
0x18001f663: MOV qword ptr [RBP + 0x98], R13
0x18001f66a: MOV qword ptr [RBP + 0x90], R13
0x18001f671: MOV qword ptr [RBP + 0x88], R13
0x18001f678: MOV ECX, R13D
0x18001f67b: CMP dword ptr [RBP + 0x74], ECX
0x18001f67e: JLE 0x18001f699
0x18001f680: MOV RDX, R13
0x18001f683: MOV RAX, qword ptr [RBP + 0xb0]
0x18001f68a: MOV dword ptr [RDX + RAX*0x1], R13D
0x18001f68e: INC ECX
0x18001f690: LEA RDX, [RDX + 0x4]
0x18001f694: CMP ECX, dword ptr [RBP + 0x74]
0x18001f697: JL 0x18001f683
0x18001f699: LEA RAX, [RBP + 0xc0]
0x18001f6a0: MOV RCX, qword ptr [RBP + 0xb8]
0x18001f6a7: CMP RCX, RAX
0x18001f6aa: JZ 0x18001f6b3
0x18001f6ac: CALL qword ptr [0x1800783e0]
0x18001f6b2: NOP
0x18001f6b3: LEA RCX, [RBP + 0x360]
0x18001f6ba: CALL 0x180001900
0x18001f6bf: LEA RAX, [RBP + 0x3b0]
0x18001f6c6: MOV RCX, qword ptr [RBP + 0x3a8]
0x18001f6cd: CMP RCX, RAX
0x18001f6d0: JZ 0x18001f6d9
0x18001f6d2: CALL qword ptr [0x1800783e0]
0x18001f6d8: NOP
0x18001f6d9: LEA RCX, [RBP + 0x440]
0x18001f6e0: CALL 0x180001900
0x18001f6e5: LEA RAX, [RBP + 0x490]
0x18001f6ec: MOV RCX, qword ptr [RBP + 0x488]
0x18001f6f3: CMP RCX, RAX
0x18001f6f6: JZ 0x18001f709
0x18001f6f8: CALL qword ptr [0x1800783e0]
0x18001f6fe: JMP 0x18001f709
0x18001f700: MOV R12D, 0x19
0x18001f706: MOV EBX, R12D
0x18001f709: MOV RCX, qword ptr [RBP + 0x1c0]
0x18001f710: MOV RAX, qword ptr [RBP + 0x1d0]
0x18001f717: SUB qword ptr [RCX + 0x8], RAX
0x18001f71b: LEA RCX, [RBP + 0x2f0]
0x18001f722: CALL 0x180001900
0x18001f727: LEA RAX, [RBP + 0x340]
0x18001f72e: MOV RCX, qword ptr [RBP + 0x338]
0x18001f735: CMP RCX, RAX
0x18001f738: JZ 0x18001f741
0x18001f73a: CALL qword ptr [0x1800783e0]
0x18001f740: NOP
0x18001f741: LEA RCX, [RBP + 0x3d0]
0x18001f748: CALL 0x180001900
0x18001f74d: LEA RAX, [RBP + 0x420]
0x18001f754: MOV RCX, qword ptr [RBP + 0x418]
0x18001f75b: CMP RCX, RAX
0x18001f75e: JZ 0x18001f767
0x18001f760: CALL qword ptr [0x1800783e0]
0x18001f766: NOP
0x18001f767: LEA RCX, [RBP + 0x210]
0x18001f76e: CALL 0x180001900
0x18001f773: LEA RAX, [RBP + 0x260]
0x18001f77a: MOV RCX, qword ptr [RBP + 0x258]
0x18001f781: CMP RCX, RAX
0x18001f784: JZ 0x18001f78d
0x18001f786: CALL qword ptr [0x1800783e0]
0x18001f78c: NOP
0x18001f78d: LEA RCX, [RBP + 0x150]
0x18001f794: CALL 0x180001900
0x18001f799: LEA RAX, [RBP + 0x1a0]
0x18001f7a0: MOV RCX, qword ptr [RBP + 0x198]
0x18001f7a7: CMP RCX, RAX
0x18001f7aa: JZ 0x18001f7b3
0x18001f7ac: CALL qword ptr [0x1800783e0]
0x18001f7b2: NOP
0x18001f7b3: LEA RCX, [RBP + 0x280]
0x18001f7ba: CALL 0x180001900
0x18001f7bf: LEA RAX, [RBP + 0x2d0]
0x18001f7c6: MOV RCX, qword ptr [RBP + 0x2c8]
0x18001f7cd: CMP RCX, RAX
0x18001f7d0: JZ 0x18001f7d9
0x18001f7d2: CALL qword ptr [0x1800783e0]
0x18001f7d8: NOP
0x18001f7d9: LEA RCX, [RBP + 0x4b0]
0x18001f7e0: CALL 0x180001900
0x18001f7e5: LEA RAX, [RBP + 0x500]
0x18001f7ec: MOV RCX, qword ptr [RBP + 0x4f8]
0x18001f7f3: CMP RCX, RAX
0x18001f7f6: JZ 0x18001f7ff
0x18001f7f8: CALL qword ptr [0x1800783e0]
0x18001f7fe: NOP
0x18001f7ff: LEA RCX, [RBP + 0x520]
0x18001f806: CALL 0x180001900
0x18001f80b: LEA RAX, [RBP + 0x570]
0x18001f812: MOV RCX, qword ptr [RBP + 0x568]
0x18001f819: CMP RCX, RAX
0x18001f81c: JZ 0x18001f825
0x18001f81e: CALL qword ptr [0x1800783e0]
0x18001f824: NOP
0x18001f825: LEA RCX, [RBP + 0x600]
0x18001f82c: CALL 0x180001900
0x18001f831: LEA RAX, [RBP + 0x650]
0x18001f838: MOV RCX, qword ptr [RBP + 0x648]
0x18001f83f: CMP RCX, RAX
0x18001f842: JZ 0x18001f84b
0x18001f844: CALL qword ptr [0x1800783e0]
0x18001f84a: NOP
0x18001f84b: LEA RCX, [RBP + 0x590]
0x18001f852: CALL 0x180001900
0x18001f857: LEA RAX, [RBP + 0x5e0]
0x18001f85e: MOV RCX, qword ptr [RBP + 0x5d8]
0x18001f865: CMP RCX, RAX
0x18001f868: JZ 0x18001f871
0x18001f86a: CALL qword ptr [0x1800783e0]
0x18001f870: NOP
0x18001f871: LEA RCX, [RBP + 0x670]
0x18001f878: CALL 0x180001900
0x18001f87d: LEA RAX, [RBP + 0x6c0]
0x18001f884: MOV RCX, qword ptr [RBP + 0x6b8]
0x18001f88b: CMP RCX, RAX
0x18001f88e: JZ 0x18001f897
0x18001f890: CALL qword ptr [0x1800783e0]
0x18001f896: NOP
0x18001f897: LEA RCX, [RBP + 0x6d0]
0x18001f89e: CALL 0x180001900
0x18001f8a3: LEA RAX, [RBP + 0x720]
0x18001f8aa: MOV RCX, qword ptr [RBP + 0x718]
0x18001f8b1: CMP RCX, RAX
0x18001f8b4: JZ 0x18001f8bc
0x18001f8b6: CALL qword ptr [0x1800783e0]
0x18001f8bc: MOV EAX, EBX
0x18001f8be: MOV RCX, qword ptr [RBP + 0x730]
0x18001f8c5: XOR RCX, RSP
0x18001f8c8: CALL 0x180072a30
0x18001f8cd: MOV RBX, qword ptr [RSP + 0x898]
0x18001f8d5: MOVAPS XMM6, xmmword ptr [RSP + 0x840]
0x18001f8dd: ADD RSP, 0x850
0x18001f8e4: POP R15
0x18001f8e6: POP R14
0x18001f8e8: POP R13
0x18001f8ea: POP R12
0x18001f8ec: POP RDI
0x18001f8ed: POP RSI
0x18001f8ee: POP RBP
0x18001f8ef: RET
```

## procedure_pseudo_code {'procedure': '0x180024770'}
Evidence: `ev_a9a718742f815bebfd1b9c1f7c120ec6f4de44295cd2d82e417af07cc5e2bbfd`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180024770(longlong param_1,float param_2,int param_3,undefined8 param_4,longlong param_5,
                  longlong param_6,int param_7,longlong param_8,longlong param_9,longlong param_10,
                  longlong param_11,longlong param_12,longlong param_13,longlong param_14,
                  longlong *param_15,longlong *param_16)

{
  ushort uVar1;
  ushort uVar2;
  longlong lVar3;
  int iVar4;
  uint uVar5;
  float *pfVar6;
  float *pfVar7;
  float *pfVar8;
  uint uVar9;
  longlong lVar10;
  longlong lVar11;
  ushort *puVar12;
  float *pfVar13;
  int iVar14;
  uint uVar15;
  longlong lVar16;
  short sVar17;
  longlong lVar18;
  float *pfVar19;
  longlong lVar20;
  ulonglong uVar21;
  float fVar22;
  float fVar23;
  undefined1 auStackY_128 [32];
  int local_f8;
  int local_f4;
  undefined4 uStack_9c;
  longlong local_88;
  undefined8 uStack_80;
  float local_78 [2];
  longlong *local_70;
  undefined8 local_68;
  ulonglong local_60;
  float *local_58;
  undefined8 local_50;
  ulonglong local_48;
  
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_128;
  iVar14 = (int)param_4;
  lVar20 = (longlong)iVar14;
  local_70 = (longlong *)0x0;
  local_68 = 0;
  local_60 = 0;
  local_58 = (float *)0x0;
  local_50 = 0;
  iVar4 = omp_get_thread_num();
  local_70 = (longlong *)(*(longlong *)(param_1 + 0x30) + (longlong)iVar4 * 0x18);
  local_68 = CONCAT44(local_68._4_4_,0x20);
  pfVar19 = (float *)(*local_70 + local_70[1]);
  local_60 = (longlong)(iVar14 * 4) * 4 + 0x1fU & 0xffffffffffffffe0;
  lVar3 = local_70[1];
  local_70[1] = local_60 + lVar3;
  local_58 = pfVar19;
  if (local_60 + lVar3 <= (ulonglong)local_70[2]) {
    local_f8 = 0;
    fVar23 = param_2;
    if (0 < param_3) {
      do {
        local_f4 = 0;
        if (0 < (int)param_2) {
          do {
            lVar3 = (longlong)
                    ((int)((ulonglong)(longlong)*(int *)(param_5 + 8) >> 2) * local_f8 + local_f4) *
                    4;
            fVar22 = (float)rintf();
            lVar11 = (longlong)(int)fVar22;
            uVar1 = *(ushort *)(param_8 + lVar11 * 2);
            local_78[0] = (float)(uint)uVar1;
            lVar10 = 0;
            if (3 < lVar20) {
              lVar18 = (longlong)param_7;
              pfVar8 = pfVar19 + 2;
              puVar12 = (ushort *)(param_14 + 4 + (ulonglong)uVar1 * 2);
              lVar16 = (lVar20 - 4U >> 2) + 1;
              lVar10 = lVar16 * 4;
              do {
                pfVar8[-2] = *(float *)((ulonglong)puVar12[-2] * lVar18 + lVar3 + param_6);
                pfVar8[-1] = *(float *)((ulonglong)puVar12[-1] * lVar18 + lVar3 + param_6);
                *pfVar8 = *(float *)((ulonglong)*puVar12 * lVar18 + lVar3 + param_6);
                pfVar8[1] = *(float *)((ulonglong)puVar12[1] * lVar18 + lVar3 + param_6);
                puVar12 = puVar12 + 4;
                pfVar8 = pfVar8 + 4;
                lVar16 = lVar16 + -1;
              } while (lVar16 != 0);
            }
            if (lVar10 < lVar20) {
              puVar12 = (ushort *)(param_14 + ((int)(uint)uVar1 + lVar10) * 2);
              do {
                pfVar19[lVar10] =
                     *(float *)((ulonglong)*puVar12 * (longlong)param_7 + lVar3 + param_6);
                lVar10 = lVar10 + 1;
                puVar12 = puVar12 + 1;
              } while (lVar10 < lVar20);
            }
            uVar5 = (uint)*(ushort *)(param_9 + lVar11 * 2);
            uVar2 = *(ushort *)(param_10 + lVar11 * 2);
            if (uVar2 == 0) {
              local_78[0] = *pfVar19;
              lVar10 = 1;
              if (1 < lVar20) {
                if (3 < lVar20 + -1) {
                  pfVar8 = pfVar19 + 3;
                  lVar11 = (lVar20 - 5U >> 2) + 1;
                  lVar10 = lVar11 * 4 + 1;
                  do {
                    pfVar7 = local_78;
                    if (local_78[0] < pfVar8[-2]) {
                      pfVar7 = pfVar8 + -2;
                    }
                    local_78[0] = *pfVar7;
                    pfVar6 = local_78;
                    if (*pfVar7 < pfVar8[-1]) {
                      pfVar6 = pfVar8 + -1;
                    }
                    local_78[0] = *pfVar6;
                    pfVar7 = local_78;
                    if (*pfVar6 < *pfVar8) {
                      pfVar7 = pfVar8;
                    }
                    local_78[0] = *pfVar7;
                    pfVar6 = local_78;
                    if (*pfVar7 < pfVar8[1]) {
                      pfVar6 = pfVar8 + 1;
                    }
                    local_78[0] = *pfVar6;
                    pfVar8 = pfVar8 + 4;
                    lVar11 = lVar11 + -1;
                  } while (lVar11 != 0);
                }
                if (lVar10 < lVar20) {
                  pfVar8 = pfVar19 + lVar10;
                  lVar10 = lVar20 - lVar10;
                  do {
                    pfVar7 = local_78;
                    if (local_78[0] < *pfVar8) {
                      pfVar7 = pfVar8;
                    }
                    local_78[0] = *pfVar7;
                    pfVar8 = pfVar8 + 1;
                    lVar10 = lVar10 + -1;
                  } while (lVar10 != 0);
                }
              }
              sVar17 = 0;
            }
            else {
              uVar21 = (ulonglong)uVar5;
              pfVar8 = pfVar19 + lVar20;
              pfVar7 = pfVar19 + lVar20 * 2;
              pfVar6 = pfVar19 + lVar20 * 3;
              uStack_80 = CONCAT44(uStack_9c,(uint)uVar1);
              local_88 = param_11;
              FUN_180013fd0(iVar14,&local_88,(longlong)pfVar19,pfVar8,pfVar7,(longlong)pfVar6);
              iVar4 = (uint)*(ushort *)(param_12 + uVar21 * 2) - (int)local_78[0];
              fVar23 = *(float *)(param_13 + uVar21 * 4);
              if (iVar4 < 0) {
                iVar4 = 0;
                fVar23 = 0.0;
              }
              fVar23 = ((fVar23 * pfVar6[iVar4] + pfVar7[iVar4]) * fVar23 + pfVar8[iVar4]) * fVar23
                       + pfVar19[iVar4];
              uVar15 = 0;
              sVar17 = 0;
              uVar9 = 1;
              if (1 < uVar2) {
                pfVar13 = (float *)(param_13 + 4 + uVar21 * 4);
                puVar12 = (ushort *)(param_12 + uVar21 * 2);
                do {
                  puVar12 = puVar12 + 1;
                  iVar4 = (uint)*puVar12 - (int)local_78[0];
                  fVar22 = *pfVar13;
                  fVar22 = ((fVar22 * pfVar6[iVar4] + pfVar7[iVar4]) * fVar22 + pfVar8[iVar4]) *
                           fVar22 + pfVar19[iVar4];
                  if (fVar23 < fVar22) {
                    uVar15 = uVar9 & 0xffff;
                    fVar23 = fVar22;
                  }
                  sVar17 = (short)uVar15;
                  uVar9 = uVar9 + 1;
                  pfVar13 = pfVar13 + 1;
                } while ((int)uVar9 < (int)(uint)uVar2);
              }
            }
            *(float *)(lVar3 + *param_15) = (float)(int)((int)sVar17 + uVar5);
            *(float *)(lVar3 + *param_16) = fVar23;
            local_f4 = local_f4 + 1;
          } while (local_f4 < (int)param_2);
        }
        local_f8 = local_f8 + 1;
      } while (local_f8 < param_3);
    }
    local_70[1] = local_70[1] - local_60;
    return;
  }
                    /* WARNING: Subroutine does not return */
  FUN_180001b90("Allocating memory exceeding the limit ...",(longlong)iVar4 * 3,local_70,param_4);
}


```

## procedure_pseudo_code {'procedure': '0x18002d190'}
Evidence: `ev_ad026f025a7941b7339b6bcc78dbd20a984ae959125fbc982a9cc240c9895bb7`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_18002d190(Mat *param_1,uint param_2)

{
  int *piVar1;
  int iVar2;
  Mat *this;
  ulonglong uVar3;
  ulonglong uVar4;
  uint uVar5;
  ulonglong uVar6;
  undefined8 uVar7;
  undefined1 auStackY_1d8 [32];
  uint local_1a8;
  int local_1a4;
  Mat *local_1a0;
  undefined8 local_198;
  uint local_190 [26];
  undefined4 local_128;
  int local_124;
  undefined4 local_120;
  undefined4 local_11c;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  longlong local_f0;
  undefined4 *local_e8;
  undefined8 *local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined4 local_c8;
  undefined4 local_b8;
  int local_b4;
  undefined4 local_b0;
  undefined4 local_ac;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  longlong local_80;
  undefined4 *local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined4 local_58;
  ulonglong local_48;
  
  local_198 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_1d8;
  memset(&local_128,0,0x68);
  local_128 = *(undefined4 *)param_1;
  local_124 = *(int *)(param_1 + 4);
  local_120 = *(undefined4 *)(param_1 + 8);
  local_11c = *(undefined4 *)(param_1 + 0xc);
  local_118 = *(undefined8 *)(param_1 + 0x10);
  local_110 = *(undefined8 *)(param_1 + 0x18);
  local_108 = *(undefined8 *)(param_1 + 0x20);
  local_100 = *(undefined8 *)(param_1 + 0x28);
  local_f8 = *(undefined8 *)(param_1 + 0x30);
  local_f0 = *(longlong *)(param_1 + 0x38);
  local_e8 = &local_120;
  uVar4 = 0;
  local_d0 = 0;
  local_d8 = 0;
  if (local_f0 != 0) {
    LOCK();
    *(int *)(local_f0 + 0x14) = *(int *)(local_f0 + 0x14) + 1;
    UNLOCK();
  }
  local_e0 = &local_d8;
  if (*(int *)(param_1 + 4) < 3) {
    local_d8 = **(undefined8 **)(param_1 + 0x48);
    local_d0 = *(undefined8 *)(*(longlong *)(param_1 + 0x48) + 8);
  }
  else {
    local_124 = 0;
    cv::Mat::copySize((Mat *)&local_128,param_1);
  }
  local_c8 = *(undefined4 *)(param_1 + 0x60);
  FUN_18002c790((longlong)&local_128,param_2);
  if (local_f0 != 0) {
    LOCK();
    piVar1 = (int *)(local_f0 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_128);
    }
  }
  local_f0 = 0;
  local_118 = 0;
  local_100 = 0;
  local_108 = 0;
  local_110 = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < local_124) {
    do {
      *(undefined4 *)(uVar3 + (longlong)local_e8) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < local_124);
  }
  if (local_e0 != &local_d8) {
    cv::fastFree(local_e0);
  }
  iVar2 = **(int **)(param_1 + 0x40);
  uVar5 = (*(int **)(param_1 + 0x40))[1];
  uVar7 = 0x68;
  memset(&local_b8,0,0x68);
  local_1a8 = uVar5;
  local_1a4 = iVar2 + param_2 * 2;
  this = (Mat *)FUN_180011dc0(local_190,&local_1a8,uVar7,
                              *(longlong *)(param_1 + 0x10) -
                              (longlong)(int)param_2 * *(longlong *)(param_1 + 0x50),
                              (longlong)(int)*(longlong *)(param_1 + 0x50));
  local_b8 = *(undefined4 *)this;
  local_b4 = *(int *)(this + 4);
  local_b0 = *(undefined4 *)(this + 8);
  local_ac = *(undefined4 *)(this + 0xc);
  local_a8 = *(undefined8 *)(this + 0x10);
  local_a0 = *(undefined8 *)(this + 0x18);
  local_98 = *(undefined8 *)(this + 0x20);
  local_90 = *(undefined8 *)(this + 0x28);
  local_88 = *(undefined8 *)(this + 0x30);
  local_80 = *(longlong *)(this + 0x38);
  local_78 = &local_b0;
  local_70 = &local_68;
  local_60 = 0;
  local_68 = 0;
  if (local_80 != 0) {
    LOCK();
    *(int *)(local_80 + 0x14) = *(int *)(local_80 + 0x14) + 1;
    UNLOCK();
  }
  local_1a0 = this;
  if (*(int *)(this + 4) < 3) {
    local_68 = **(undefined8 **)(this + 0x48);
    local_60 = *(undefined8 *)(*(longlong *)(this + 0x48) + 8);
  }
  else {
    local_b4 = 0;
    cv::Mat::copySize((Mat *)&local_b8,this);
  }
  local_58 = 0;
  if (*(longlong *)(this + 0x38) != 0) {
    LOCK();
    piVar1 = (int *)(*(longlong *)(this + 0x38) + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(this);
    }
  }
  *(undefined8 *)(this + 0x38) = 0;
  *(undefined8 *)(this + 0x10) = 0;
  *(undefined8 *)(this + 0x28) = 0;
  *(undefined8 *)(this + 0x20) = 0;
  *(undefined8 *)(this + 0x18) = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < *(int *)(this + 4)) {
    do {
      *(undefined4 *)(uVar3 + *(longlong *)(this + 0x40)) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < *(int *)(this + 4));
  }
  if (*(Mat **)(this + 0x48) != this + 0x50) {
    cv::fastFree(*(Mat **)(this + 0x48));
  }
  FUN_18002f490((longlong)&local_b8,param_2);
  if (local_80 != 0) {
    LOCK();
    piVar1 = (int *)(local_80 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_b8);
    }
  }
  local_80 = 0;
  local_a8 = 0;
  local_90 = 0;
  local_98 = 0;
  local_a0 = 0;
  uVar3 = uVar4;
  if (0 < local_b4) {
    do {
      *(undefined4 *)((longlong)local_78 + uVar3) = 0;
      uVar5 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar5;
      uVar3 = uVar3 + 4;
    } while ((int)uVar5 < local_b4);
  }
  if (local_70 != &local_68) {
    cv::fastFree(local_70);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002f490'}
Evidence: `ev_ae66b9f69809a828d2874db7a04f49a37c1997c3bf8b300a2fab4fefc243353f`
```c

void FUN_18002f490(longlong param_1,int param_2)

{
  int iVar1;
  undefined4 uVar2;
  undefined4 *puVar3;
  undefined4 *puVar4;
  undefined4 *puVar5;
  longlong lVar6;
  longlong lVar7;
  int iVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  
  iVar8 = 0;
  iVar1 = **(int **)(param_1 + 0x40);
  lVar7 = (longlong)param_2;
  if (0 < iVar1) {
    lVar10 = (longlong)(*(int **)(param_1 + 0x40))[1];
    do {
      lVar6 = 0;
      lVar9 = (longlong)(*(int *)(param_1 + 0x50) * iVar8) + *(longlong *)(param_1 + 0x10);
      if (3 < lVar7) {
        lVar11 = (lVar7 - 4U >> 2) + 1;
        lVar6 = lVar11 * 4;
        puVar3 = (undefined4 *)(lVar9 + 4);
        puVar4 = (undefined4 *)(lVar9 + -8);
        puVar5 = (undefined4 *)(lVar9 + (lVar10 + -2) * 4);
        do {
          puVar4[1] = *puVar3;
          puVar3[lVar10 + -1] = puVar5[1];
          *puVar4 = puVar3[1];
          puVar3[lVar10] = *puVar5;
          puVar4[-1] = puVar3[2];
          puVar3[lVar10 + 1] = puVar5[-1];
          puVar4[-2] = puVar3[3];
          puVar3[lVar10 + 2] = puVar5[-2];
          lVar11 = lVar11 + -1;
          puVar3 = puVar3 + 4;
          puVar4 = puVar4 + -4;
          puVar5 = puVar5 + -4;
        } while (lVar11 != 0);
      }
      if (lVar6 < lVar7) {
        puVar3 = (undefined4 *)(lVar9 + ((lVar10 - lVar6) + -1) * 4);
        lVar11 = lVar7 - lVar6;
        puVar4 = (undefined4 *)(lVar9 + (lVar10 + lVar6) * 4);
        puVar5 = (undefined4 *)(lVar9 - (lVar6 * 4 + 4));
        do {
          *puVar5 = puVar4[1 - lVar10];
          uVar2 = *puVar3;
          puVar3 = puVar3 + -1;
          *puVar4 = uVar2;
          lVar11 = lVar11 + -1;
          puVar4 = puVar4 + 1;
          puVar5 = puVar5 + -1;
        } while (lVar11 != 0);
      }
      iVar8 = iVar8 + 1;
    } while (iVar8 < iVar1);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18000f040'}
Evidence: `ev_af114e000dad2a39ad9794db5ff1638af6e3ff9633460844957a550a14f3d37d`
```c

void FUN_18000f040(int param_1,int param_2,longlong param_3,longlong param_4,int param_5,
                  longlong param_6,longlong param_7,longlong param_8)

{
  float fVar1;
  byte bVar2;
  byte bVar3;
  undefined2 uVar4;
  undefined2 uVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  ulonglong uVar10;
  longlong lVar11;
  longlong lVar12;
  longlong lVar13;
  float fVar14;
  undefined1 uStack_97;
  undefined1 uStack_87;
  
  if (0 < param_2) {
    lVar12 = 0;
    do {
      lVar13 = 0;
      if (0 < (longlong)param_1) {
        lVar9 = 0;
        lVar11 = param_5 * lVar12;
        lVar8 = 0;
        do {
          uVar10 = (ulonglong)
                   *(ushort *)
                    (**(longlong **)(param_6 + 0x48) * lVar12 + lVar13 * 2 +
                    *(longlong *)(param_6 + 0x10));
          lVar6 = lVar13 * 4;
          lVar13 = lVar13 + 1;
          fVar1 = *(float *)(**(longlong **)(param_7 + 0x48) * lVar12 + lVar6 +
                            *(longlong *)(param_7 + 0x10));
          fVar14 = 1.0 - fVar1;
          lVar6 = *(longlong *)(param_4 + (ulonglong)*(ushort *)(param_3 + uVar10 * 2) * 8) + lVar8;
          lVar7 = *(int *)(param_8 + 0x50) * lVar12 + lVar9;
          lVar9 = lVar9 + 0xc;
          uVar4 = *(undefined2 *)(lVar6 + lVar11);
          uStack_97 = (byte)((ushort)uVar4 >> 8);
          bVar2 = *(byte *)(lVar6 + 2 + lVar11);
          lVar6 = *(longlong *)(param_4 + (ulonglong)*(ushort *)(param_3 + 2 + uVar10 * 2) * 8) +
                  lVar8;
          lVar8 = lVar8 + 3;
          uVar5 = *(undefined2 *)(lVar6 + lVar11);
          uStack_87 = (byte)((ushort)uVar5 >> 8);
          bVar3 = *(byte *)(lVar6 + 2 + lVar11);
          lVar6 = *(longlong *)(param_8 + 0x10);
          *(ulonglong *)(lVar7 + lVar6) =
               CONCAT44((float)uStack_97 * fVar14 + (float)uStack_87 * fVar1,
                        (float)(byte)uVar4 * fVar14 + (float)(byte)uVar5 * fVar1);
          *(float *)(lVar7 + 8 + lVar6) = (float)bVar2 * fVar14 + (float)bVar3 * fVar1;
        } while (lVar13 < param_1);
      }
      lVar12 = lVar12 + 1;
    } while (lVar12 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002d840'}
Evidence: `ev_b2bb6add63b2252d16ae78999b0c1dc8e226b1a376c1144e5f3d843768bbc673`
```c

void FUN_18002d840(int param_1,int param_2,int param_3,int param_4,float param_5,float param_6,
                  longlong *param_7,longlong *param_8,longlong *param_9)

{
  byte bVar1;
  undefined2 uVar2;
  int iVar3;
  longlong lVar4;
  int iVar5;
  longlong lVar6;
  longlong lVar7;
  int iVar8;
  longlong lVar9;
  int iVar10;
  int iVar11;
  longlong lVar12;
  longlong lVar13;
  byte bStack_67;
  int local_58;
  
  if (0 < param_2) {
    lVar13 = 0;
    local_58 = 0;
    do {
      iVar11 = 0;
      lVar4 = 0;
      if (3 < param_1) {
        lVar9 = 0;
        lVar12 = 0;
        iVar8 = 2;
        lVar4 = (ulonglong)((param_1 - 4U >> 2) + 1) << 2;
        iVar10 = (int)((float)local_58 * param_6);
        do {
          iVar5 = (int)((float)iVar11 * param_5);
          if (param_3 <= iVar5) {
            iVar5 = param_3 + -1;
          }
          iVar3 = iVar10;
          if (param_4 <= iVar10) {
            iVar3 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar3 + (longlong)iVar5 + *param_8) !=
              '\0') {
            lVar6 = (int)param_7[1] * lVar13 + lVar12;
            uVar2 = *(undefined2 *)(lVar6 + *param_7);
            bStack_67 = (byte)((ushort)uVar2 >> 8);
            bVar1 = *(byte *)(lVar6 + 2 + *param_7);
            lVar6 = *param_9;
            lVar7 = (int)param_9[1] * lVar13 + lVar9;
            *(ulonglong *)(lVar7 + lVar6) = CONCAT44((float)bStack_67,(float)(byte)uVar2);
            *(float *)(lVar7 + 8 + lVar6) = (float)bVar1;
          }
          iVar5 = (int)((float)(iVar8 + -1) * param_5);
          if (param_3 <= iVar5) {
            iVar5 = param_3 + -1;
          }
          iVar3 = iVar10;
          if (param_4 <= iVar10) {
            iVar3 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar3 + (longlong)iVar5 + *param_8) !=
              '\0') {
            lVar6 = (int)param_7[1] * lVar13 + lVar12;
            uVar2 = *(undefined2 *)(lVar6 + 3 + *param_7);
            bStack_67 = (byte)((ushort)uVar2 >> 8);
            bVar1 = *(byte *)(lVar6 + 5 + *param_7);
            lVar6 = *param_9;
            lVar7 = (int)param_9[1] * lVar13 + lVar9;
            *(ulonglong *)(lVar7 + 0xc + lVar6) = CONCAT44((float)bStack_67,(float)(byte)uVar2);
            *(float *)(lVar7 + 0x14 + lVar6) = (float)bVar1;
          }
          iVar5 = (int)((float)iVar8 * param_5);
          if (param_3 <= iVar5) {
            iVar5 = param_3 + -1;
          }
          iVar3 = iVar10;
          if (param_4 <= iVar10) {
            iVar3 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar3 + (longlong)iVar5 + *param_8) !=
              '\0') {
            lVar6 = (int)param_7[1] * lVar13 + lVar12;
            uVar2 = *(undefined2 *)(lVar6 + 6 + *param_7);
            bStack_67 = (byte)((ushort)uVar2 >> 8);
            bVar1 = *(byte *)(lVar6 + 8 + *param_7);
            lVar6 = *param_9;
            lVar7 = (int)param_9[1] * lVar13 + lVar9;
            *(ulonglong *)(lVar7 + 0x18 + lVar6) = CONCAT44((float)bStack_67,(float)(byte)uVar2);
            *(float *)(lVar7 + 0x20 + lVar6) = (float)bVar1;
          }
          iVar5 = (int)((float)(iVar8 + 1) * param_5);
          if (param_3 <= iVar5) {
            iVar5 = param_3 + -1;
          }
          iVar3 = iVar10;
          if (param_4 <= iVar10) {
            iVar3 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar3 + (longlong)iVar5 + *param_8) !=
              '\0') {
            lVar6 = (int)param_7[1] * lVar13 + lVar12;
            uVar2 = *(undefined2 *)(lVar6 + 9 + *param_7);
            bStack_67 = (byte)((ushort)uVar2 >> 8);
            bVar1 = *(byte *)(lVar6 + 0xb + *param_7);
            lVar6 = *param_9;
            lVar7 = (int)param_9[1] * lVar13 + lVar9;
            *(ulonglong *)(lVar7 + 0x24 + lVar6) = CONCAT44((float)bStack_67,(float)(byte)uVar2);
            *(float *)(lVar7 + 0x2c + lVar6) = (float)bVar1;
          }
          iVar11 = iVar11 + 4;
          lVar12 = lVar12 + 0xc;
          lVar9 = lVar9 + 0x30;
          iVar8 = iVar8 + 4;
        } while (iVar11 < param_1 + -3);
      }
      if (iVar11 < param_1) {
        lVar9 = lVar4 * 3;
        lVar4 = lVar4 * 0xc;
        do {
          iVar8 = (int)((float)iVar11 * param_5);
          if (param_3 <= iVar8) {
            iVar8 = param_3 + -1;
          }
          iVar10 = (int)((float)local_58 * param_6);
          if (param_4 <= (int)((float)local_58 * param_6)) {
            iVar10 = param_4 + -1;
          }
          if (*(char *)((longlong)(int)param_8[1] * (longlong)iVar10 + (longlong)iVar8 + *param_8)
              != '\0') {
            lVar12 = (int)param_7[1] * lVar13 + lVar9;
            uVar2 = *(undefined2 *)(lVar12 + *param_7);
            bStack_67 = (byte)((ushort)uVar2 >> 8);
            bVar1 = *(byte *)(lVar12 + 2 + *param_7);
            lVar12 = *param_9;
            lVar6 = (int)param_9[1] * lVar13 + lVar4;
            *(ulonglong *)(lVar6 + lVar12) = CONCAT44((float)bStack_67,(float)(byte)uVar2);
            *(float *)(lVar6 + 8 + lVar12) = (float)bVar1;
          }
          iVar11 = iVar11 + 1;
          lVar9 = lVar9 + 3;
          lVar4 = lVar4 + 0xc;
        } while (iVar11 < param_1);
      }
      local_58 = local_58 + 1;
      lVar13 = lVar13 + 1;
    } while (local_58 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180014890'}
Evidence: `ev_bb2e3c3e524897a281f424abbe687b7acd4355942d41c191c3f385a6fefc3501`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180014890(longlong param_1,longlong *param_2,int *param_3,longlong param_4,longlong param_5
                  ,longlong param_6,longlong param_7)

{
  float fVar1;
  float fVar2;
  float fVar3;
  uint uVar4;
  float *pfVar5;
  float *pfVar6;
  ulonglong uVar7;
  longlong *plVar8;
  int iVar9;
  float *pfVar10;
  longlong lVar11;
  longlong lVar12;
  longlong *plVar13;
  int iVar14;
  ulonglong uVar15;
  short *psVar16;
  uint uVar17;
  int iVar19;
  longlong lVar20;
  int *piVar21;
  longlong lVar22;
  longlong lVar23;
  float fVar24;
  float fVar25;
  float fVar26;
  undefined1 auStackY_2b8 [32];
  longlong in_stack_fffffffffffffdb0;
  int local_22c;
  float local_228;
  longlong local_218;
  undefined4 uStack_210;
  undefined4 uStack_20c;
  longlong local_208;
  longlong *local_200;
  int local_1f8 [2];
  int *local_1f0;
  ulonglong local_1e8;
  longlong local_1d8;
  undefined8 uStack_1d0;
  int local_1c8;
  longlong local_1c0;
  longlong local_1b8;
  short *local_1b0;
  longlong local_1a8;
  longlong local_1a0;
  longlong local_198;
  longlong local_188;
  undefined8 uStack_180;
  longlong local_178;
  ulonglong local_170;
  longlong local_168;
  undefined8 uStack_160;
  longlong local_158;
  undefined4 uStack_150;
  undefined4 uStack_14c;
  longlong local_148;
  longlong local_138;
  undefined4 uStack_130;
  undefined4 uStack_12c;
  longlong local_128;
  undefined4 uStack_120;
  undefined4 uStack_11c;
  longlong local_118;
  undefined4 uStack_110;
  undefined4 uStack_10c;
  longlong local_108;
  undefined4 uStack_100;
  undefined4 uStack_fc;
  longlong local_f8;
  undefined4 uStack_f0;
  undefined4 uStack_ec;
  longlong local_e8;
  undefined4 uStack_e0;
  undefined4 uStack_dc;
  longlong local_d8;
  undefined4 uStack_d0;
  undefined4 uStack_cc;
  longlong local_c8;
  undefined4 uStack_c0;
  undefined4 uStack_bc;
  longlong local_b8;
  undefined4 uStack_b0;
  undefined4 uStack_ac;
  undefined8 local_a8;
  longlong *local_a0;
  undefined8 local_98;
  longlong local_90;
  longlong local_88;
  undefined8 local_80;
  ulonglong local_78;
  ulonglong uVar18;
  
  local_a8 = 0xfffffffffffffffe;
  local_78 = DAT_18008b080 ^ (ulonglong)auStackY_2b8;
  local_148 = param_5;
  fVar3 = *(float *)(param_1 + 0x4dc);
  local_228 = *(float *)(param_1 + 0x4e0);
  local_22c = *param_3;
  iVar14 = param_3[1];
  uVar4 = *(uint *)(param_1 + 0x78);
  local_1c8 = (int)uVar4 / 2;
  iVar19 = iVar14 + -1;
  iVar9 = *(int *)(param_1 + 0x7c);
  pfVar5 = *(float **)(param_3 + 4);
  fVar1 = (float)param_3[0xb];
  local_80 = 0;
  local_98 = 0x20;
  uVar7 = 0;
  local_170 = 0;
  local_90 = 0;
  local_88 = param_2[1] + *param_2;
  local_208 = param_4;
  local_200 = param_2;
  local_1f0 = param_3;
  local_178 = param_1;
  local_a0 = param_2;
  FUN_180024560((longlong *)&local_a0,param_1,&local_1c0,local_22c);
  if (0 < iVar19) {
    uVar15 = uVar7;
    uVar18 = uVar7;
    pfVar6 = pfVar5;
    do {
      pfVar10 = pfVar6 + 1;
      *(float *)((local_1c0 - (longlong)pfVar5) + -4 + (longlong)pfVar10) = *pfVar10 - *pfVar6;
      for (; (longlong)uVar7 < (longlong)(iVar19 * iVar9 + 1); uVar7 = uVar7 + 1) {
        fVar24 = (float)(int)uVar15 * fVar1 + *pfVar5;
        if (*pfVar10 <= fVar24 && fVar24 != *pfVar10) break;
        *(short *)(local_1a0 + uVar7 * 2) = (short)uVar18;
        *(float *)(local_198 + uVar7 * 4) = fVar24 - *pfVar6;
        uVar15 = (ulonglong)((int)uVar15 + 1);
      }
      uVar17 = (int)uVar18 + 1;
      uVar18 = (ulonglong)uVar17;
      pfVar6 = pfVar10;
    } while ((int)uVar17 < iVar19);
  }
  lVar20 = (longlong)local_22c;
  lVar22 = local_198;
  lVar23 = local_1a0;
  if (0 < lVar20) {
    iVar14 = iVar14 - uVar4;
    local_1f8[0] = iVar14;
    local_22c = 0;
    lVar11 = local_1b8 - (longlong)local_1b0;
    lVar12 = local_1a8 - (longlong)local_1b0;
    local_1e8 = -(longlong)local_1b0;
    psVar16 = local_1b0;
    local_218 = lVar11;
    local_158 = lVar12;
    do {
      iVar9 = (uint)*(ushort *)(*(longlong *)(param_3 + 8) + local_1e8 + (longlong)psVar16) -
              local_1c8;
      local_218 = CONCAT44(local_218._4_4_,iVar9);
      plVar13 = (longlong *)&local_22c;
      if (-1 < iVar9) {
        plVar13 = &local_218;
      }
      plVar8 = (longlong *)local_1f8;
      if ((int)*plVar13 <= iVar14) {
        plVar8 = plVar13;
      }
      iVar9 = (int)*plVar8;
      local_218 = CONCAT44(local_218._4_4_,iVar9);
      fVar24 = *pfVar5;
      fVar25 = ceilf((pfVar5[iVar9] - fVar24) / fVar1);
      fVar2 = pfVar5[(longlong)(int)uVar4 + (longlong)iVar9 + -1];
      fVar26 = floorf((fVar2 - fVar24) / fVar1);
      for (iVar9 = (int)fVar26; fVar2 < (float)iVar9 * fVar1 + fVar24; iVar9 = iVar9 + -1) {
      }
      *(short *)(lVar11 + (longlong)psVar16) = (short)local_218;
      *psVar16 = (short)(int)fVar25;
      *(short *)(lVar12 + (longlong)psVar16) = ((short)iVar9 - (short)(int)fVar25) + 1;
      psVar16 = psVar16 + 1;
      lVar20 = lVar20 + -1;
      param_3 = local_1f0;
      lVar22 = local_198;
      lVar23 = local_1a0;
    } while (lVar20 != 0);
  }
  piVar21 = (int *)(local_178 + 0x244);
  local_1e8 = (ulonglong)
              ((longlong)(piVar21 + *(int *)(local_178 + 0x254)) + (3 - (longlong)piVar21)) >> 2;
  if (piVar21 + *(int *)(local_178 + 0x254) < piVar21) {
    local_1e8 = 0;
  }
  if (local_1e8 != 0) {
    do {
      lVar11 = (longlong)*piVar21;
      lVar20 = *(longlong *)(local_148 + lVar11 * 8);
      uVar17 = (int)*(undefined8 *)(lVar20 + 0x50) * (int)local_228 + 0x1fU & 0xffffffe0;
      if (*(char *)(local_178 + 0x24) == '\0') {
        if ((char)local_1f0[10] == '\0') {
          lVar11 = lVar11 * 0x68;
          local_1d8 = *(longlong *)(lVar11 + 0x10 + param_7);
          uStack_110 = (undefined4)*(undefined8 *)(lVar11 + 0x50 + param_7);
          uStack_1d0 = CONCAT44(uStack_10c,uStack_110);
          lVar12 = param_6 + lVar11;
          local_188 = *(longlong *)(lVar12 + 0x10);
          uStack_f0 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
          uStack_180 = CONCAT44(uStack_ec,uStack_f0);
          lVar11 = local_208 + lVar11;
          local_168 = *(longlong *)(lVar11 + 0x10);
          uStack_c0 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
          uStack_160 = CONCAT44(uStack_bc,uStack_c0);
          uVar7 = (ulonglong)(uint)local_228;
          local_118 = local_1d8;
          local_f8 = local_188;
          local_c8 = local_168;
          FUN_1800251b0((longlong)local_200,(int)fVar3,local_228,(ulonglong)uVar4,
                        (longlong)&local_168,*(longlong *)(lVar20 + 0x10),uVar17,local_1b8,
                        (longlong)local_1b0,local_1a8,local_1c0,lVar23,lVar22,
                        in_stack_fffffffffffffdb0,&local_188,&local_1d8);
        }
        else {
          lVar11 = lVar11 * 0x68;
          local_1d8 = *(longlong *)(lVar11 + 0x10 + param_7);
          uStack_b0 = (undefined4)*(undefined8 *)(lVar11 + 0x50 + param_7);
          uStack_1d0 = CONCAT44(uStack_ac,uStack_b0);
          lVar12 = param_6 + lVar11;
          local_188 = *(longlong *)(lVar12 + 0x10);
          uStack_100 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
          uStack_180 = CONCAT44(uStack_fc,uStack_100);
          lVar11 = local_208 + lVar11;
          local_168 = *(longlong *)(lVar11 + 0x10);
          uStack_120 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
          uStack_160 = CONCAT44(uStack_11c,uStack_120);
          uVar7 = (ulonglong)(uint)local_228;
          local_128 = local_168;
          local_108 = local_188;
          local_b8 = local_1d8;
          FUN_180024c90((longlong)local_200,fVar3,(int)local_228,(ulonglong)uVar4,
                        (longlong)&local_168,*(longlong *)(lVar20 + 0x10),uVar17,local_1b8,
                        (longlong)local_1b0,local_1a8,local_1c0,lVar23,lVar22,
                        in_stack_fffffffffffffdb0,&local_188,&local_1d8);
        }
      }
      else {
        lVar11 = lVar11 * 0x68;
        local_168 = *(longlong *)(lVar11 + 0x10 + param_7);
        uStack_150 = (undefined4)*(undefined8 *)(lVar11 + 0x50 + param_7);
        uStack_160 = CONCAT44(uStack_14c,uStack_150);
        lVar12 = param_6 + lVar11;
        local_218 = *(longlong *)(lVar12 + 0x10);
        uStack_210 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
        uStack_180 = CONCAT44(uStack_20c,uStack_210);
        lVar11 = local_208 + lVar11;
        local_1d8 = *(longlong *)(lVar11 + 0x10);
        uStack_d0 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
        uStack_1d0 = CONCAT44(uStack_cc,uStack_d0);
        in_stack_fffffffffffffdb0 = *(longlong *)(local_1f0 + 6);
        uVar7 = (ulonglong)(uint)local_228;
        local_188 = local_218;
        local_158 = local_168;
        local_d8 = local_1d8;
        FUN_180024770((longlong)local_200,fVar3,(int)local_228,(ulonglong)uVar4,(longlong)&local_1d8
                      ,*(longlong *)(lVar20 + 0x10),uVar17,local_1b8,(longlong)local_1b0,local_1a8,
                      local_1c0,lVar23,lVar22,in_stack_fffffffffffffdb0,&local_188,&local_168);
      }
      local_1d8 = *(longlong *)(lVar12 + 0x10);
      uStack_e0 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
      uStack_1d0 = CONCAT44(uStack_dc,uStack_e0);
      local_188 = *(longlong *)(lVar11 + 0x10);
      uStack_130 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
      uStack_180 = CONCAT44(uStack_12c,uStack_130);
      local_138 = local_188;
      local_e8 = local_1d8;
      FUN_180014730((int)fVar3,(int)local_228,uVar7,&local_1d8,*pfVar5,fVar1);
      piVar21 = piVar21 + 1;
      local_170 = local_170 + 1;
    } while (local_170 != local_1e8);
  }
  local_a0[1] = local_a0[1] - local_90;
  return;
}


```

## procedure_assembly {'procedure': '0x180003960'}
Evidence: `ev_bc6d07bdf598eb01ed26a4ed05eabcf546f81f56624c125ab96a0eda1626dda0`
```c
0x180003960: TEST EDX, EDX
0x180003962: JLE 0x180003c2c
0x180003968: MOV R11, RSP
0x18000396b: MOV qword ptr [R11 + 0x18], R8
0x18000396f: PUSH RBX
0x180003970: PUSH RDI
0x180003971: PUSH R15
0x180003973: SUB RSP, 0xb0
0x18000397a: MOV R10, qword ptr [RSP + 0xf0]
0x180003982: MOVSS XMM5, dword ptr [RSP + 0xd8]
0x18000398b: MOV qword ptr [R11 + 0x8], RBP
0x18000398f: MOV qword ptr [R11 + -0x20], RSI
0x180003993: MOV RSI, qword ptr [RSP + 0x128]
0x18000399b: MOV qword ptr [R11 + -0x28], R12
0x18000399f: MOV R12, qword ptr [RSP + 0xf8]
0x1800039a7: MOV qword ptr [R11 + -0x30], R13
0x1800039ab: MOV qword ptr [R11 + -0x38], R14
0x1800039af: MOVAPS xmmword ptr [R11 + -0x48], XMM6
0x1800039b4: MOVSS XMM6, dword ptr [0x18007c518]
0x1800039bc: MOVAPS xmmword ptr [RSP + 0x70], XMM7
0x1800039c1: MOVSS XMM7, dword ptr [RSP + 0x108]
0x1800039ca: MOV R14, qword ptr [RSP + 0x120]
0x1800039d2: MOVAPS xmmword ptr [R11 + -0x68], XMM8
0x1800039d7: MOVAPS xmmword ptr [R11 + -0x78], XMM9
0x1800039dc: MOVSS XMM9, dword ptr [RSP + 0x110]
0x1800039e6: MOV RAX, R8
0x1800039e9: MOVSXD R15, ECX
0x1800039ec: XOR R8D, R8D
0x1800039ef: MOVAPS xmmword ptr [RSP + 0x40], XMM10
0x1800039f5: MOVAPS xmmword ptr [RSP + 0x30], XMM11
0x1800039fb: MOVSXD RCX, EDX
0x1800039fe: MOV EDI, R9D
0x180003a01: MOV EBX, R8D
0x180003a04: MOVSS XMM11, dword ptr [0x18007c504]
0x180003a0d: MOVSS XMM10, dword ptr [RSP + 0x100]
0x180003a17: MOV qword ptr [RSP + 0x8], RCX
0x180003a1c: NOP dword ptr [RAX]
0x180003a20: MOV R11, R8
0x180003a23: TEST R15, R15
0x180003a26: JLE 0x180003bc8
0x180003a2c: MOVUPS XMM1, xmmword ptr [RAX]
0x180003a2f: MOVUPS XMM0, xmmword ptr [RAX + 0x10]
0x180003a33: MOVAPS XMM8, XMM9
0x180003a37: MULSS XMM8, XMM11
0x180003a3c: MOVAPS xmmword ptr [RSP + 0x10], XMM1
0x180003a41: MOVAPS xmmword ptr [RSP + 0x20], XMM0
0x180003a46: PSRLDQ XMM1, 0x8
0x180003a4b: PSRLDQ XMM0, 0x8
0x180003a50: MOVQ RAX, XMM1
0x180003a55: MOVQ R13, XMM0
0x180003a5a: MOV qword ptr [RSP], RAX
0x180003a5e: NOP
0x180003a60: MOVSS XMM3, dword ptr [RAX + RBX*0x4]
0x180003a65: MOV RCX, qword ptr [RSP + 0x118]
0x180003a6d: MOV R8D, dword ptr [R13 + RBX*0x4]
0x180003a72: MOV EAX, R8D
0x180003a75: LEA RBP, [R11*0x4]
0x180003a7d: MOVAPS XMM1, xmmword ptr [RCX]
0x180003a80: MOV RCX, qword ptr [RSP + 0x10]
0x180003a85: MOVDQA XMM0, XMM1
0x180003a89: PSRLDQ XMM0, 0x8
0x180003a8e: MOVD EDX, XMM0
0x180003a92: MOVAPS XMM0, XMM6
0x180003a95: SUBSS XMM0, XMM3
0x180003a99: MOVSS XMM2, dword ptr [RCX + RBP*0x1]
0x180003a9e: MOVQ RCX, XMM1
0x180003aa3: MOVAPS XMM1, XMM6
0x180003aa6: IMUL EAX, EDX
0x180003aa9: MOVSXD R9, EAX
0x180003aac: LEA EAX, [R8 + 0x1]
0x180003ab0: ADD R9, RCX
0x180003ab3: SUBSS XMM1, XMM2
0x180003ab7: IMUL EAX, EDX
0x180003aba: MOVSXD RDX, EAX
0x180003abd: MOV RAX, qword ptr [RSP + 0x20]
0x180003ac2: MOVSXD RAX, dword ptr [RAX + RBP*0x1]
0x180003ac6: ADD RDX, RCX
0x180003ac9: MOVAPS XMM4, XMM1
0x180003acc: MULSS XMM1, XMM3
0x180003ad0: MULSS XMM3, XMM2
0x180003ad4: MULSS XMM1, dword ptr [RDX + RAX*0x4]
0x180003ad9: MULSS XMM3, dword ptr [RDX + RAX*0x4 + 0x4]
0x180003adf: MULSS XMM4, XMM0
0x180003ae3: MULSS XMM4, dword ptr [R9 + RAX*0x4]
0x180003ae9: MULSS XMM0, XMM2
0x180003aed: MULSS XMM0, dword ptr [R9 + RAX*0x4 + 0x4]
0x180003af4: ADDSS XMM4, XMM0
0x180003af8: ADDSS XMM4, XMM1
0x180003afc: ADDSS XMM4, XMM3
0x180003b00: COMISS XMM10, XMM4
0x180003b04: MOVAPS XMM0, XMM4
0x180003b07: SUBSS XMM0, XMM10
0x180003b0c: SUBSS XMM0, XMM8
0x180003b11: DIVSS XMM0, XMM9
0x180003b16: CVTTSS2SI EAX, XMM0
0x180003b1a: JC 0x180003b24
0x180003b1c: XOR R8D, R8D
0x180003b1f: MOV EDX, R8D
0x180003b22: JMP 0x180003b85
0x180003b24: COMISS XMM4, XMM7
0x180003b27: JNC 0x180003b7c
0x180003b29: CMP EAX, EDI
0x180003b2b: JGE 0x180003b7c
0x180003b2d: TEST EAX, EAX
0x180003b2f: MOV R8D, 0x0
0x180003b35: CMOVS EAX, R8D
0x180003b39: CDQE
0x180003b3b: MOVSXD RDX, dword ptr [R12 + RAX*0x4]
0x180003b3f: MOVSS XMM1, dword ptr [R10 + RDX*0x4]
0x180003b45: COMISS XMM1, XMM4
0x180003b48: JA 0x180003b60
0x180003b4a: NOP word ptr [RAX + RAX*0x1]
0x180003b50: INC EDX
0x180003b52: MOVSXD RAX, EDX
0x180003b55: MOVSS XMM1, dword ptr [R10 + RAX*0x4]
0x180003b5b: COMISS XMM1, XMM4
0x180003b5e: JBE 0x180003b50
0x180003b60: MOVSXD RAX, EDX
0x180003b63: MOVAPS XMM5, XMM4
0x180003b66: DEC EDX
0x180003b68: SUBSS XMM5, dword ptr [R10 + RAX*0x4 + -0x4]
0x180003b6f: SUBSS XMM1, dword ptr [R10 + RAX*0x4 + -0x4]
0x180003b76: DIVSS XMM5, XMM1
0x180003b7a: JMP 0x180003b85
0x180003b7c: LEA EDX, [RDI + -0x2]
0x180003b7f: MOVAPS XMM5, XMM6
0x180003b82: XOR R8D, R8D
0x180003b85: MOVSXD RAX, dword ptr [R14 + 0x8]
0x180003b89: IMUL RAX, RBX
0x180003b8d: LEA RCX, [RAX + R11*0x2]
0x180003b91: MOV RAX, qword ptr [R14]
0x180003b94: INC R11
0x180003b97: MOV word ptr [RCX + RAX*0x1], DX
0x180003b9b: MOVSXD RCX, dword ptr [RSI + 0x8]
0x180003b9f: MOV RAX, qword ptr [RSI]
0x180003ba2: IMUL RCX, RBX
0x180003ba6: ADD RCX, RBP
0x180003ba9: MOVSS dword ptr [RCX + RAX*0x1], XMM5
0x180003bae: MOV RAX, qword ptr [RSP]
0x180003bb2: CMP R11, R15
0x180003bb5: JL 0x180003a60
0x180003bbb: MOV RAX, qword ptr [RSP + 0xe0]
0x180003bc3: MOV RCX, qword ptr [RSP + 0x8]
0x180003bc8: INC RBX
0x180003bcb: CMP RBX, RCX
0x180003bce: JL 0x180003a20
0x180003bd4: MOVAPS XMM11, xmmword ptr [RSP + 0x30]
0x180003bda: MOVAPS XMM10, xmmword ptr [RSP + 0x40]
0x180003be0: MOVAPS XMM9, xmmword ptr [RSP + 0x50]
0x180003be6: MOV R14, qword ptr [RSP + 0x90]
0x180003bee: MOV R13, qword ptr [RSP + 0x98]
0x180003bf6: MOV R12, qword ptr [RSP + 0xa0]
0x180003bfe: MOV RSI, qword ptr [RSP + 0xa8]
0x180003c06: MOV RBP, qword ptr [RSP + 0xd0]
0x180003c0e: MOVAPS XMM8, xmmword ptr [RSP + 0x60]
0x180003c14: MOVAPS XMM7, xmmword ptr [RSP + 0x70]
0x180003c19: MOVAPS XMM6, xmmword ptr [RSP + 0x80]
0x180003c21: ADD RSP, 0xb0
0x180003c28: POP R15
0x180003c2a: POP RDI
0x180003c2b: POP RBX
0x180003c2c: RET
```

## procedure_pseudo_code {'procedure': '0x1800157e0'}
Evidence: `ev_bef161eff7ef9bc56337ebf856bf218be55969bf34da0445febc03f6094ecb75`
```c

void FUN_1800157e0(float *param_1,longlong *param_2,int param_3)

{
  float *pfVar1;
  float *pfVar2;
  longlong lVar3;
  float *pfVar4;
  longlong lVar5;
  float *pfVar6;
  float *pfVar7;
  longlong lVar8;
  longlong lVar9;
  
  lVar3 = *param_2;
  lVar9 = (longlong)param_3;
  lVar8 = 0;
  if (3 < lVar9) {
    lVar5 = (lVar9 - 4U >> 2) + 1;
    lVar8 = lVar5 * 4;
    pfVar4 = (float *)(lVar3 + 8);
    pfVar6 = param_1;
    do {
      pfVar1 = pfVar6 + 2;
      pfVar2 = pfVar6 + 3;
      pfVar7 = pfVar6 + 4;
      pfVar4[-2] = *pfVar6 * 6.0 + pfVar6[-1] + pfVar6[1];
      pfVar4[-1] = (pfVar6[1] + *pfVar6) * 4.0;
      *pfVar4 = pfVar6[1] * 6.0 + *pfVar6 + *pfVar1;
      pfVar4[1] = (*pfVar1 + pfVar6[1]) * 4.0;
      pfVar4[2] = *pfVar1 * 6.0 + pfVar6[1] + *pfVar2;
      pfVar4[3] = (*pfVar2 + *pfVar1) * 4.0;
      pfVar4[4] = *pfVar2 * 6.0 + *pfVar1 + *pfVar7;
      pfVar4[5] = (*pfVar7 + *pfVar2) * 4.0;
      lVar5 = lVar5 + -1;
      pfVar4 = pfVar4 + 8;
      pfVar6 = pfVar7;
    } while (lVar5 != 0);
  }
  if (lVar8 < lVar9) {
    pfVar4 = param_1 + lVar8;
    do {
      lVar8 = lVar8 + 1;
      pfVar6 = pfVar4 + 1;
      *(float *)(lVar3 + -8 + lVar8 * 8) = *pfVar4 * 6.0 + pfVar4[-1] + *pfVar6;
      *(float *)(lVar3 + -4 + lVar8 * 8) = (*pfVar6 + *pfVar4) * 4.0;
      pfVar4 = pfVar6;
    } while (lVar8 < lVar9);
  }
  *param_2 = param_2[1];
  param_2[1] = param_2[2];
  param_2[2] = lVar3;
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180011630'}
Evidence: `ev_bf0ded9713bab33dede85925e4c967f184e5ffe275a7ef20d2f6067f3d7dce15`
```c

ulonglong * FUN_180011630(ulonglong *param_1,float *param_2,float *param_3)

{
  float fVar1;
  float fVar2;
  float *pfVar3;
  float *pfVar4;
  float *pfVar5;
  float *pfVar6;
  float *pfVar7;
  float *pfVar8;
  float *pfVar9;
  float *pfVar10;
  float *pfVar11;
  float *pfVar12;
  
  pfVar6 = param_2 + (((longlong)param_3 - (longlong)param_2 >> 3) -
                      ((longlong)param_3 - (longlong)param_2 >> 0x3f) >> 1) * 2;
  FUN_1800118f0(param_2,pfVar6,param_3 + -2);
  pfVar4 = pfVar6 + 2;
  if (param_2 < pfVar6) {
    while( true ) {
      fVar1 = pfVar6[-2];
      fVar2 = *pfVar6;
      pfVar11 = pfVar6 + -2;
      if ((((fVar1 < fVar2) || (fVar2 < fVar1)) || ((int)pfVar6[-1] < (int)pfVar6[1])) ||
         (fVar2 < fVar1)) break;
      if (((fVar2 <= fVar1) && ((int)pfVar6[1] < (int)pfVar6[-1])) ||
         (pfVar6 = pfVar11, pfVar11 <= param_2)) break;
    }
  }
  pfVar9 = pfVar4;
  pfVar11 = pfVar6;
  if (pfVar4 < param_3) {
    fVar1 = *pfVar6;
    while ((fVar2 = *pfVar4, pfVar9 = pfVar4, fVar1 <= fVar2 && (fVar2 <= fVar1))) {
      if ((((int)pfVar4[1] < (int)pfVar6[1]) || (fVar1 < fVar2)) ||
         (((fVar1 <= fVar2 && ((int)pfVar6[1] < (int)pfVar4[1])) ||
          (pfVar4 = pfVar4 + 2, pfVar9 = pfVar4, param_3 <= pfVar4)))) break;
    }
  }
LAB_180011700:
  pfVar3 = pfVar6;
  pfVar12 = pfVar11;
  pfVar10 = pfVar9;
  pfVar5 = pfVar4;
  if (pfVar4 < param_3) {
    pfVar7 = pfVar9 + -2;
    do {
      fVar1 = *pfVar6;
      fVar2 = *pfVar4;
      pfVar8 = pfVar7;
      pfVar10 = pfVar9;
      if (fVar2 <= fVar1) {
        pfVar5 = pfVar4;
        if (fVar2 < fVar1) break;
        if ((int)pfVar4[1] <= (int)pfVar6[1]) {
          if ((fVar2 < fVar1) || ((fVar2 <= fVar1 && ((int)pfVar4[1] < (int)pfVar6[1])))) break;
          pfVar8 = pfVar7 + 2;
          pfVar10 = pfVar9 + 2;
          if ((pfVar9 != pfVar4) && (pfVar8 != pfVar4)) {
            fVar1 = *pfVar8;
            *pfVar8 = fVar2;
            *pfVar4 = fVar1;
            fVar1 = pfVar7[3];
            pfVar7[3] = pfVar4[1];
            pfVar4[1] = fVar1;
          }
        }
      }
      pfVar4 = pfVar4 + 2;
      pfVar7 = pfVar8;
      pfVar9 = pfVar10;
      pfVar5 = pfVar4;
    } while (pfVar4 < param_3);
  }
  do {
    if (pfVar11 <= param_2) break;
    pfVar6 = pfVar12 + -2;
    fVar1 = *pfVar6;
    fVar2 = *pfVar3;
    pfVar4 = pfVar3;
    if (fVar2 <= fVar1) {
      if (fVar2 < fVar1) break;
      if ((int)pfVar3[1] <= (int)pfVar12[-1]) {
        if ((fVar2 < fVar1) || ((fVar2 <= fVar1 && ((int)pfVar3[1] < (int)pfVar12[-1])))) break;
        pfVar4 = pfVar3 + -2;
        if (pfVar4 != pfVar6) {
          fVar2 = *pfVar4;
          *pfVar4 = fVar1;
          *pfVar6 = fVar2;
          fVar1 = pfVar3[-1];
          pfVar3[-1] = pfVar12[-1];
          pfVar12[-1] = fVar1;
        }
      }
    }
    pfVar11 = pfVar11 + -2;
    pfVar3 = pfVar4;
    pfVar12 = pfVar6;
  } while( true );
  if (pfVar11 == param_2) {
    if (pfVar5 == param_3) {
      *param_1 = (ulonglong)pfVar3;
      param_1[1] = (ulonglong)pfVar10;
      return param_1;
    }
    if ((pfVar10 != pfVar5) && (pfVar3 != pfVar10)) {
      fVar1 = *pfVar3;
      *pfVar3 = *pfVar10;
      *pfVar10 = fVar1;
      fVar1 = pfVar3[1];
      pfVar3[1] = pfVar10[1];
      pfVar10[1] = fVar1;
    }
    pfVar4 = pfVar5 + 2;
    pfVar6 = pfVar3 + 2;
    pfVar9 = pfVar10 + 2;
    if (pfVar3 != pfVar5) {
      fVar1 = *pfVar3;
      *pfVar3 = *pfVar5;
      *pfVar5 = fVar1;
      fVar1 = pfVar3[1];
      pfVar3[1] = pfVar5[1];
      pfVar5[1] = fVar1;
    }
  }
  else {
    pfVar12 = pfVar11 + -2;
    if (pfVar5 == param_3) {
      pfVar6 = pfVar3 + -2;
      if (pfVar12 != pfVar6) {
        fVar1 = *pfVar12;
        *pfVar12 = *pfVar6;
        *pfVar6 = fVar1;
        fVar1 = pfVar11[-1];
        pfVar11[-1] = pfVar3[-1];
        pfVar3[-1] = fVar1;
      }
      pfVar9 = pfVar10 + -2;
      pfVar4 = pfVar5;
      pfVar11 = pfVar12;
      if (pfVar6 != pfVar9) {
        fVar1 = *pfVar6;
        *pfVar6 = *pfVar9;
        *pfVar9 = fVar1;
        fVar1 = pfVar3[-1];
        pfVar3[-1] = pfVar10[-1];
        pfVar10[-1] = fVar1;
      }
    }
    else {
      if (pfVar5 != pfVar12) {
        fVar1 = *pfVar5;
        *pfVar5 = *pfVar12;
        *pfVar12 = fVar1;
        fVar1 = pfVar5[1];
        pfVar5[1] = pfVar11[-1];
        pfVar11[-1] = fVar1;
      }
      pfVar4 = pfVar5 + 2;
      pfVar6 = pfVar3;
      pfVar9 = pfVar10;
      pfVar11 = pfVar12;
    }
  }
  goto LAB_180011700;
}


```

## procedure_pseudo_code {'procedure': '0x1800120e0'}
Evidence: `ev_c0a72f305270cde830947a9fb033cf34f8d74de6b92d924cd7bb878fc6d06435`
```c

void FUN_1800120e0(undefined8 param_1,longlong param_2,longlong param_3,undefined4 param_4,
                  uint param_5)

{
  ulonglong uVar1;
  longlong lVar2;
  undefined8 local_res8;
  undefined4 local_58;
  undefined4 local_54;
  undefined4 local_50 [2];
  longlong local_48;
  undefined8 local_40;
  undefined4 local_38 [2];
  longlong local_30;
  undefined8 local_28;
  
  if (0 < (int)param_5) {
    lVar2 = param_3 - param_2;
    uVar1 = (ulonglong)param_5;
    do {
      local_48 = lVar2 + param_2;
      local_40 = 0;
      local_50[0] = 0x2010000;
      local_28 = 0;
      local_38[0] = 0x1010000;
      local_res8 = 0xffffffffffffffff;
      local_58 = param_4;
      local_54 = param_4;
      local_30 = param_2;
      cv::blur(local_38,local_50,&local_58,&local_res8,4);
      param_2 = param_2 + 0x68;
      uVar1 = uVar1 - 1;
    } while (uVar1 != 0);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180027570'}
Evidence: `ev_c37706c610878c2593f2dd955c83a0566fd6789dd3adccd46f52457cbd28d062`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180027570(int param_1,int param_2,undefined8 param_3,longlong *param_4,longlong *param_5,
                  undefined1 *param_6,int param_7,undefined4 param_8,longlong *param_9)

{
  float *pfVar1;
  float fVar2;
  float fVar3;
  float fVar4;
  undefined2 uVar5;
  undefined1 uVar6;
  float *pfVar7;
  longlong lVar8;
  longlong lVar9;
  int iVar10;
  undefined1 uVar11;
  char cVar12;
  int iVar13;
  char cVar14;
  undefined1 auStackY_128 [32];
  undefined4 uStack_bc;
  undefined4 uStack_b4;
  longlong local_a8;
  undefined8 uStack_a0;
  undefined8 local_98;
  longlong local_88;
  longlong lStack_80;
  longlong local_78;
  longlong lStack_70;
  float local_68 [4];
  char local_58 [8];
  ulonglong local_50;
  
  local_50 = DAT_18008b080 ^ (ulonglong)auStackY_128;
  cVar14 = '\0';
  iVar10 = 0;
  uVar6 = 0;
  local_58[1] = '\0';
  local_58[2] = '\0';
  uVar5 = local_58._1_2_;
  local_58[3] = 0;
  local_58[0] = '\0';
  if (0 < param_2) {
    local_58[2] = 0;
    uVar11 = 0;
    cVar12 = local_58[2];
    do {
      local_58._1_2_ = uVar5;
      iVar13 = 0;
      if (0 < param_1) {
        lVar9 = (longlong)param_7;
        do {
          local_a8 = *param_4;
          local_98 = CONCAT44(uStack_b4,param_8);
          local_78 = param_9[2];
          lStack_70 = param_9[3];
          uStack_a0 = CONCAT44(uStack_bc,(int)param_4[1]);
          local_88 = *param_9;
          lStack_80 = param_9[1];
          pfVar7 = FUN_18002eb70(local_68,iVar13,iVar10,&local_88,&local_a8);
          fVar2 = pfVar7[1];
          fVar3 = pfVar7[2];
          fVar4 = pfVar7[3];
          pfVar1 = (float *)(*param_5 +
                            (longlong)
                            ((int)((ulonglong)(longlong)(int)param_5[1] >> 2) * iVar10 + iVar13) * 4
                            );
          if (0.0 < *pfVar7) {
            cVar14 = '\x01';
          }
          *pfVar1 = *pfVar7;
          *(float *)(lVar9 + (longlong)pfVar1) = fVar2;
          *(float *)((longlong)pfVar1 + lVar9 * 2) = fVar3;
          if (0.0 < fVar2) {
            uVar11 = 1;
          }
          local_58[0] = cVar14;
          if (0.0 < fVar3) {
            cVar12 = 1;
          }
          *(float *)(lVar9 * 3 + (longlong)pfVar1) = fVar4;
          local_58[2] = cVar12;
          local_58[1] = uVar11;
          if (0.0 < fVar4) {
            uVar6 = 1;
          }
          iVar13 = iVar13 + 1;
          local_58[3] = uVar6;
        } while (iVar13 < param_1);
      }
      iVar10 = iVar10 + 1;
      uVar5 = local_58._1_2_;
    } while (iVar10 < param_2);
  }
  lVar9 = -(longlong)param_6;
  lVar8 = 4;
  do {
    if (param_6[(longlong)(local_58 + lVar9)] != '\0') {
      *param_6 = 1;
    }
    param_6 = param_6 + 1;
    lVar8 = lVar8 + -1;
  } while (lVar8 != 0);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180018e40'}
Evidence: `ev_c530dfe1a8513adc37fdfaf8229a6364f9ce11f5e90afe000c490514a0981b72`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180018e40(longlong *param_1,longlong param_2,longlong param_3,longlong param_4,int param_5)

{
  int iVar1;
  int iVar2;
  undefined1 auStackY_128 [32];
  longlong local_e8;
  undefined8 uStack_e0;
  longlong local_d8;
  undefined8 uStack_d0;
  longlong local_c8;
  longlong lStack_c0;
  longlong local_b8;
  longlong lStack_b0;
  longlong local_a8;
  undefined8 uStack_a0;
  longlong local_98;
  undefined8 uStack_90;
  longlong local_88;
  undefined4 local_80;
  undefined8 local_78;
  undefined4 local_70;
  undefined8 local_68;
  undefined4 local_60;
  longlong local_58;
  undefined4 local_50;
  undefined8 local_48;
  undefined4 local_40;
  undefined8 local_38;
  undefined4 local_30;
  undefined8 local_28;
  undefined4 local_20;
  ulonglong local_18;
  
  local_18 = DAT_18008b080 ^ (ulonglong)auStackY_128;
  iVar1 = (*(int **)(param_3 + 0x40))[1];
  iVar2 = **(int **)(param_3 + 0x40);
  if (param_5 == 1) {
    local_98 = *(longlong *)(param_4 + 0x10);
    local_b8 = param_1[2];
    lStack_b0 = param_1[3];
    uStack_e0 = CONCAT44(uStack_e0._4_4_,*(undefined4 *)(param_4 + 0x50));
    local_a8 = *(longlong *)(param_3 + 0x10);
    uStack_90 = uStack_e0;
    uStack_e0 = CONCAT44(uStack_e0._4_4_,*(undefined4 *)(param_3 + 0x50));
    local_d8 = *(longlong *)(param_2 + 0x10);
    uStack_a0 = uStack_e0;
    uStack_e0 = CONCAT44(uStack_e0._4_4_,*(undefined4 *)(param_2 + 0x50));
    local_e8 = *(longlong *)(param_2 + 0x78);
    uStack_d0 = uStack_e0;
    uStack_e0 = CONCAT44(uStack_e0._4_4_,*(undefined4 *)(param_2 + 0xb8));
    local_c8 = *param_1;
    lStack_c0 = param_1[1];
    FUN_180018610(iVar1,iVar2,&local_c8,&local_e8,&local_d8,&local_a8,&local_98);
  }
  else {
    local_58 = *(longlong *)(param_2 + 0x10);
    local_b8 = param_1[2];
    lStack_b0 = param_1[3];
    local_50 = *(undefined4 *)(param_2 + 0x50);
    local_48 = *(undefined8 *)(param_2 + 0x78);
    local_40 = *(undefined4 *)(param_2 + 0xb8);
    local_38 = *(undefined8 *)(param_2 + 0xe0);
    local_30 = *(undefined4 *)(param_2 + 0x120);
    local_28 = *(undefined8 *)(param_2 + 0x148);
    local_20 = *(undefined4 *)(param_2 + 0x188);
    local_88 = *(longlong *)(param_3 + 0x10);
    local_80 = *(undefined4 *)(param_3 + 0x50);
    local_78 = *(undefined8 *)(param_3 + 0x78);
    local_70 = *(undefined4 *)(param_3 + 0xb8);
    local_68 = *(undefined8 *)(param_3 + 0xe0);
    local_60 = *(undefined4 *)(param_3 + 0x120);
    local_e8 = *(longlong *)(param_4 + 0x10);
    uStack_e0 = CONCAT44(uStack_e0._4_4_,*(undefined4 *)(param_4 + 0x50));
    uStack_d0 = uStack_e0;
    local_c8 = *param_1;
    lStack_c0 = param_1[1];
    local_d8 = local_e8;
    FUN_1800188c0(iVar1,iVar2,&local_58,&local_88,&local_c8,&local_d8);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180025b60'}
Evidence: `ev_de802c09073927f5bdb6e6f7961b2da5852434de9bb1ec20e263351a595ee437`
```c

void FUN_180025b60(int *param_1,longlong param_2,longlong param_3,longlong param_4)

{
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  longlong local_18;
  undefined8 uStack_10;
  
  local_28 = *(longlong *)(param_4 + 0x10);
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_4 + 0x50));
  local_18 = *(longlong *)(param_3 + 0x10);
  uStack_20 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_38 = *(longlong *)(param_2 + 0x10);
  uStack_10 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_2 + 0x50));
  FUN_18002d840(*param_1,param_1[1],param_1[0x137],param_1[0x138],
                (float)param_1[0x137] / (float)*param_1,(float)param_1[0x138] / (float)param_1[1],
                &local_38,&local_18,&local_28);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180017d60'}
Evidence: `ev_e23d3c2b21e85bd7c8059ee76027a24335dffe61e17208fae65b9aea9c40f86f`
```c

void FUN_180017d60(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5,
                  longlong *param_6,longlong *param_7,longlong *param_8)

{
  longlong lVar1;
  longlong lVar2;
  float fVar3;
  float fVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  float fVar9;
  
  if (0 < param_2) {
    lVar8 = (longlong)param_1;
    lVar7 = 0;
    do {
      lVar6 = 0;
      if (3 < lVar8) {
        do {
          lVar2 = lVar6 * 4;
          fVar3 = *(float *)((int)param_3[1] * lVar7 + *param_3 + lVar2);
          fVar4 = *(float *)((int)param_5[1] * lVar7 + lVar2 + *param_5);
          fVar9 = (*(float *)((int)param_4[1] * lVar7 + *param_4 + lVar2) - fVar4 * fVar3) *
                  *(float *)((int)param_6[1] * lVar7 + lVar2 + *param_6);
          *(float *)((int)param_7[1] * lVar7 + lVar2 + *param_7) = fVar9;
          *(float *)((int)param_8[1] * lVar7 + lVar2 + *param_8) = fVar3 - fVar9 * fVar4;
          fVar3 = *(float *)((int)param_3[1] * lVar7 + *param_3 + 4 + lVar2);
          fVar4 = *(float *)((int)param_5[1] * lVar7 + lVar2 + 4 + *param_5);
          fVar9 = (*(float *)((int)param_4[1] * lVar7 + *param_4 + 4 + lVar2) - fVar4 * fVar3) *
                  *(float *)((int)param_6[1] * lVar7 + lVar2 + 4 + *param_6);
          *(float *)((int)param_7[1] * lVar7 + lVar2 + 4 + *param_7) = fVar9;
          lVar5 = lVar2 + 0xc;
          lVar1 = lVar2 + 8;
          *(float *)((int)param_8[1] * lVar7 + lVar2 + 4 + *param_8) = fVar3 - fVar9 * fVar4;
          fVar3 = *(float *)((int)param_3[1] * lVar7 + *param_3 + lVar1);
          fVar4 = *(float *)((int)param_5[1] * lVar7 + *param_5 + lVar1);
          lVar6 = lVar6 + 4;
          fVar9 = (*(float *)((int)param_4[1] * lVar7 + *param_4 + lVar1) - fVar4 * fVar3) *
                  *(float *)((int)param_6[1] * lVar7 + *param_6 + lVar1);
          *(float *)((int)param_7[1] * lVar7 + lVar1 + *param_7) = fVar9;
          *(float *)((int)param_8[1] * lVar7 + lVar1 + *param_8) = fVar3 - fVar9 * fVar4;
          fVar3 = *(float *)((int)param_3[1] * lVar7 + *param_3 + lVar5);
          fVar4 = *(float *)((int)param_5[1] * lVar7 + *param_5 + lVar5);
          fVar9 = (*(float *)((int)param_4[1] * lVar7 + *param_4 + lVar5) - fVar4 * fVar3) *
                  *(float *)((int)param_6[1] * lVar7 + *param_6 + lVar5);
          *(float *)((int)param_7[1] * lVar7 + *param_7 + lVar5) = fVar9;
          *(float *)((int)param_8[1] * lVar7 + lVar5 + *param_8) = fVar3 - fVar9 * fVar4;
        } while (lVar6 < lVar8 + -3);
      }
      for (; lVar6 < lVar8; lVar6 = lVar6 + 1) {
        lVar1 = lVar6 * 4;
        fVar3 = *(float *)((int)param_3[1] * lVar7 + *param_3 + lVar1);
        fVar4 = *(float *)((int)param_5[1] * lVar7 + lVar1 + *param_5);
        fVar9 = (*(float *)((int)param_4[1] * lVar7 + *param_4 + lVar1) - fVar4 * fVar3) *
                *(float *)((int)param_6[1] * lVar7 + lVar1 + *param_6);
        *(float *)((int)param_7[1] * lVar7 + lVar1 + *param_7) = fVar9;
        *(float *)((int)param_8[1] * lVar7 + lVar1 + *param_8) = fVar3 - fVar9 * fVar4;
      }
      lVar7 = lVar7 + 1;
    } while (lVar7 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180015100'}
Evidence: `ev_e3ea72f1e8944cbc8bf62d4582151080b7754312d99e2d15acc670aeb8696bef`
```c

void FUN_180015100(longlong param_1,longlong *param_2,int param_3)

{
  longlong lVar1;
  float *pfVar2;
  float *pfVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  
  lVar1 = *param_2;
  lVar5 = (longlong)param_3;
  lVar4 = 0;
  if (3 < lVar5) {
    lVar6 = (lVar5 - 4U >> 2) + 1;
    lVar4 = lVar6 * 4;
    pfVar2 = (float *)(lVar1 + 8);
    pfVar3 = (float *)(param_1 + -4);
    do {
      pfVar2[-2] = (pfVar3[2] + *pfVar3) * 4.0 + pfVar3[-1] + pfVar3[3] + pfVar3[1] * 6.0;
      pfVar2[-1] = (pfVar3[4] + pfVar3[2]) * 4.0 + pfVar3[5] + pfVar3[1] + pfVar3[3] * 6.0;
      *pfVar2 = (pfVar3[6] + pfVar3[4]) * 4.0 + pfVar3[7] + pfVar3[3] + pfVar3[5] * 6.0;
      pfVar2[1] = pfVar3[9] + pfVar3[5] + (pfVar3[8] + pfVar3[6]) * 4.0 + pfVar3[7] * 6.0;
      lVar6 = lVar6 + -1;
      pfVar2 = pfVar2 + 4;
      pfVar3 = pfVar3 + 8;
    } while (lVar6 != 0);
  }
  if (lVar4 < lVar5) {
    pfVar2 = (float *)(param_1 + -4 + lVar4 * 8);
    do {
      lVar4 = lVar4 + 1;
      *(float *)(lVar1 + -4 + lVar4 * 4) =
           pfVar2[3] + pfVar2[-1] + (pfVar2[2] + *pfVar2) * 4.0 + pfVar2[1] * 6.0;
      pfVar2 = pfVar2 + 2;
    } while (lVar4 < lVar5);
  }
  *param_2 = param_2[1];
  param_2[1] = param_2[2];
  param_2[2] = param_2[3];
  lVar4 = param_2[4];
  param_2[4] = lVar1;
  param_2[3] = lVar4;
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180024320'}
Evidence: `ev_e53783163e3f3580d1a69ab2762106ba96e066506d5bac20081cc14526cc7ae3`
```c

void FUN_180024320(longlong param_1,longlong param_2,longlong param_3)

{
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  longlong local_18;
  undefined8 uStack_10;
  
  local_28 = *(longlong *)(param_3 + 0x10);
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_18 = *(longlong *)(param_2 + 0x10);
  uStack_20 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_2 + 0x50));
  local_38 = *(longlong *)(param_1 + 0x10);
  uStack_10 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_1 + 0x50));
  FUN_18002ba10((*(int **)(param_1 + 0x40))[1],**(int **)(param_1 + 0x40),&local_38,&local_18,
                &local_28);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180017560'}
Evidence: `ev_e6a433ee89fdcbd512b92a0bf82c31e09297ac12f59d0083d9a88ca096788f84`
```c

void FUN_180017560(longlong param_1,longlong param_2,longlong param_3,longlong param_4,
                  longlong param_5)

{
  longlong lVar1;
  int *piVar2;
  int iVar3;
  int iVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  int *piVar12;
  longlong lVar13;
  int *piVar14;
  undefined4 uStack_8c;
  undefined4 uStack_7c;
  undefined4 uStack_6c;
  longlong local_68;
  undefined8 uStack_60;
  longlong local_58;
  undefined8 uStack_50;
  longlong local_48;
  undefined8 uStack_40;
  
  if (*(char *)(param_1 + 0x24) == '\0') {
    lVar13 = (longlong)*(int *)(param_1 + 0x78);
    piVar14 = (int *)(param_1 + 0x244);
    iVar3 = *(int *)(param_1 + 0x4dc);
    iVar4 = *(int *)(param_1 + 0x4e0);
    piVar2 = piVar14 + *(int *)(param_1 + 0x254);
    if (piVar14 != piVar2) {
      lVar8 = (longlong)(*(int *)(param_1 + 0x78) / 2);
      lVar1 = lVar8 + 1;
      do {
        if (lVar1 < lVar13) {
          lVar5 = (longlong)*piVar14;
          piVar12 = (int *)(param_2 + lVar1 * 4);
          lVar10 = (lVar13 - lVar8) + -1;
          do {
            lVar9 = 0;
            if (0 < lVar13) {
              lVar11 = 0;
              do {
                lVar6 = (longlong)*piVar12 * 0x68 + *(longlong *)(param_5 + lVar5 * 8);
                local_68 = *(longlong *)(lVar6 + 0x10);
                lVar7 = *(longlong *)(param_3 + lVar5 * 8);
                local_58 = *(longlong *)(lVar7 + 0x10 + lVar11);
                uStack_60 = CONCAT44(uStack_7c,*(undefined4 *)(lVar6 + 0x50));
                uStack_50 = CONCAT44(uStack_8c,*(undefined4 *)(lVar7 + 0x50 + lVar11));
                lVar7 = (longlong)*(int *)(param_2 + lVar9 * 4) * 0x68 +
                        *(longlong *)(param_4 + lVar5 * 8);
                local_48 = *(longlong *)(lVar7 + 0x10);
                uStack_40 = CONCAT44(uStack_6c,*(undefined4 *)(lVar7 + 0x50));
                FUN_180016dd0(iVar3,iVar4,&local_48,&local_58,&local_68);
                lVar9 = lVar9 + 1;
                lVar11 = lVar11 + 0x68;
              } while (lVar9 < lVar13);
            }
            piVar12 = piVar12 + 1;
            lVar10 = lVar10 + -1;
          } while (lVar10 != 0);
        }
        piVar14 = piVar14 + 1;
      } while (piVar14 != piVar2);
    }
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180026940'}
Evidence: `ev_edc5a1718abecaf9af7d394289617a0e0839a3fa464684d2302323ef6681f077`
```c

void FUN_180026940(longlong *param_1,longlong param_2,uint param_3,int param_4,Mat *param_5,
                  Mat *param_6,Mat *param_7,Mat *param_8,Mat *param_9,Mat *param_10)

{
  int iVar1;
  undefined4 uVar2;
  undefined4 uVar3;
  Mat *pMVar4;
  ulonglong uVar5;
  ulonglong uVar6;
  uint uVar7;
  uint uVar8;
  uint uVar9;
  ulonglong uVar10;
  Mat *pMVar11;
  int iVar12;
  longlong lVar13;
  int *piVar14;
  longlong lVar15;
  uint local_298;
  undefined4 local_294;
  uint local_290;
  undefined4 local_28c;
  uint local_288;
  undefined4 local_284;
  uint local_280;
  undefined4 local_27c;
  undefined8 local_278;
  Mat local_268 [4];
  int local_264;
  undefined8 local_258;
  undefined8 local_250;
  undefined8 local_248;
  undefined8 local_240;
  longlong local_230;
  longlong local_228;
  undefined1 *local_220;
  undefined1 local_218 [32];
  Mat local_1f8 [4];
  int local_1f4;
  undefined8 local_1e8;
  undefined8 local_1e0;
  undefined8 local_1d8;
  undefined8 local_1d0;
  longlong local_1c0;
  longlong local_1b8;
  undefined1 *local_1b0;
  undefined1 local_1a8 [32];
  Mat local_188 [4];
  int local_184;
  undefined8 local_178;
  undefined8 local_170;
  undefined8 local_168;
  undefined8 local_160;
  longlong local_150;
  longlong local_148;
  undefined1 *local_140;
  undefined1 local_138 [32];
  Mat local_118 [4];
  int local_114;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  undefined8 local_f0;
  longlong local_e0;
  longlong local_d8;
  undefined1 *local_d0;
  undefined1 local_c8 [32];
  Mat local_a8 [4];
  int local_a4;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_80;
  longlong local_70;
  longlong local_68;
  undefined1 *local_60;
  undefined1 local_58 [32];
  
  local_278 = 0xfffffffffffffffe;
  uVar9 = *(uint *)(param_2 + 0x4dc);
  uVar2 = *(undefined4 *)(param_2 + 0x4e0);
  uVar8 = *(uint *)(param_2 + 0x6fc);
  uVar3 = *(undefined4 *)(param_2 + 0x700);
  piVar14 = (int *)(param_2 + 0x14);
  if (*(int *)(param_2 + 0xa8) < *(int *)(param_2 + 0x14)) {
    piVar14 = (int *)(param_2 + 0xa8);
  }
  iVar12 = *piVar14;
  FUN_18002e650(param_1,param_2,param_3,param_5);
  local_288 = uVar9;
  local_284 = uVar2;
  pMVar4 = (Mat *)FUN_18000e170(param_1,(uint *)local_1f8,&local_288,0,iVar12);
  pMVar11 = param_6;
  if (param_6 != pMVar4) {
    FUN_180001780(param_6,pMVar4);
  }
  *(undefined4 *)(pMVar11 + 0x60) = *(undefined4 *)(pMVar4 + 0x60);
  if (local_1c0 != 0) {
    LOCK();
    piVar14 = (int *)(local_1c0 + 0x14);
    iVar12 = *piVar14;
    *piVar14 = *piVar14 + -1;
    UNLOCK();
    if (iVar12 == 1) {
      cv::Mat::deallocate(local_1f8);
    }
  }
  uVar6 = 0;
  local_1c0 = 0;
  local_1e8 = 0;
  local_1d0 = 0;
  local_1d8 = 0;
  local_1e0 = 0;
  uVar5 = uVar6;
  if (0 < local_1f4) {
    do {
      *(undefined4 *)(local_1b8 + uVar6) = 0;
      uVar7 = (int)uVar5 + 1;
      uVar6 = uVar6 + 4;
      uVar5 = (ulonglong)uVar7;
    } while ((int)uVar7 < local_1f4);
  }
  if (local_1b0 != local_1a8) {
    cv::fastFree(local_1b0);
  }
  param_6._0_4_ = uVar9;
  param_6._4_4_ = uVar2;
  pMVar4 = (Mat *)FUN_18000e170(param_1,(uint *)local_118,(uint *)&param_6,param_4 / 2,1);
  pMVar11 = param_10;
  if (param_10 != pMVar4) {
    FUN_180001780(param_10,pMVar4);
  }
  *(undefined4 *)(pMVar11 + 0x60) = *(undefined4 *)(pMVar4 + 0x60);
  if (local_e0 != 0) {
    LOCK();
    piVar14 = (int *)(local_e0 + 0x14);
    iVar12 = *piVar14;
    *piVar14 = *piVar14 + -1;
    UNLOCK();
    if (iVar12 == 1) {
      cv::Mat::deallocate(local_118);
    }
  }
  iVar12 = 0;
  uVar6 = 0;
  local_e0 = 0;
  local_108 = 0;
  local_f0 = 0;
  local_f8 = 0;
  local_100 = 0;
  if (0 < local_114) {
    lVar13 = 0;
    uVar5 = uVar6;
    do {
      *(undefined4 *)(local_d8 + lVar13) = 0;
      uVar7 = (int)uVar5 + 1;
      uVar5 = (ulonglong)uVar7;
      lVar13 = lVar13 + 4;
    } while ((int)uVar7 < local_114);
  }
  if (local_d0 != local_c8) {
    cv::fastFree(local_d0);
  }
  local_298 = uVar9;
  local_294 = uVar2;
  pMVar4 = (Mat *)FUN_18000e170(param_1,(uint *)local_268,&local_298,0,1);
  pMVar11 = param_9;
  if (param_9 != pMVar4) {
    FUN_180001780(param_9,pMVar4);
  }
  *(undefined4 *)(pMVar11 + 0x60) = *(undefined4 *)(pMVar4 + 0x60);
  if (local_230 != 0) {
    LOCK();
    piVar14 = (int *)(local_230 + 0x14);
    iVar1 = *piVar14;
    *piVar14 = *piVar14 + -1;
    UNLOCK();
    if (iVar1 == 1) {
      cv::Mat::deallocate(local_268);
    }
  }
  local_230 = 0;
  local_258 = 0;
  local_240 = 0;
  local_248 = 0;
  local_250 = 0;
  if (0 < local_264) {
    do {
      *(undefined4 *)(local_228 + uVar6) = 0;
      iVar12 = iVar12 + 1;
      uVar6 = uVar6 + 4;
    } while (iVar12 < local_264);
  }
  if (local_220 != local_218) {
    cv::fastFree(local_220);
  }
  lVar15 = 4;
  lVar13 = 4;
  uVar6 = 0;
  pMVar11 = param_7;
  local_290 = uVar8;
  local_28c = uVar3;
  do {
    pMVar4 = (Mat *)FUN_18000e170(param_1,(uint *)local_188,&local_290,param_3,1);
    if (pMVar11 != pMVar4) {
      FUN_180001780(pMVar11,pMVar4);
    }
    *(undefined4 *)(pMVar11 + 0x60) = *(undefined4 *)(pMVar4 + 0x60);
    if (local_150 != 0) {
      LOCK();
      piVar14 = (int *)(local_150 + 0x14);
      iVar12 = *piVar14;
      *piVar14 = *piVar14 + -1;
      UNLOCK();
      if (iVar12 == 1) {
        cv::Mat::deallocate(local_188);
      }
    }
    local_150 = 0;
    local_178 = 0;
    local_160 = 0;
    local_168 = 0;
    local_170 = 0;
    uVar5 = uVar6;
    uVar10 = uVar6;
    if (0 < local_184) {
      do {
        *(undefined4 *)(local_148 + uVar5) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar5 = uVar5 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_184);
    }
    if (local_140 != local_138) {
      cv::fastFree(local_140);
    }
    pMVar11 = pMVar11 + 0x68;
    lVar13 = lVar13 + -1;
    pMVar4 = param_8;
    local_280 = uVar9;
    local_27c = uVar2;
  } while (lVar13 != 0);
  do {
    pMVar11 = (Mat *)FUN_18000e170(param_1,(uint *)local_a8,&local_280,0,1);
    if (pMVar4 != pMVar11) {
      FUN_180001780(pMVar4,pMVar11);
    }
    *(undefined4 *)(pMVar4 + 0x60) = *(undefined4 *)(pMVar11 + 0x60);
    if (local_70 != 0) {
      LOCK();
      piVar14 = (int *)(local_70 + 0x14);
      iVar12 = *piVar14;
      *piVar14 = *piVar14 + -1;
      UNLOCK();
      if (iVar12 == 1) {
        cv::Mat::deallocate(local_a8);
      }
    }
    local_70 = 0;
    local_98 = 0;
    local_80 = 0;
    local_88 = 0;
    local_90 = 0;
    uVar5 = uVar6;
    uVar10 = uVar6;
    if (0 < local_a4) {
      do {
        *(undefined4 *)(local_68 + uVar5) = 0;
        uVar9 = (int)uVar10 + 1;
        uVar5 = uVar5 + 4;
        uVar10 = (ulonglong)uVar9;
      } while ((int)uVar9 < local_a4);
    }
    if (local_60 != local_58) {
      cv::fastFree(local_60);
    }
    lVar15 = lVar15 + -1;
    pMVar4 = pMVar4 + 0x68;
  } while (lVar15 != 0);
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002ce00'}
Evidence: `ev_efff0abb66a4d5f1d6bef7a93d7f6d624df7c2ea918004e74b6f9d461119d574`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_18002ce00(Mat *param_1,uint param_2)

{
  int *piVar1;
  int iVar2;
  Mat *this;
  ulonglong uVar3;
  ulonglong uVar4;
  uint uVar5;
  ulonglong uVar6;
  undefined8 uVar7;
  undefined1 auStackY_1d8 [32];
  uint local_1a8;
  int local_1a4;
  Mat *local_1a0;
  undefined8 local_198;
  uint local_190 [26];
  undefined4 local_128;
  int local_124;
  undefined4 local_120;
  undefined4 local_11c;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  longlong local_f0;
  undefined4 *local_e8;
  undefined8 *local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined4 local_c8;
  undefined4 local_b8;
  int local_b4;
  undefined4 local_b0;
  undefined4 local_ac;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  longlong local_80;
  undefined4 *local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined4 local_58;
  ulonglong local_48;
  
  local_198 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_1d8;
  memset(&local_128,0,0x68);
  local_128 = *(undefined4 *)param_1;
  local_124 = *(int *)(param_1 + 4);
  local_120 = *(undefined4 *)(param_1 + 8);
  local_11c = *(undefined4 *)(param_1 + 0xc);
  local_118 = *(undefined8 *)(param_1 + 0x10);
  local_110 = *(undefined8 *)(param_1 + 0x18);
  local_108 = *(undefined8 *)(param_1 + 0x20);
  local_100 = *(undefined8 *)(param_1 + 0x28);
  local_f8 = *(undefined8 *)(param_1 + 0x30);
  local_f0 = *(longlong *)(param_1 + 0x38);
  local_e8 = &local_120;
  uVar4 = 0;
  local_d0 = 0;
  local_d8 = 0;
  if (local_f0 != 0) {
    LOCK();
    *(int *)(local_f0 + 0x14) = *(int *)(local_f0 + 0x14) + 1;
    UNLOCK();
  }
  local_e0 = &local_d8;
  if (*(int *)(param_1 + 4) < 3) {
    local_d8 = **(undefined8 **)(param_1 + 0x48);
    local_d0 = *(undefined8 *)(*(longlong *)(param_1 + 0x48) + 8);
  }
  else {
    local_124 = 0;
    cv::Mat::copySize((Mat *)&local_128,param_1);
  }
  local_c8 = *(undefined4 *)(param_1 + 0x60);
  FUN_18002f300((longlong)&local_128,param_2);
  if (local_f0 != 0) {
    LOCK();
    piVar1 = (int *)(local_f0 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_128);
    }
  }
  local_f0 = 0;
  local_118 = 0;
  local_100 = 0;
  local_108 = 0;
  local_110 = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < local_124) {
    do {
      *(undefined4 *)(uVar3 + (longlong)local_e8) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < local_124);
  }
  if (local_e0 != &local_d8) {
    cv::fastFree(local_e0);
  }
  iVar2 = **(int **)(param_1 + 0x40);
  uVar5 = (*(int **)(param_1 + 0x40))[1];
  uVar7 = 0x68;
  memset(&local_b8,0,0x68);
  local_1a8 = uVar5;
  local_1a4 = iVar2 + param_2 * 2;
  this = (Mat *)FUN_180011dc0(local_190,&local_1a8,uVar7,
                              *(longlong *)(param_1 + 0x10) -
                              (longlong)(int)param_2 * *(longlong *)(param_1 + 0x50),
                              (longlong)(int)*(longlong *)(param_1 + 0x50));
  local_b8 = *(undefined4 *)this;
  local_b4 = *(int *)(this + 4);
  local_b0 = *(undefined4 *)(this + 8);
  local_ac = *(undefined4 *)(this + 0xc);
  local_a8 = *(undefined8 *)(this + 0x10);
  local_a0 = *(undefined8 *)(this + 0x18);
  local_98 = *(undefined8 *)(this + 0x20);
  local_90 = *(undefined8 *)(this + 0x28);
  local_88 = *(undefined8 *)(this + 0x30);
  local_80 = *(longlong *)(this + 0x38);
  local_78 = &local_b0;
  local_70 = &local_68;
  local_60 = 0;
  local_68 = 0;
  if (local_80 != 0) {
    LOCK();
    *(int *)(local_80 + 0x14) = *(int *)(local_80 + 0x14) + 1;
    UNLOCK();
  }
  local_1a0 = this;
  if (*(int *)(this + 4) < 3) {
    local_68 = **(undefined8 **)(this + 0x48);
    local_60 = *(undefined8 *)(*(longlong *)(this + 0x48) + 8);
  }
  else {
    local_b4 = 0;
    cv::Mat::copySize((Mat *)&local_b8,this);
  }
  local_58 = 0;
  if (*(longlong *)(this + 0x38) != 0) {
    LOCK();
    piVar1 = (int *)(*(longlong *)(this + 0x38) + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(this);
    }
  }
  *(undefined8 *)(this + 0x38) = 0;
  *(undefined8 *)(this + 0x10) = 0;
  *(undefined8 *)(this + 0x28) = 0;
  *(undefined8 *)(this + 0x20) = 0;
  *(undefined8 *)(this + 0x18) = 0;
  uVar3 = uVar4;
  uVar6 = uVar4;
  if (0 < *(int *)(this + 4)) {
    do {
      *(undefined4 *)(uVar3 + *(longlong *)(this + 0x40)) = 0;
      uVar5 = (int)uVar6 + 1;
      uVar3 = uVar3 + 4;
      uVar6 = (ulonglong)uVar5;
    } while ((int)uVar5 < *(int *)(this + 4));
  }
  if (*(Mat **)(this + 0x48) != this + 0x50) {
    cv::fastFree(*(Mat **)(this + 0x48));
  }
  FUN_18002c920((longlong)&local_b8,param_2);
  if (local_80 != 0) {
    LOCK();
    piVar1 = (int *)(local_80 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate((Mat *)&local_b8);
    }
  }
  local_80 = 0;
  local_a8 = 0;
  local_90 = 0;
  local_98 = 0;
  local_a0 = 0;
  uVar3 = uVar4;
  if (0 < local_b4) {
    do {
      *(undefined4 *)((longlong)local_78 + uVar3) = 0;
      uVar5 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar5;
      uVar3 = uVar3 + 4;
    } while ((int)uVar5 < local_b4);
  }
  if (local_70 != &local_68) {
    cv::fastFree(local_70);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x180023d80'}
Evidence: `ev_f1fc8a1cd8fba07602499b568845920f99c683f47391f140fbac3db136220720`
```c

float FUN_180023d80(int param_1,int param_2,longlong *param_3,longlong *param_4)

{
  float fVar1;
  longlong lVar2;
  longlong lVar3;
  longlong lVar4;
  float *pfVar5;
  float *pfVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  float *pfVar10;
  longlong lVar11;
  longlong lVar12;
  float fVar13;
  
  fVar13 = 0.0;
  lVar2 = *param_3;
  lVar9 = (longlong)*(int *)(param_3[4] + (longlong)param_1 * 4);
  lVar7 = (longlong)*(int *)(param_3[4] + 4 + (longlong)param_1 * 4);
  lVar3 = (longlong)*(int *)(param_3[5] + (longlong)param_2 * 4);
  lVar8 = (longlong)*(int *)(param_3[5] + 4 + (longlong)param_2 * 4);
  if (lVar3 < lVar8) {
    pfVar10 = (float *)(param_3[1] + lVar3 * 4);
    lVar8 = lVar8 - lVar3;
    do {
      fVar1 = *pfVar10;
      lVar3 = (longlong)(*(int *)((param_3[3] - param_3[1]) + (longlong)pfVar10) * (int)param_4[1])
              + *param_4;
      if (lVar9 < lVar7) {
        lVar12 = lVar9;
        if (3 < lVar7 - lVar9) {
          lVar11 = param_3[2] - lVar2;
          lVar4 = ((lVar7 - lVar9) - 4U >> 2) + 1;
          lVar12 = lVar9 + lVar4 * 4;
          pfVar5 = (float *)(lVar2 + 4 + lVar9 * 4);
          do {
            pfVar6 = pfVar5 + 4;
            fVar13 = fVar1 * pfVar5[-1] *
                     *(float *)(lVar3 + (longlong)*(int *)(lVar11 + -4 + (longlong)pfVar5) * 4) +
                     fVar13 + fVar1 * *pfVar5 *
                              *(float *)(lVar3 + (longlong)
                                                 *(int *)(lVar11 + -0x10 + (longlong)pfVar6) * 4) +
                     fVar1 * pfVar5[1] *
                     *(float *)(lVar3 + (longlong)*(int *)(lVar11 + -0xc + (longlong)pfVar6) * 4) +
                     fVar1 * pfVar5[2] *
                     *(float *)(lVar3 + (longlong)*(int *)(lVar11 + -8 + (longlong)pfVar6) * 4);
            lVar4 = lVar4 + -1;
            pfVar5 = pfVar6;
          } while (lVar4 != 0);
        }
        if (lVar12 < lVar7) {
          lVar4 = lVar7 - lVar12;
          pfVar5 = (float *)(lVar2 + lVar12 * 4);
          do {
            fVar13 = fVar13 + fVar1 * *pfVar5 *
                              *(float *)(lVar3 + (longlong)
                                                 *(int *)((param_3[2] - lVar2) + (longlong)pfVar5) *
                                                 4);
            lVar4 = lVar4 + -1;
            pfVar5 = pfVar5 + 1;
          } while (lVar4 != 0);
        }
      }
      pfVar10 = pfVar10 + 1;
      lVar8 = lVar8 + -1;
    } while (lVar8 != 0);
  }
  return fVar13;
}


```

## procedure_assembly {'procedure': '0x180012b70'}
Evidence: `ev_f407aba66df2a758d2f8dd0943adfc11f68a1f6010345dd21ee35fc8d739a496`
```c
0x180012b70: MOV RAX, RSP
0x180012b73: PUSH RBP
0x180012b74: PUSH RSI
0x180012b75: PUSH RDI
0x180012b76: PUSH R12
0x180012b78: PUSH R13
0x180012b7a: PUSH R14
0x180012b7c: PUSH R15
0x180012b7e: LEA RBP, [RAX + -0x828]
0x180012b85: SUB RSP, 0x8f0
0x180012b8c: MOV qword ptr [RBP + 0x1e0], -0x2
0x180012b97: MOV qword ptr [RAX + 0x10], RBX
0x180012b9b: MOVAPS xmmword ptr [RAX + -0x48], XMM6
0x180012b9f: MOVAPS xmmword ptr [RAX + -0x58], XMM7
0x180012ba3: MOVAPS xmmword ptr [RAX + -0x68], XMM8
0x180012ba8: MOVAPS xmmword ptr [RAX + -0x78], XMM9
0x180012bad: MOVAPS xmmword ptr [RAX + -0x88], XMM10
0x180012bb5: MOVAPS xmmword ptr [RAX + -0x98], XMM11
0x180012bbd: MOVAPS xmmword ptr [RAX + -0xa8], XMM12
0x180012bc5: MOVAPS xmmword ptr [RAX + -0xb8], XMM13
0x180012bcd: MOVAPS xmmword ptr [RAX + -0xc8], XMM14
0x180012bd5: MOVAPS xmmword ptr [RAX + -0xd8], XMM15
0x180012bdd: MOV RAX, qword ptr [0x18008b080]
0x180012be4: XOR RAX, RSP
0x180012be7: MOV qword ptr [RBP + 0x740], RAX
0x180012bee: MOV RDI, R9
0x180012bf1: MOV qword ptr [RBP + -0x10], R9
0x180012bf5: MOV qword ptr [RSP + 0x68], R8
0x180012bfa: MOV RBX, RDX
0x180012bfd: MOV R14, RCX
0x180012c00: MOV qword ptr [RBP + -0x60], RCX
0x180012c04: MOV EAX, dword ptr [RCX + 0x5e4]
0x180012c0a: MOV dword ptr [RSP + 0x58], EAX
0x180012c0e: MOV EAX, dword ptr [RCX + 0x5e8]
0x180012c14: MOV dword ptr [RSP + 0x50], EAX
0x180012c18: MOVSXD RSI, dword ptr [RCX + 0x5ec]
0x180012c1f: MOV dword ptr [RSP + 0x5c], ESI
0x180012c23: MOV EAX, dword ptr [RCX + 0x5f0]
0x180012c29: MOV dword ptr [RSP + 0x48], EAX
0x180012c2d: XOR R12D, R12D
0x180012c30: MOV EAX, dword ptr [RBP + 0x858]
0x180012c36: TEST EAX, EAX
0x180012c38: JZ 0x180012cbf
0x180012c3e: CMP EAX, 0x1
0x180012c41: JZ 0x180012cb4
0x180012c43: MOV qword ptr [RBP + -0x20], R12
0x180012c47: MOV qword ptr [RBP + -0x18], R12
0x180012c4b: LEA EDX, [R12 + 0x31]
0x180012c50: LEA RCX, [RBP + -0x20]
0x180012c54: CALL qword ptr [0x1800783e8]
0x180012c5a: MOVUPS XMM0, xmmword ptr [0x18007bb88]
0x180012c61: MOVUPS xmmword ptr [RAX], XMM0
0x180012c64: MOVUPS XMM1, xmmword ptr [0x18007bb98]
0x180012c6b: MOVUPS xmmword ptr [RAX + 0x10], XMM1
0x180012c6f: MOVUPS XMM0, xmmword ptr [0x18007bba8]
0x180012c76: MOVUPS xmmword ptr [RAX + 0x20], XMM0
0x180012c7a: MOVZX ECX, byte ptr [0x18007bbb8]
0x180012c81: MOV byte ptr [RAX + 0x30], CL
0x180012c84: MOV dword ptr [RSP + 0x20], 0x171
0x180012c8c: LEA R9, [0x18007bb50]
0x180012c93: LEA R8, [0x18007bb68]
0x180012c9a: LEA RDX, [RBP + -0x20]
0x180012c9e: MOV ECX, 0xffffff29
0x180012ca3: CALL qword ptr [0x1800783f8]
0x180012ca9: NOP
0x180012caa: LEA RCX, [RBP + -0x20]
0x180012cae: CALL qword ptr [0x1800783f0]
0x180012cb4: MOVSD XMM14, qword ptr [R14 + 0x88]
0x180012cbd: JMP 0x180012cc8
0x180012cbf: MOVSD XMM14, qword ptr [RCX + 0x80]
0x180012cc8: MOVSD XMM15, qword ptr [R14 + 0xa0]
0x180012cd1: MOV EAX, dword ptr [R14 + 0x9c]
0x180012cd8: MOV dword ptr [RSP + 0x4c], EAX
0x180012cdc: LEA R15, [0x180001750]
0x180012ce3: MOV qword ptr [RSP + 0x20], R15
0x180012ce8: LEA R9, [0x180006700]
0x180012cef: MOV EDX, 0x68
0x180012cf4: LEA R8D, [RDX + -0x66]
0x180012cf8: LEA RCX, [RBP + 0x4d0]
0x180012cff: CALL 0x1800726b8
0x180012d04: NOP
0x180012d05: MOV qword ptr [RSP + 0x20], R15
0x180012d0a: LEA R9, [0x180006700]
0x180012d11: MOV EDX, 0x68
0x180012d16: LEA R8D, [RDX + -0x66]
0x180012d1a: LEA RCX, [RBP + 0x670]
0x180012d21: CALL 0x1800726b8
0x180012d26: NOP
0x180012d27: XOR EDX, EDX
0x180012d29: LEA R8D, [RDX + 0x68]
0x180012d2d: LEA RCX, [RBP + 0x320]
0x180012d34: CALL 0x180072a62
0x180012d39: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x180012d41: MOVDQA xmmword ptr [RBP + 0x320], XMM0
0x180012d49: XORPS XMM1, XMM1
0x180012d4c: MOVDQA xmmword ptr [RBP + 0x330], XMM1
0x180012d54: XORPS XMM0, XMM0
0x180012d57: MOVDQA xmmword ptr [RBP + 0x340], XMM0
0x180012d5f: MOVDQA xmmword ptr [RBP + 0x350], XMM1
0x180012d67: LEA RAX, [RBP + 0x328]
0x180012d6e: MOV qword ptr [RBP + 0x360], RAX
0x180012d75: LEA RAX, [RBP + 0x370]
0x180012d7c: MOV qword ptr [RBP + 0x368], RAX
0x180012d83: MOV qword ptr [RBP + 0x378], R12
0x180012d8a: MOV qword ptr [RBP + 0x370], R12
0x180012d91: MOV dword ptr [RBP + 0x380], R12D
0x180012d98: XOR EAX, EAX
0x180012d9a: MOV qword ptr [RBP + 0x280], RAX
0x180012da1: MOV qword ptr [RBP + 0x288], RAX
0x180012da8: MOV qword ptr [RBP + 0x290], RAX
0x180012daf: MOV qword ptr [RBP + 0x298], RAX
0x180012db6: MOV qword ptr [RBP + 0x2a0], RAX
0x180012dbd: MOV qword ptr [RBP + 0x280], RBX
0x180012dc4: MOV dword ptr [RBP + 0x288], 0x20
0x180012dce: MOV qword ptr [RBP + 0x290], R12
0x180012dd5: MOV RAX, qword ptr [RBX + 0x8]
0x180012dd9: ADD RAX, qword ptr [RBX]
0x180012ddc: MOV qword ptr [RBP + 0x298], RAX
0x180012de3: LEA RAX, [RBP + 0x320]
0x180012dea: MOV qword ptr [RSP + 0x20], RAX
0x180012def: LEA R9, [RBP + 0x670]
0x180012df6: LEA R8, [RBP + 0x4d0]
0x180012dfd: MOV RDX, R14
0x180012e00: LEA RCX, [RBP + 0x280]
0x180012e07: CALL 0x180023f10
0x180012e0c: XOR EDX, EDX
0x180012e0e: LEA R8D, [RDX + 0x68]
0x180012e12: LEA RCX, [RBP + 0x2b0]
0x180012e19: CALL 0x180072a62
0x180012e1e: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x180012e26: MOVDQA xmmword ptr [RBP + 0x2b0], XMM0
0x180012e2e: XORPS XMM1, XMM1
0x180012e31: MOVDQA xmmword ptr [RBP + 0x2c0], XMM1
0x180012e39: XORPS XMM0, XMM0
0x180012e3c: MOVDQA xmmword ptr [RBP + 0x2d0], XMM0
0x180012e44: MOVDQA xmmword ptr [RBP + 0x2e0], XMM1
0x180012e4c: LEA RAX, [RBP + 0x2b8]
0x180012e53: MOV qword ptr [RBP + 0x2f0], RAX
0x180012e5a: LEA RAX, [RBP + 0x300]
0x180012e61: MOV qword ptr [RBP + 0x2f8], RAX
0x180012e68: MOV qword ptr [RBP + 0x308], R12
0x180012e6f: MOV qword ptr [RBP + 0x300], R12
0x180012e76: MOV dword ptr [RBP + 0x310], R12D
0x180012e7d: XOR EDX, EDX
0x180012e7f: LEA R8D, [RDX + 0x68]
0x180012e83: LEA RCX, [RBP + 0x390]
0x180012e8a: CALL 0x180072a62
0x180012e8f: MOVDQA XMM0, xmmword ptr [0x18007c5e0]
0x180012e97: MOVDQA xmmword ptr [RBP + 0x390], XMM0
0x180012e9f: XORPS XMM1, XMM1
0x180012ea2: MOVDQA xmmword ptr [RBP + 0x3a0], XMM1
0x180012eaa: XORPS XMM0, XMM0
0x180012ead: MOVDQA xmmword ptr [RBP + 0x3b0], XMM0
0x180012eb5: MOVDQA xmmword ptr [RBP + 0x3c0], XMM1
0x180012ebd: LEA RAX, [RBP + 0x398]
0x180012ec4: MOV qword ptr [RBP + 0x3d0], RAX
0x180012ecb: LEA RAX, [RBP + 0x3e0]
0x180012ed2: MOV qword ptr [RBP + 0x3d8], RAX
0x180012ed9: MOV qword ptr [RBP + 0x3e8], R12
0x180012ee0: MOV qword ptr [RBP + 0x3e0], R12
0x180012ee7: MOV dword ptr [RBP + 0x3f0], R12D
0x180012eee: MOV R8, qword ptr [RBP + 0x330]
0x180012ef5: MOV qword ptr [RBP + -0x50], R8
0x180012ef9: MOV qword ptr [RBP + 0x160], R8
0x180012f00: MOV R9D, dword ptr [RBP + 0x370]
0x180012f07: MOV dword ptr [RSP + 0x64], R9D
0x180012f0c: MOV dword ptr [RBP + 0x168], R9D
0x180012f13: MOV RDX, qword ptr [RBP + 0x680]
0x180012f1a: MOV qword ptr [RSP + 0x70], RDX
0x180012f1f: MOV qword ptr [RBP + 0x180], RDX
0x180012f26: MOV ECX, dword ptr [RBP + 0x6c0]
0x180012f2c: MOV dword ptr [RSP + 0x60], ECX
0x180012f30: MOV dword ptr [RBP + 0x188], ECX
0x180012f36: MOV R10, qword ptr [RBP + 0x4e0]
0x180012f3d: MOV qword ptr [RBP + -0x80], R10
0x180012f41: MOV qword ptr [RBP + 0x130], R10
0x180012f48: MOV R11D, dword ptr [RBP + 0x520]
0x180012f4f: MOV dword ptr [RSP + 0x54], R11D
0x180012f54: MOV dword ptr [RBP + 0x138], R11D
0x180012f5b: MOV RBX, qword ptr [RBP + 0x850]
0x180012f62: MOV RAX, qword ptr [RBX + 0x10]
0x180012f66: MOV qword ptr [RBP + 0x40], RAX
0x180012f6a: MOV RAX, qword ptr [RBX + 0x50]
0x180012f6e: MOV dword ptr [RBP + 0x48], EAX
0x180012f71: MOV RAX, qword ptr [RDI + 0x10]
0x180012f75: MOV qword ptr [RBP + 0x30], RAX
0x180012f79: MOV RAX, qword ptr [RDI + 0x50]
0x180012f7d: MOV dword ptr [RBP + 0x38], EAX
0x180012f80: MOVUPS XMM9, xmmword ptr [R14 + 0x608]
0x180012f88: MOVUPS XMM10, xmmword ptr [R14 + 0x618]
0x180012f90: MOVUPS XMM11, xmmword ptr [R14 + 0x628]
0x180012f98: MOV R15D, R12D
0x180012f9b: MOV R13, RSI
0x180012f9e: MOV qword ptr [RSP + 0x40], R12
0x180012fa3: MOVSXD RAX, dword ptr [RSP + 0x48]
0x180012fa8: MOV qword ptr [RBP + -0x30], RAX
0x180012fac: TEST RAX, RAX
0x180012faf: JLE 0x180013110
0x180012fb5: MOVAPS XMM12, xmmword ptr [RBP + 0x40]
0x180012fba: MOVAPS XMM13, xmmword ptr [RBP + 0x30]
0x180012fbf: MOVSS XMM8, dword ptr [0x18007c518]
0x180012fc8: NOP dword ptr [RAX + RAX*0x1]
0x180012fd0: XOR R14D, R14D
0x180012fd3: XOR EDI, EDI
0x180012fd5: TEST R13, R13
0x180012fd8: JLE 0x1800130f8
0x180012fde: MOVDQA xmmword ptr [RBP + 0x1f0], XMM13
0x180012fe7: MOVAPS xmmword ptr [RBP + 0x220], XMM9
0x180012fef: MOVAPS xmmword ptr [RBP + 0x230], XMM10
0x180012ff7: MOVAPS xmmword ptr [RBP + 0x240], XMM11
0x180012fff: MOVDQA xmmword ptr [RBP + 0x150], XMM12
0x180013008: MOVAPS xmmword ptr [RBP + 0x250], XMM9
0x180013010: MOVAPS xmmword ptr [RBP + 0x260], XMM10
0x180013018: MOVAPS xmmword ptr [RBP + 0x270], XMM11
0x180013020: XORPS XMM7, XMM7
0x180013023: CVTSD2SS XMM7, XMM14
0x180013028: MOVSXD RCX, ECX
0x18001302b: MOV RBX, R12
0x18001302e: IMUL RBX, RCX
0x180013032: ADD RBX, RDX
0x180013035: MOVSXD RAX, R11D
0x180013038: SUB RAX, RCX
0x18001303b: IMUL RAX, R12
0x18001303f: SUB RAX, RDX
0x180013042: ADD RAX, R10
0x180013045: MOV qword ptr [RBP + -0x70], RAX
0x180013049: MOVSXD RAX, R9D
0x18001304c: IMUL RAX, R12
0x180013050: ADD RAX, R8
0x180013053: MOV qword ptr [RBP + -0x40], RAX
0x180013057: MOV R12, qword ptr [RBP + -0x70]
0x18001305b: NOP dword ptr [RAX + RAX*0x1]
0x180013060: LEA R9, [RBP + 0x150]
0x180013067: LEA R8, [RBP + 0x250]
0x18001306e: MOV EDX, R15D
0x180013071: MOV ECX, R14D
0x180013074: CALL 0x180023d80
0x180013079: MOVAPS XMM6, XMM0
0x18001307c: MOV SIL, 0x1
0x18001307f: LEA R9, [RBP + 0x1f0]
0x180013086: LEA R8, [RBP + 0x220]
0x18001308d: MOV EDX, R15D
0x180013090: MOV ECX, R14D
0x180013093: CALL 0x180023d80
0x180013098: ADDSS XMM0, XMM8
0x18001309d: COMISS XMM7, XMM6
0x1800130a0: JBE 0x1800130a8
0x1800130a2: XOR SIL, SIL
0x1800130a5: XORPS XMM0, XMM0
0x1800130a8: MOVSS dword ptr [R12 + RBX*0x1], XMM0
0x1800130ae: MOVZX EAX, SIL
0x1800130b2: MOVD XMM0, EAX
0x1800130b6: CVTDQ2PS XMM0, XMM0
0x1800130b9: MOVSS dword ptr [RBX], XMM0
0x1800130bd: MOV RAX, qword ptr [RBP + -0x40]
0x1800130c1: MOV byte ptr [RDI + RAX*0x1], SIL
0x1800130c5: INC R14D
0x1800130c8: INC RDI
0x1800130cb: ADD RBX, 0x4
0x1800130cf: CMP RDI, R13
0x1800130d2: JL 0x180013060
0x1800130d4: MOV R12, qword ptr [RSP + 0x40]
0x1800130d9: MOV RDX, qword ptr [RSP + 0x70]
0x1800130de: MOV RAX, qword ptr [RBP + -0x30]
0x1800130e2: MOV R8, qword ptr [RBP + -0x50]
0x1800130e6: MOV R9D, dword ptr [RSP + 0x64]
0x1800130eb: MOV ECX, dword ptr [RSP + 0x60]
0x1800130ef: MOV R10, qword ptr [RBP + -0x80]
0x1800130f3: MOV R11D, dword ptr [RSP + 0x54]
0x1800130f8: INC R15D
0x1800130fb: INC R12
0x1800130fe: MOV qword ptr [RSP + 0x40], R12
0x180013103: CMP R12, RAX
0x180013106: JL 0x180012fd0
0x18001310c: MOV R14, qword ptr [RBP + -0x60]
0x180013110: MOVSXD RAX, dword ptr [RSP + 0x4c]
0x180013115: TEST RAX, RAX
0x180013118: JLE 0x18001379b
0x18001311e: XORPS XMM6, XMM6
0x180013121: XOR R15D, R15D
0x180013124: LEA R12, [0x180001750]
0x18001312b: NOP dword ptr [RAX + RAX*0x1]
0x180013130: MOV EAX, dword ptr [RBP + 0x4d0]
0x180013136: MOV dword ptr [RBP + 0x5a0], EAX
0x18001313c: MOV EDX, dword ptr [RBP + 0x4d4]
0x180013142: MOV dword ptr [RBP + 0x5a4], EDX
0x180013148: MOV EAX, dword ptr [RBP + 0x4d8]
0x18001314e: MOV dword ptr [RBP + 0x5a8], EAX
0x180013154: MOV EAX, dword ptr [RBP + 0x4dc]
0x18001315a: MOV dword ptr [RBP + 0x5ac], EAX
0x180013160: MOV RAX, qword ptr [RBP + 0x4e0]
0x180013167: MOV qword ptr [RBP + 0x5b0], RAX
0x18001316e: MOV RAX, qword ptr [RBP + 0x4e8]
0x180013175: MOV qword ptr [RBP + 0x5b8], RAX
0x18001317c: MOV RAX, qword ptr [RBP + 0x4f0]
0x180013183: MOV qword ptr [RBP + 0x5c0], RAX
0x18001318a: MOV RAX, qword ptr [RBP + 0x4f8]
0x180013191: MOV qword ptr [RBP + 0x5c8], RAX
0x180013198: MOV RAX, qword ptr [RBP + 0x500]
0x18001319f: MOV qword ptr [RBP + 0x5d0], RAX
0x1800131a6: MOV RCX, qword ptr [RBP + 0x508]
0x1800131ad: MOV qword ptr [RBP + 0x5d8], RCX
0x1800131b4: LEA RAX, [RBP + 0x5a8]
0x1800131bb: MOV qword ptr [RBP + 0x5e0], RAX
0x1800131c2: LEA R8, [RBP + 0x5f0]
0x1800131c9: MOV qword ptr [RBP + 0x5e8], R8
0x1800131d0: MOV qword ptr [RBP + 0x5f8], R15
0x1800131d7: MOV qword ptr [RBP + 0x5f0], R15
0x1800131de: TEST RCX, RCX
0x1800131e1: JZ 0x1800131fa
0x1800131e3: MOV EAX, 0x1
0x1800131e8: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800131ed: MOV EDX, dword ptr [RBP + 0x4d4]
0x1800131f3: MOV R8, qword ptr [RBP + 0x5e8]
0x1800131fa: CMP EDX, 0x2
0x1800131fd: JG 0x180013224
0x1800131ff: MOV RAX, qword ptr [RBP + 0x518]
0x180013206: MOV RCX, qword ptr [RAX]
0x180013209: MOV qword ptr [R8], RCX
0x18001320c: MOV RAX, qword ptr [RBP + 0x518]
0x180013213: MOV RCX, qword ptr [RAX + 0x8]
0x180013217: MOV RAX, qword ptr [RBP + 0x5e8]
0x18001321e: MOV qword ptr [RAX + 0x8], RCX
0x180013222: JMP 0x18001323f
0x180013224: MOV dword ptr [RBP + 0x5a4], R15D
0x18001322b: LEA RDX, [RBP + 0x4d0]
0x180013232: LEA RCX, [RBP + 0x5a0]
0x180013239: CALL qword ptr [0x180078290]
0x18001323f: MOV EAX, dword ptr [RBP + 0x530]
0x180013245: MOV dword ptr [RBP + 0x600], EAX
0x18001324b: MOV EAX, dword ptr [RBP + 0x670]
0x180013251: MOV dword ptr [RBP + 0x608], EAX
0x180013257: MOV EDX, dword ptr [RBP + 0x674]
0x18001325d: MOV dword ptr [RBP + 0x60c], EDX
0x180013263: MOV EAX, dword ptr [RBP + 0x678]
0x180013269: MOV dword ptr [RBP + 0x610], EAX
0x18001326f: MOV EAX, dword ptr [RBP + 0x67c]
0x180013275: MOV dword ptr [RBP + 0x614], EAX
0x18001327b: MOV RAX, qword ptr [RBP + 0x680]
0x180013282: MOV qword ptr [RBP + 0x618], RAX
0x180013289: MOV RAX, qword ptr [RBP + 0x688]
0x180013290: MOV qword ptr [RBP + 0x620], RAX
0x180013297: MOV RAX, qword ptr [RBP + 0x690]
0x18001329e: MOV qword ptr [RBP + 0x628], RAX
0x1800132a5: MOV RAX, qword ptr [RBP + 0x698]
0x1800132ac: MOV qword ptr [RBP + 0x630], RAX
0x1800132b3: MOV RAX, qword ptr [RBP + 0x6a0]
0x1800132ba: MOV qword ptr [RBP + 0x638], RAX
0x1800132c1: MOV RCX, qword ptr [RBP + 0x6a8]
0x1800132c8: MOV qword ptr [RBP + 0x640], RCX
0x1800132cf: LEA RAX, [RBP + 0x610]
0x1800132d6: MOV qword ptr [RBP + 0x648], RAX
0x1800132dd: LEA R8, [RBP + 0x658]
0x1800132e4: MOV qword ptr [RBP + 0x650], R8
0x1800132eb: MOV qword ptr [RBP + 0x660], R15
0x1800132f2: MOV qword ptr [RBP + 0x658], R15
0x1800132f9: TEST RCX, RCX
0x1800132fc: JZ 0x180013315
0x1800132fe: MOV EAX, 0x1
0x180013303: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180013308: MOV EDX, dword ptr [RBP + 0x674]
0x18001330e: MOV R8, qword ptr [RBP + 0x650]
0x180013315: CMP EDX, 0x2
0x180013318: JG 0x18001333f
0x18001331a: MOV RAX, qword ptr [RBP + 0x6b8]
0x180013321: MOV RCX, qword ptr [RAX]
0x180013324: MOV qword ptr [R8], RCX
0x180013327: MOV RAX, qword ptr [RBP + 0x6b8]
0x18001332e: MOV RCX, qword ptr [RAX + 0x8]
0x180013332: MOV RAX, qword ptr [RBP + 0x650]
0x180013339: MOV qword ptr [RAX + 0x8], RCX
0x18001333d: JMP 0x18001335a
0x18001333f: MOV dword ptr [RBP + 0x60c], R15D
0x180013346: LEA RDX, [RBP + 0x670]
0x18001334d: LEA RCX, [RBP + 0x608]
0x180013354: CALL qword ptr [0x180078290]
0x18001335a: MOV EAX, dword ptr [RBP + 0x6d0]
0x180013360: MOV dword ptr [RBP + 0x668], EAX
0x180013366: MOV EAX, dword ptr [RBP + 0x538]
0x18001336c: MOV dword ptr [RBP + 0x400], EAX
0x180013372: MOV EDX, dword ptr [RBP + 0x53c]
0x180013378: MOV dword ptr [RBP + 0x404], EDX
0x18001337e: MOV EAX, dword ptr [RBP + 0x540]
0x180013384: MOV dword ptr [RBP + 0x408], EAX
0x18001338a: MOV EAX, dword ptr [RBP + 0x544]
0x180013390: MOV dword ptr [RBP + 0x40c], EAX
0x180013396: MOV RAX, qword ptr [RBP + 0x548]
0x18001339d: MOV qword ptr [RBP + 0x410], RAX
0x1800133a4: MOV RAX, qword ptr [RBP + 0x550]
0x1800133ab: MOV qword ptr [RBP + 0x418], RAX
0x1800133b2: MOV RAX, qword ptr [RBP + 0x558]
0x1800133b9: MOV qword ptr [RBP + 0x420], RAX
0x1800133c0: MOV RAX, qword ptr [RBP + 0x560]
0x1800133c7: MOV qword ptr [RBP + 0x428], RAX
0x1800133ce: MOV RAX, qword ptr [RBP + 0x568]
0x1800133d5: MOV qword ptr [RBP + 0x430], RAX
0x1800133dc: MOV RCX, qword ptr [RBP + 0x570]
0x1800133e3: MOV qword ptr [RBP + 0x438], RCX
0x1800133ea: LEA RAX, [RBP + 0x408]
0x1800133f1: MOV qword ptr [RBP + 0x440], RAX
0x1800133f8: LEA R8, [RBP + 0x450]
0x1800133ff: MOV qword ptr [RBP + 0x448], R8
0x180013406: MOV qword ptr [RBP + 0x458], R15
0x18001340d: MOV qword ptr [RBP + 0x450], R15
0x180013414: TEST RCX, RCX
0x180013417: JZ 0x180013430
0x180013419: MOV EAX, 0x1
0x18001341e: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180013423: MOV EDX, dword ptr [RBP + 0x53c]
0x180013429: MOV R8, qword ptr [RBP + 0x448]
0x180013430: CMP EDX, 0x2
0x180013433: JG 0x18001345a
0x180013435: MOV RAX, qword ptr [RBP + 0x580]
0x18001343c: MOV RCX, qword ptr [RAX]
0x18001343f: MOV qword ptr [R8], RCX
0x180013442: MOV RAX, qword ptr [RBP + 0x580]
0x180013449: MOV RCX, qword ptr [RAX + 0x8]
0x18001344d: MOV RAX, qword ptr [RBP + 0x448]
0x180013454: MOV qword ptr [RAX + 0x8], RCX
0x180013458: JMP 0x180013475
0x18001345a: MOV dword ptr [RBP + 0x404], R15D
0x180013461: LEA RDX, [RBP + 0x538]
0x180013468: LEA RCX, [RBP + 0x400]
0x18001346f: CALL qword ptr [0x180078290]
0x180013475: MOV EAX, dword ptr [RBP + 0x598]
0x18001347b: MOV dword ptr [RBP + 0x460], EAX
0x180013481: MOV EAX, dword ptr [RBP + 0x6d8]
0x180013487: MOV dword ptr [RBP + 0x468], EAX
0x18001348d: MOV EDX, dword ptr [RBP + 0x6dc]
0x180013493: MOV dword ptr [RBP + 0x46c], EDX
0x180013499: MOV EAX, dword ptr [RBP + 0x6e0]
0x18001349f: MOV dword ptr [RBP + 0x470], EAX
0x1800134a5: MOV EAX, dword ptr [RBP + 0x6e4]
0x1800134ab: MOV dword ptr [RBP + 0x474], EAX
0x1800134b1: MOV RAX, qword ptr [RBP + 0x6e8]
0x1800134b8: MOV qword ptr [RBP + 0x478], RAX
0x1800134bf: MOV RAX, qword ptr [RBP + 0x6f0]
0x1800134c6: MOV qword ptr [RBP + 0x480], RAX
0x1800134cd: MOV RAX, qword ptr [RBP + 0x6f8]
0x1800134d4: MOV qword ptr [RBP + 0x488], RAX
0x1800134db: MOV RAX, qword ptr [RBP + 0x700]
0x1800134e2: MOV qword ptr [RBP + 0x490], RAX
0x1800134e9: MOV RAX, qword ptr [RBP + 0x708]
0x1800134f0: MOV qword ptr [RBP + 0x498], RAX
0x1800134f7: MOV RCX, qword ptr [RBP + 0x710]
0x1800134fe: MOV qword ptr [RBP + 0x4a0], RCX
0x180013505: LEA RAX, [RBP + 0x470]
0x18001350c: MOV qword ptr [RBP + 0x4a8], RAX
0x180013513: LEA R8, [RBP + 0x4b8]
0x18001351a: MOV qword ptr [RBP + 0x4b0], R8
0x180013521: MOV qword ptr [RBP + 0x4c0], R15
0x180013528: MOV qword ptr [RBP + 0x4b8], R15
0x18001352f: TEST RCX, RCX
0x180013532: JZ 0x18001354b
0x180013534: MOV EAX, 0x1
0x180013539: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x18001353e: MOV EDX, dword ptr [RBP + 0x6dc]
0x180013544: MOV R8, qword ptr [RBP + 0x4b0]
0x18001354b: CMP EDX, 0x2
0x18001354e: JG 0x180013575
0x180013550: MOV RAX, qword ptr [RBP + 0x720]
0x180013557: MOV RCX, qword ptr [RAX]
0x18001355a: MOV qword ptr [R8], RCX
0x18001355d: MOV RAX, qword ptr [RBP + 0x720]
0x180013564: MOV RCX, qword ptr [RAX + 0x8]
0x180013568: MOV RAX, qword ptr [RBP + 0x4b0]
0x18001356f: MOV qword ptr [RAX + 0x8], RCX
0x180013573: JMP 0x180013590
0x180013575: MOV dword ptr [RBP + 0x46c], R15D
0x18001357c: LEA RDX, [RBP + 0x6d8]
0x180013583: LEA RCX, [RBP + 0x468]
0x18001358a: CALL qword ptr [0x180078290]
0x180013590: MOV EAX, dword ptr [RBP + 0x738]
0x180013596: MOV dword ptr [RBP + 0x4c8], EAX
0x18001359c: MOV EDI, dword ptr [R14 + 0x2a0]
0x1800135a3: MOV RBX, R15
0x1800135a6: MOV ESI, 0x2
0x1800135ab: NOP dword ptr [RAX + RAX*0x1]
0x1800135b0: MOV qword ptr [RBP + 0x8], R15
0x1800135b4: MOV dword ptr [RBP + -0x8], 0x2010000
0x1800135bb: LEA RAX, [RBP + 0x400]
0x1800135c2: ADD RAX, RBX
0x1800135c5: MOV qword ptr [RBP], RAX
0x1800135c9: MOV qword ptr [RBP + 0x20], R15
0x1800135cd: MOV dword ptr [RBP + 0x10], 0x1010000
0x1800135d4: LEA RAX, [RBP + 0x5a0]
0x1800135db: ADD RAX, RBX
0x1800135de: MOV qword ptr [RBP + 0x18], RAX
0x1800135e2: MOV dword ptr [RSP + 0x40], EDI
0x1800135e6: MOV dword ptr [RSP + 0x44], EDI
0x1800135ea: MOV dword ptr [RSP + 0x28], 0x1
0x1800135f2: MOVSD qword ptr [RSP + 0x20], XMM6
0x1800135f8: MOVAPS XMM3, XMM6
0x1800135fb: LEA R8, [RSP + 0x40]
0x180013600: LEA RDX, [RBP + -0x8]
0x180013604: LEA RCX, [RBP + 0x10]
0x180013608: CALL qword ptr [0x180078480]
0x18001360e: ADD RBX, 0x68
0x180013612: DEC RSI
0x180013615: JNZ 0x1800135b0
0x180013617: MOV RAX, qword ptr [RBP + 0x330]
0x18001361e: MOV qword ptr [RBP + -0x60], RAX
0x180013622: MOV EAX, dword ptr [RBP + 0x370]
0x180013628: MOV dword ptr [RBP + -0x58], EAX
0x18001362b: MOVAPS XMM0, xmmword ptr [RBP + -0x60]
0x18001362f: MOVDQA xmmword ptr [RBP + 0x1c0], XMM0
0x180013637: MOV RAX, qword ptr [RBP + 0x6e8]
0x18001363e: MOV qword ptr [RBP + -0x80], RAX
0x180013642: MOV EAX, dword ptr [RBP + 0x728]
0x180013648: MOV dword ptr [RBP + -0x78], EAX
0x18001364b: MOVAPS XMM0, xmmword ptr [RBP + -0x80]
0x18001364f: MOVDQA xmmword ptr [RBP + 0x170], XMM0
0x180013657: MOV RAX, qword ptr [RBP + 0x548]
0x18001365e: MOV qword ptr [RBP + -0x50], RAX
0x180013662: MOV EAX, dword ptr [RBP + 0x588]
0x180013668: MOV dword ptr [RBP + -0x48], EAX
0x18001366b: MOVAPS XMM0, xmmword ptr [RBP + -0x50]
0x18001366f: MOVDQA xmmword ptr [RBP + 0x1a0], XMM0
0x180013677: MOV RAX, qword ptr [RBP + 0x4e0]
0x18001367e: MOV qword ptr [RBP + -0x30], RAX
0x180013682: MOV EAX, dword ptr [RBP + 0x520]
0x180013688: MOV dword ptr [RBP + -0x28], EAX
0x18001368b: MOVAPS XMM0, xmmword ptr [RBP + -0x30]
0x18001368f: MOVDQA xmmword ptr [RBP + 0x190], XMM0
0x180013697: LEA RAX, [RBP + 0x1c0]
0x18001369e: MOV qword ptr [RSP + 0x28], RAX
0x1800136a3: LEA RAX, [RBP + 0x170]
0x1800136aa: MOV qword ptr [RSP + 0x20], RAX
0x1800136af: LEA R9, [RBP + 0x1a0]
0x1800136b6: LEA R8, [RBP + 0x190]
0x1800136bd: MOV EDX, dword ptr [RSP + 0x48]
0x1800136c1: MOV ECX, dword ptr [RSP + 0x5c]
0x1800136c5: CALL 0x1800121b0
0x1800136ca: XORPS XMM1, XMM1
0x1800136cd: CVTSS2SD XMM1, XMM0
0x1800136d1: LEA RDX, [RBP + 0x538]
0x1800136d8: COMISD XMM15, XMM1
0x1800136dd: JA 0x180013731
0x1800136df: LEA RCX, [RBP + 0x4d0]
0x1800136e6: CALL 0x1800241f0
0x1800136eb: LEA RDX, [RBP + 0x6d8]
0x1800136f2: LEA RCX, [RBP + 0x670]
0x1800136f9: CALL 0x1800241f0
0x1800136fe: NOP
0x1800136ff: MOV R9, R12
0x180013702: LEA EDX, [RSI + 0x68]
0x180013705: LEA R8D, [RSI + 0x2]
0x180013709: LEA RCX, [RBP + 0x400]
0x180013710: CALL 0x180072720
0x180013715: NOP
0x180013716: MOV R9, R12
0x180013719: LEA EDX, [RSI + 0x68]
0x18001371c: LEA R8D, [RSI + 0x2]
0x180013720: LEA RCX, [RBP + 0x5a0]
0x180013727: CALL 0x180072720
0x18001372c: JMP 0x180013130
0x180013731: LEA RCX, [RBP + 0x2b0]
0x180013738: CALL 0x180001780
0x18001373d: MOV EAX, dword ptr [RBP + 0x598]
0x180013743: MOV dword ptr [RBP + 0x310], EAX
0x180013749: LEA RDX, [RBP + 0x6d8]
0x180013750: LEA RCX, [RBP + 0x390]
0x180013757: CALL 0x180001780
0x18001375c: MOV EAX, dword ptr [RBP + 0x738]
0x180013762: MOV dword ptr [RBP + 0x3f0], EAX
0x180013768: MOV R9, R12
0x18001376b: MOV EDX, 0x68
0x180013770: LEA R8D, [RDX + -0x66]
0x180013774: LEA RCX, [RBP + 0x400]
0x18001377b: CALL 0x180072720
0x180013780: NOP
0x180013781: MOV R9, R12
0x180013784: MOV EDX, 0x68
0x180013789: LEA R8D, [RDX + -0x66]
0x18001378d: LEA RCX, [RBP + 0x5a0]
0x180013794: CALL 0x180072720
0x180013799: JMP 0x1800137e3
0x18001379b: LEA RDX, [RBP + 0x4d0]
0x1800137a2: LEA RCX, [RBP + 0x2b0]
0x1800137a9: CALL 0x180001780
0x1800137ae: MOV EAX, dword ptr [RBP + 0x530]
0x1800137b4: MOV dword ptr [RBP + 0x310], EAX
0x1800137ba: LEA RDX, [RBP + 0x670]
0x1800137c1: LEA RCX, [RBP + 0x390]
0x1800137c8: CALL 0x180001780
0x1800137cd: MOV EAX, dword ptr [RBP + 0x6d0]
0x1800137d3: MOV dword ptr [RBP + 0x3f0], EAX
0x1800137d9: XOR R15D, R15D
0x1800137dc: LEA R12, [0x180001750]
0x1800137e3: MOV RDI, qword ptr [RBP + -0x10]
0x1800137e7: MOV RAX, qword ptr [RDI + 0x10]
0x1800137eb: MOV qword ptr [RSP + 0x70], RAX
0x1800137f0: MOV RAX, qword ptr [RDI + 0x50]
0x1800137f4: MOV dword ptr [RSP + 0x78], EAX
0x1800137f8: MOVAPS XMM0, xmmword ptr [RSP + 0x70]
0x1800137fd: MOVDQA xmmword ptr [RBP + 0x1b0], XMM0
0x180013805: MOV RCX, qword ptr [RBP + 0x850]
0x18001380c: MOV RAX, qword ptr [RCX + 0x10]
0x180013810: MOV qword ptr [RBP + -0x40], RAX
0x180013814: MOV RAX, qword ptr [RCX + 0x50]
0x180013818: MOV dword ptr [RBP + -0x38], EAX
0x18001381b: MOVAPS XMM0, xmmword ptr [RBP + -0x40]
0x18001381f: MOVDQA xmmword ptr [RBP + 0x1d0], XMM0
0x180013827: MOV RAX, qword ptr [RBP + 0x2c0]
0x18001382e: MOV qword ptr [RBP + -0x70], RAX
0x180013832: MOV EAX, dword ptr [RBP + 0x300]
0x180013838: MOV dword ptr [RBP + -0x68], EAX
0x18001383b: MOVAPS XMM0, xmmword ptr [RBP + -0x70]
0x18001383f: MOVDQA xmmword ptr [RBP + 0x140], XMM0
0x180013847: MOVUPS XMM0, xmmword ptr [R14 + 0x690]
0x18001384f: MOVAPS xmmword ptr [RBP + 0x200], XMM0
0x180013856: MOVUPS XMM1, xmmword ptr [R14 + 0x6a0]
0x18001385e: MOVAPS xmmword ptr [RBP + 0x210], XMM1
0x180013865: XORPS XMM3, XMM3
0x180013868: CVTSD2SS XMM3, XMM14
0x18001386d: LEA RAX, [RBP + 0x1b0]
0x180013874: MOV qword ptr [RSP + 0x30], RAX
0x180013879: LEA RAX, [RBP + 0x1d0]
0x180013880: MOV qword ptr [RSP + 0x28], RAX
0x180013885: LEA RAX, [RBP + 0x140]
0x18001388c: MOV qword ptr [RSP + 0x20], RAX
0x180013891: LEA R8, [RBP + 0x200]
0x180013898: MOV EDX, dword ptr [RSP + 0x50]
0x18001389c: MOV ECX, dword ptr [RSP + 0x58]
0x1800138a0: CALL 0x1800124d0
0x1800138a5: OR EBX, 0xffffffff
0x1800138a8: CMP byte ptr [R14 + 0x11c], 0x1
0x1800138b0: JNZ 0x180013a30
0x1800138b6: MOV EAX, dword ptr [RBP + 0x858]
0x1800138bc: TEST EAX, EAX
0x1800138be: JNZ 0x18001397c
0x1800138c4: MOV RDX, RDI
0x1800138c7: LEA RCX, [RBP + 0xc0]
0x1800138ce: CALL 0x1800013f0
0x1800138d3: MOV EAX, dword ptr [RDI + 0x60]
0x1800138d6: MOV dword ptr [RBP + 0x120], EAX
0x1800138dc: LEA R9D, [RBX + 0x6]
0x1800138e0: LEA R8, [RBP + 0xc0]
0x1800138e7: MOV RDX, qword ptr [RSP + 0x68]
0x1800138ec: MOV RCX, R14
0x1800138ef: CALL 0x180004b70
0x1800138f4: NOP
0x1800138f5: MOV RCX, qword ptr [RBP + 0xf8]
0x1800138fc: TEST RCX, RCX
0x1800138ff: JZ 0x18001391a
0x180013901: MOV EAX, EBX
0x180013903: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180013908: CMP EAX, 0x1
0x18001390b: JNZ 0x18001391a
0x18001390d: LEA RCX, [RBP + 0xc0]
0x180013914: CALL qword ptr [0x180078400]
0x18001391a: MOV qword ptr [RBP + 0xf8], R15
0x180013921: MOV qword ptr [RBP + 0xd0], R15
0x180013928: MOV qword ptr [RBP + 0xe8], R15
0x18001392f: MOV qword ptr [RBP + 0xe0], R15
0x180013936: MOV qword ptr [RBP + 0xd8], R15
0x18001393d: MOV ECX, R15D
0x180013940: CMP dword ptr [RBP + 0xc4], 0x0
0x180013947: JLE 0x180013969
0x180013949: MOV RDX, R15
0x18001394c: NOP dword ptr [RAX]
0x180013950: MOV RAX, qword ptr [RBP + 0x100]
0x180013957: MOV dword ptr [RDX + RAX*0x1], R15D
0x18001395b: INC ECX
0x18001395d: LEA RDX, [RDX + 0x4]
0x180013961: CMP ECX, dword ptr [RBP + 0xc4]
0x180013967: JL 0x180013950
0x180013969: LEA RAX, [RBP + 0x110]
0x180013970: MOV RCX, qword ptr [RBP + 0x108]
0x180013977: JMP 0x180013a24
0x18001397c: CMP EAX, 0x1
0x18001397f: JNZ 0x180013a30
0x180013985: MOV RDX, RDI
0x180013988: LEA RCX, [RBP + 0x50]
0x18001398c: CALL 0x1800013f0
0x180013991: MOV EAX, dword ptr [RDI + 0x60]
0x180013994: MOV dword ptr [RBP + 0xb0], EAX
0x18001399a: MOV R9D, 0xa
0x1800139a0: LEA R8, [RBP + 0x50]
0x1800139a4: MOV RDX, qword ptr [RSP + 0x68]
0x1800139a9: MOV RCX, R14
0x1800139ac: CALL 0x180004b70
0x1800139b1: NOP
0x1800139b2: MOV RCX, qword ptr [RBP + 0x88]
0x1800139b9: TEST RCX, RCX
0x1800139bc: JZ 0x1800139d4
0x1800139be: MOV EAX, EBX
0x1800139c0: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x1800139c5: CMP EAX, 0x1
0x1800139c8: JNZ 0x1800139d4
0x1800139ca: LEA RCX, [RBP + 0x50]
0x1800139ce: CALL qword ptr [0x180078400]
0x1800139d4: MOV qword ptr [RBP + 0x88], R15
0x1800139db: MOV qword ptr [RBP + 0x60], R15
0x1800139df: MOV qword ptr [RBP + 0x78], R15
0x1800139e3: MOV qword ptr [RBP + 0x70], R15
0x1800139e7: MOV qword ptr [RBP + 0x68], R15
0x1800139eb: MOV ECX, R15D
0x1800139ee: CMP dword ptr [RBP + 0x54], 0x0
0x1800139f2: JLE 0x180013a16
0x1800139f4: MOV RDX, R15
0x1800139f7: NOP word ptr [RAX + RAX*0x1]
0x180013a00: MOV RAX, qword ptr [RBP + 0x90]
0x180013a07: MOV dword ptr [RDX + RAX*0x1], R15D
0x180013a0b: INC ECX
0x180013a0d: LEA RDX, [RDX + 0x4]
0x180013a11: CMP ECX, dword ptr [RBP + 0x54]
0x180013a14: JL 0x180013a00
0x180013a16: LEA RAX, [RBP + 0xa0]
0x180013a1d: MOV RCX, qword ptr [RBP + 0x98]
0x180013a24: CMP RCX, RAX
0x180013a27: JZ 0x180013a30
0x180013a29: CALL qword ptr [0x1800783e0]
0x180013a2f: NOP
0x180013a30: MOV RCX, qword ptr [RBP + 0x3c8]
0x180013a37: TEST RCX, RCX
0x180013a3a: JZ 0x180013a55
0x180013a3c: MOV EAX, EBX
0x180013a3e: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180013a43: CMP EAX, 0x1
0x180013a46: JNZ 0x180013a55
0x180013a48: LEA RCX, [RBP + 0x390]
0x180013a4f: CALL qword ptr [0x180078400]
0x180013a55: MOV qword ptr [RBP + 0x3c8], R15
0x180013a5c: MOV qword ptr [RBP + 0x3a0], R15
0x180013a63: MOV qword ptr [RBP + 0x3b8], R15
0x180013a6a: MOV qword ptr [RBP + 0x3b0], R15
0x180013a71: MOV qword ptr [RBP + 0x3a8], R15
0x180013a78: MOV ECX, R15D
0x180013a7b: CMP dword ptr [RBP + 0x394], 0x0
0x180013a82: JLE 0x180013aa9
0x180013a84: MOV RDX, R15
0x180013a87: NOP word ptr [RAX + RAX*0x1]
0x180013a90: MOV RAX, qword ptr [RBP + 0x3d0]
0x180013a97: MOV dword ptr [RDX + RAX*0x1], R15D
0x180013a9b: INC ECX
0x180013a9d: LEA RDX, [RDX + 0x4]
0x180013aa1: CMP ECX, dword ptr [RBP + 0x394]
0x180013aa7: JL 0x180013a90
0x180013aa9: LEA RAX, [RBP + 0x3e0]
0x180013ab0: MOV RCX, qword ptr [RBP + 0x3d8]
0x180013ab7: CMP RCX, RAX
0x180013aba: JZ 0x180013ac3
0x180013abc: CALL qword ptr [0x1800783e0]
0x180013ac2: NOP
0x180013ac3: MOV RCX, qword ptr [RBP + 0x2e8]
0x180013aca: TEST RCX, RCX
0x180013acd: JZ 0x180013ae8
0x180013acf: MOV EAX, EBX
0x180013ad1: XADD.LOCK dword ptr [RCX + 0x14], EAX
0x180013ad6: CMP EAX, 0x1
0x180013ad9: JNZ 0x180013ae8
0x180013adb: LEA RCX, [RBP + 0x2b0]
0x180013ae2: CALL qword ptr [0x180078400]
0x180013ae8: MOV qword ptr [RBP + 0x2e8], R15
0x180013aef: MOV qword ptr [RBP + 0x2c0], R15
0x180013af6: MOV qword ptr [RBP + 0x2d8], R15
0x180013afd: MOV qword ptr [RBP + 0x2d0], R15
0x180013b04: MOV qword ptr [RBP + 0x2c8], R15
0x180013b0b: MOV ECX, R15D
0x180013b0e: CMP dword ptr [RBP + 0x2b4], 0x0
0x180013b15: JLE 0x180013b39
0x180013b17: MOV RDX, R15
0x180013b1a: NOP word ptr [RAX + RAX*0x1]
0x180013b20: MOV RAX, qword ptr [RBP + 0x2f0]
0x180013b27: MOV dword ptr [RDX + RAX*0x1], R15D
0x180013b2b: INC ECX
0x180013b2d: LEA RDX, [RDX + 0x4]
0x180013b31: CMP ECX, dword ptr [RBP + 0x2b4]
0x180013b37: JL 0x180013b20
0x180013b39: LEA RAX, [RBP + 0x300]
0x180013b40: MOV RCX, qword ptr [RBP + 0x2f8]
0x180013b47: CMP RCX, RAX
0x180013b4a: JZ 0x180013b53
0x180013b4c: CALL qword ptr [0x1800783e0]
0x180013b52: NOP
0x180013b53: MOV RCX, qword ptr [RBP + 0x280]
0x180013b5a: MOV RAX, qword ptr [RBP + 0x290]
0x180013b61: SUB qword ptr [RCX + 0x8], RAX
0x180013b65: MOV RAX, qword ptr [RBP + 0x358]
0x180013b6c: TEST RAX, RAX
0x180013b6f: JZ 0x180013b88
0x180013b71: XADD.LOCK dword ptr [RAX + 0x14], EBX
0x180013b76: CMP EBX, 0x1
0x180013b79: JNZ 0x180013b88
0x180013b7b: LEA RCX, [RBP + 0x320]
0x180013b82: CALL qword ptr [0x180078400]
0x180013b88: MOV qword ptr [RBP + 0x358], R15
0x180013b8f: MOV qword ptr [RBP + 0x330], R15
0x180013b96: MOV qword ptr [RBP + 0x348], R15
0x180013b9d: MOV qword ptr [RBP + 0x340], R15
0x180013ba4: MOV qword ptr [RBP + 0x338], R15
0x180013bab: MOV ECX, R15D
0x180013bae: CMP dword ptr [RBP + 0x324], 0x0
0x180013bb5: JLE 0x180013bd9
0x180013bb7: MOV RDX, R15
0x180013bba: NOP word ptr [RAX + RAX*0x1]
0x180013bc0: MOV RAX, qword ptr [RBP + 0x360]
0x180013bc7: MOV dword ptr [RDX + RAX*0x1], R15D
0x180013bcb: INC ECX
0x180013bcd: LEA RDX, [RDX + 0x4]
0x180013bd1: CMP ECX, dword ptr [RBP + 0x324]
0x180013bd7: JL 0x180013bc0
0x180013bd9: LEA RAX, [RBP + 0x370]
0x180013be0: MOV RCX, qword ptr [RBP + 0x368]
0x180013be7: CMP RCX, RAX
0x180013bea: JZ 0x180013bf3
0x180013bec: CALL qword ptr [0x1800783e0]
0x180013bf2: NOP
0x180013bf3: MOV R9, R12
0x180013bf6: MOV EDX, 0x68
0x180013bfb: LEA R8D, [RDX + -0x66]
0x180013bff: LEA RCX, [RBP + 0x670]
0x180013c06: CALL 0x180072720
0x180013c0b: NOP
0x180013c0c: MOV R9, R12
0x180013c0f: MOV EDX, 0x68
0x180013c14: LEA R8D, [RDX + -0x66]
0x180013c18: LEA RCX, [RBP + 0x4d0]
0x180013c1f: CALL 0x180072720
0x180013c24: MOV RCX, qword ptr [RBP + 0x740]
0x180013c2b: XOR RCX, RSP
0x180013c2e: CALL 0x180072a30
0x180013c33: LEA R11, [RSP + 0x8f0]
0x180013c3b: MOV RBX, qword ptr [R11 + 0x48]
0x180013c3f: MOVAPS XMM6, xmmword ptr [R11 + -0x10]
0x180013c44: MOVAPS XMM7, xmmword ptr [R11 + -0x20]
0x180013c49: MOVAPS XMM8, xmmword ptr [R11 + -0x30]
0x180013c4e: MOVAPS XMM9, xmmword ptr [R11 + -0x40]
0x180013c53: MOVAPS XMM10, xmmword ptr [R11 + -0x50]
0x180013c58: MOVAPS XMM11, xmmword ptr [R11 + -0x60]
0x180013c5d: MOVAPS XMM12, xmmword ptr [R11 + -0x70]
0x180013c62: MOVAPS XMM13, xmmword ptr [R11 + -0x80]
0x180013c67: MOVAPS XMM14, xmmword ptr [R11 + -0x90]
0x180013c6f: MOVAPS XMM15, xmmword ptr [R11 + -0xa0]
0x180013c77: MOV RSP, R11
0x180013c7a: POP R15
0x180013c7c: POP R14
0x180013c7e: POP R13
0x180013c80: POP R12
0x180013c82: POP RDI
0x180013c83: POP RSI
0x180013c84: POP RBP
0x180013c85: RET
```

## procedure_pseudo_code {'procedure': '0x1800180c0'}
Evidence: `ev_f5cbad14e32ad11ada8dc34a9ebd1a0f843289159bb8472ee7a3ab128767851f`
```c

void FUN_1800180c0(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5,
                  longlong *param_6)

{
  longlong lVar1;
  float fVar2;
  float fVar3;
  float fVar4;
  float fVar5;
  longlong lVar6;
  longlong lVar7;
  float fVar8;
  float fVar9;
  float fVar10;
  float fVar11;
  float fVar12;
  
  if (0 < param_2) {
    lVar7 = 0;
    do {
      lVar6 = 0;
      if (0 < (longlong)param_1) {
        do {
          lVar1 = lVar6 * 4;
          fVar2 = *(float *)((int)param_4[1] * lVar7 + lVar1 + *param_4);
          fVar3 = *(float *)((int)param_4[3] * lVar7 + lVar1 + param_4[2]);
          fVar4 = *(float *)((int)param_4[5] * lVar7 + lVar1 + param_4[4]);
          fVar5 = *(float *)((int)param_3[1] * lVar7 + lVar1 + *param_3);
          fVar10 = *(float *)((int)param_3[3] * lVar7 + param_3[2] + lVar1) - fVar5 * fVar2;
          fVar12 = *(float *)((int)param_3[5] * lVar7 + param_3[4] + lVar1) - fVar5 * fVar3;
          fVar9 = *(float *)((int)param_3[7] * lVar7 + lVar1 + param_3[6]) - fVar5 * fVar4;
          fVar11 = fVar12 * *(float *)((int)param_5[3] * lVar7 + lVar1 + param_5[2]) +
                   fVar10 * *(float *)((int)param_5[1] * lVar7 + lVar1 + *param_5) +
                   fVar9 * *(float *)((int)param_5[5] * lVar7 + lVar1 + param_5[4]);
          fVar8 = fVar12 * *(float *)((int)param_5[7] * lVar7 + param_5[6] + lVar1) +
                  fVar10 * *(float *)((int)param_5[3] * lVar7 + lVar1 + param_5[2]) +
                  fVar9 * *(float *)((int)param_5[9] * lVar7 + param_5[8] + lVar1);
          lVar6 = lVar6 + 1;
          fVar9 = fVar12 * *(float *)((int)param_5[9] * lVar7 + lVar1 + param_5[8]) +
                  fVar10 * *(float *)((int)param_5[5] * lVar7 + param_5[4] + lVar1) +
                  fVar9 * *(float *)((int)param_5[0xb] * lVar7 + lVar1 + param_5[10]);
          *(float *)((int)param_6[3] * lVar7 + lVar1 + param_6[2]) = fVar11;
          *(float *)((int)param_6[5] * lVar7 + lVar1 + param_6[4]) = fVar8;
          *(float *)((int)param_6[7] * lVar7 + lVar1 + param_6[6]) = fVar9;
          *(float *)((int)param_6[1] * lVar7 + lVar1 + *param_6) =
               ((fVar5 - fVar11 * fVar2) - fVar8 * fVar3) - fVar9 * fVar4;
        } while (lVar6 < param_1);
      }
      lVar7 = lVar7 + 1;
    } while (lVar7 < param_2);
  }
  return;
}


```
