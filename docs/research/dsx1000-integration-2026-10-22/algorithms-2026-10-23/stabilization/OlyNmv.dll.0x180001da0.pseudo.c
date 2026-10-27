// ev_c5aafc918a44b2aef144b5f654d4afc4ecabc26808df297c7b29f2e8474e6dc3
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

void FUN_180001da0(int *param_1)

{
  byte *pbVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  longlong *plVar7;
  int *piVar8;
  longlong lVar9;
  uint uVar10;
  ulonglong uVar11;
  int iVar12;
  longlong lVar13;
  int iVar14;
  longlong lVar15;
  ulonglong uVar16;
  longlong lVar17;
  longlong lVar18;
  int iVar19;
  longlong lVar20;
  int iVar21;
  longlong lVar22;
  longlong lVar23;
  uint uVar24;
  ulonglong uVar25;
  int local_res8;
  
  plVar7 = *(longlong **)(param_1 + 2);
  piVar8 = (int *)plVar7[1];
  uVar2 = *(uint *)(plVar7 + 2);
  lVar17 = *plVar7;
  iVar19 = piVar8[1];
  iVar3 = *(int *)(lVar17 + 8);
  lVar9 = *(longlong *)(lVar17 + 0x18);
  iVar4 = *piVar8;
  iVar5 = *(int *)(lVar17 + 0x10);
  iVar6 = piVar8[4];
  iVar21 = iVar3;
  if (3 < iVar3) {
    iVar21 = 3;
  }
  uVar10 = ((param_1[1] - 1U) + iVar19) / (uint)param_1[1];
  iVar14 = *param_1 * uVar10;
  iVar12 = iVar14 + uVar10;
  lVar17 = (longlong)iVar21;
  if (iVar19 < iVar12) {
    iVar12 = iVar19;
  }
  if (iVar14 < iVar12) {
    lVar22 = (longlong)(int)(iVar14 * iVar5 * uVar2);
    lVar20 = (longlong)(iVar14 * iVar6) + *(longlong *)(piVar8 + 6);
    uVar11 = (ulonglong)(uint)(iVar12 - iVar14);
    do {
      lVar23 = 0;
      if (0 < (longlong)iVar4) {
        local_res8 = 0;
        do {
          uVar10 = 0;
          uVar24 = 0;
          if ((int)uVar2 < 1) {
LAB_180001f89:
            *(undefined4 *)(lVar20 + lVar23 * 4) = 0;
          }
          else {
            lVar18 = local_res8 + lVar22 + lVar9;
            uVar25 = (ulonglong)uVar2;
            do {
              uVar24 = uVar24 + uVar2;
              lVar15 = lVar18;
              uVar16 = (ulonglong)uVar2;
              do {
                iVar19 = 0;
                iVar21 = 0;
                lVar13 = 0;
                if (1 < lVar17) {
                  do {
                    pbVar1 = (byte *)(lVar15 + lVar13);
                    lVar13 = lVar13 + 2;
                    iVar19 = iVar19 + (uint)*pbVar1;
                    iVar21 = iVar21 + (uint)*(byte *)(lVar15 + -1 + lVar13);
                  } while (lVar13 < lVar17 + -1);
                }
                if (lVar13 < lVar17) {
                  uVar10 = uVar10 + *(byte *)(lVar15 + lVar13);
                }
                lVar15 = lVar15 + iVar3;
                uVar10 = uVar10 + iVar21 + iVar19;
                uVar16 = uVar16 - 1;
              } while (uVar16 != 0);
              lVar18 = lVar18 + iVar5;
              uVar25 = uVar25 - 1;
            } while (uVar25 != 0);
            if (uVar24 == 0) goto LAB_180001f89;
            *(float *)(lVar20 + lVar23 * 4) = (float)uVar10 / (float)uVar24;
          }
          local_res8 = local_res8 + iVar3 * uVar2;
          lVar23 = lVar23 + 1;
        } while (lVar23 < iVar4);
      }
      lVar20 = lVar20 + iVar6;
      lVar22 = lVar22 + (int)(iVar5 * uVar2);
      uVar11 = uVar11 - 1;
    } while (uVar11 != 0);
  }
  return;
}

