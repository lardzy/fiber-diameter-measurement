// ev_e6e88de875a64cd0fd6c8415fafc7d442bc58cc0f012bce500891eb2723637fa

void FUN_180001810(longlong param_1,longlong param_2,int *param_3,longlong param_4,byte param_5)

{
  longlong lVar1;
  longlong lVar2;
  int iVar3;
  longlong lVar4;
  int iVar5;
  int iVar6;
  int iVar7;
  int iVar8;
  int iVar9;
  int iVar10;
  int iVar11;
  int iVar12;
  
  iVar7 = 0;
  iVar10 = (1 << (param_5 & 0x1f)) + -1;
  iVar12 = 1 << (param_5 - 1 & 0x1f);
  if (0 < *(int *)(param_1 + 0x58)) {
    iVar5 = *param_3;
    do {
      iVar8 = 0;
      if (0 < iVar5) {
        do {
          lVar4 = (longlong)((iVar5 * iVar7 + iVar8) * 3);
          lVar1 = lVar4 * 2 + 4;
          lVar2 = lVar4 * 2 + 2;
          iVar6 = (int)*(short *)(lVar4 * 2 + param_2);
          iVar11 = *(short *)(lVar1 + param_2) - iVar12;
          iVar9 = *(short *)(lVar2 + param_2) - iVar12;
          iVar5 = param_3[0x4ad] * iVar6 + param_3[0x4af] * iVar11 + param_3[0x4ae] * iVar9;
          if (iVar5 < 0) {
            iVar5 = -((-iVar5 >> 9) + 1 >> 1);
          }
          else {
            iVar5 = (iVar5 >> 9) + 1 >> 1;
          }
          iVar3 = param_3[0x4b0] * iVar6 + param_3[0x4b2] * iVar11 + param_3[0x4b1] * iVar9;
          if (iVar3 < 0) {
            iVar3 = -((-iVar3 >> 9) + 1 >> 1);
          }
          else {
            iVar3 = (iVar3 >> 9) + 1 >> 1;
          }
          iVar6 = param_3[0x4b3] * iVar6 + iVar11 * param_3[0x4b5] + iVar9 * param_3[0x4b4];
          if (iVar6 < 0) {
            iVar6 = -((-iVar6 >> 9) + 1 >> 1);
          }
          else {
            iVar6 = (iVar6 >> 9) + 1 >> 1;
          }
          iVar9 = iVar10;
          if ((iVar5 <= iVar10) && (iVar9 = iVar5, iVar5 < 0)) {
            iVar9 = 0;
          }
          iVar5 = iVar10;
          if ((iVar3 <= iVar10) && (iVar5 = iVar3, iVar3 < 0)) {
            iVar5 = 0;
          }
          iVar11 = iVar10;
          if ((iVar6 <= iVar10) && (iVar11 = iVar6, iVar6 < 0)) {
            iVar11 = 0;
          }
          *(short *)(lVar1 + param_4) = (short)iVar9;
          *(short *)(lVar4 * 2 + param_4) = (short)iVar11;
          *(short *)(lVar2 + param_4) = (short)iVar5;
          iVar5 = *param_3;
          iVar8 = iVar8 + 1;
        } while (iVar8 < iVar5);
      }
      iVar7 = iVar7 + 1;
    } while (iVar7 < *(int *)(param_1 + 0x58));
  }
  return;
}

