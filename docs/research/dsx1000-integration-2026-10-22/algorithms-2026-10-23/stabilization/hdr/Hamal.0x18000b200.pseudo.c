// ev_c5bca8ad7947475a368a3e1856953575658d8b442d056d46a97b586f7a6a7703

void FUN_18000b200(int *param_1,int *param_2,int *param_3,longlong param_4,longlong param_5,
                  longlong param_6)

{
  int iVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  ulonglong uVar5;
  int iVar6;
  int iVar7;
  int iVar8;
  short *psVar9;
  longlong lVar10;
  int local_res18 [4];
  int local_38 [4];
  
  if (0 < *param_3) {
    _vcomp_for_static_simple_init(0,*param_3 + -1,3,1,local_res18,local_38);
    if (local_res18[0] <= local_38[0]) {
      iVar1 = *param_2;
      iVar2 = *param_1;
      psVar9 = (short *)(param_4 + ((longlong)local_res18[0] + 2) * 2);
      lVar10 = param_6 - param_4;
      uVar5 = (ulonglong)((uint)(local_38[0] - local_res18[0]) / 3 + 1);
      do {
        iVar6 = (int)*psVar9;
        iVar4 = (int)psVar9[-2];
        iVar7 = (int)psVar9[-1];
        iVar3 = *(int *)(param_5 + 0x1298) * iVar4 + iVar6 * *(int *)(param_5 + 0x1290) +
                iVar7 * *(int *)(param_5 + 0x1294);
        if (iVar3 < 0) {
          iVar3 = -((-iVar3 >> 9) + 1 >> 1);
        }
        else {
          iVar3 = (iVar3 >> 9) + 1 >> 1;
        }
        iVar8 = *(int *)(param_5 + 0x129c) * iVar6 + *(int *)(param_5 + 0x12a0) * iVar7 +
                iVar4 * *(int *)(param_5 + 0x12a4);
        if (iVar8 < 0) {
          iVar8 = -((-iVar8 >> 9) + 1 >> 1);
        }
        else {
          iVar8 = (iVar8 >> 9) + 1 >> 1;
        }
        iVar4 = *(int *)(param_5 + 0x12b0) * iVar4 + *(int *)(param_5 + 0x12a8) * iVar6 +
                *(int *)(param_5 + 0x12ac) * iVar7;
        if (iVar4 < 0) {
          iVar4 = -((-iVar4 >> 9) + 1 >> 1);
        }
        else {
          iVar4 = (iVar4 >> 9) + 1 >> 1;
        }
        iVar8 = iVar2 + iVar8;
        iVar4 = iVar2 + iVar4;
        iVar6 = iVar1;
        if ((iVar3 <= iVar1) && (iVar6 = iVar3, iVar3 < 0)) {
          iVar6 = 0;
        }
        iVar3 = iVar1;
        if ((iVar8 <= iVar1) && (iVar3 = iVar8, iVar8 < 0)) {
          iVar3 = 0;
        }
        iVar7 = iVar1;
        if ((iVar4 <= iVar1) && (iVar7 = iVar4, iVar4 < 0)) {
          iVar7 = 0;
        }
        *(short *)(lVar10 + -4 + (longlong)psVar9) = (short)iVar6;
        *(short *)(lVar10 + -2 + (longlong)psVar9) = (short)iVar3;
        *(short *)(lVar10 + (longlong)psVar9) = (short)iVar7;
        psVar9 = psVar9 + 3;
        uVar5 = uVar5 - 1;
      } while (uVar5 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

