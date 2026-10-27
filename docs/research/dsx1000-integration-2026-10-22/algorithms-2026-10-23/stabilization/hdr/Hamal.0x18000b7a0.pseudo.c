// ev_852fa445074e812e4fc3970321716931937107d27b692bb36fa8598996c3ae76

void FUN_18000b7a0(int *param_1,int *param_2,int *param_3,int *param_4,longlong param_5,
                  longlong param_6)

{
  int iVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  int *piVar6;
  ulonglong uVar7;
  short *psVar8;
  int iVar9;
  int local_res8 [2];
  int local_38 [4];
  
  if (0 < *param_1) {
    _vcomp_for_static_simple_init(0,*param_1 + -1,1,1,local_res8,local_38);
    if (local_res8[0] <= local_38[0]) {
      iVar1 = *param_2;
      iVar2 = *param_4;
      iVar3 = *param_3;
      piVar6 = (int *)(param_6 + (longlong)local_res8[0] * 4);
      uVar7 = (ulonglong)((local_38[0] - local_res8[0]) + 1);
      psVar8 = (short *)(param_5 + (longlong)local_res8[0] * 6 + 4);
      do {
        iVar9 = (int)psVar8[-2] * *piVar6 + 0x80 >> 8;
        if (0xff < iVar9) {
          iVar9 = 0xff;
        }
        iVar4 = (iVar3 * *piVar6 + 0x20 >> 6) + iVar1 * 4;
        if (iVar4 < 0x8000) {
          if (iVar4 < 0) {
            iVar4 = 0;
          }
        }
        else {
          iVar4 = 0x7fff;
        }
        iVar5 = iVar4 * (psVar8[-1] - iVar2) + 0x80 >> 8;
        if (iVar5 < 0x80) {
          if (iVar5 < -0x80) {
            iVar5 = -0x80;
          }
        }
        else {
          iVar5 = 0x7f;
        }
        iVar4 = iVar4 * (*psVar8 - iVar2) + 0x80 >> 8;
        if (iVar4 < 0x80) {
          if (iVar4 < -0x80) {
            iVar4 = -0x80;
          }
        }
        else {
          iVar4 = 0x7f;
        }
        psVar8[-2] = (short)iVar9;
        psVar8[-1] = (short)iVar5 + (short)iVar2;
        piVar6 = piVar6 + 1;
        *psVar8 = (short)iVar4 + (short)iVar2;
        uVar7 = uVar7 - 1;
        psVar8 = psVar8 + 3;
      } while (uVar7 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

