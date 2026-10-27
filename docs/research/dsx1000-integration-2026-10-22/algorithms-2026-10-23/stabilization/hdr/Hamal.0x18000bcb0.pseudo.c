// ev_f0af6669c996429503026abf6b3c6c7744da1f53f0b5ca64f47b057dd6fa9f5e

void FUN_18000bcb0(undefined4 *param_1,longlong param_2,int *param_3,int *param_4,longlong param_5,
                  int *param_6,int *param_7)

{
  int iVar1;
  int iVar2;
  undefined4 uVar3;
  int iVar4;
  int *piVar5;
  longlong lVar6;
  undefined2 *puVar7;
  longlong lVar8;
  uint uVar9;
  ulonglong uVar10;
  int local_38 [4];
  
  if (0 < *param_7) {
    _vcomp_for_static_simple_init(0,*param_7 + -1,1,1,&param_7,local_38);
    uVar10 = (ulonglong)(int)param_7;
    if ((int)param_7 <= local_38[0]) {
      iVar1 = *param_6;
      iVar2 = *param_3;
      lVar6 = (longlong)iVar2 + uVar10;
      do {
        if (0 < iVar1) {
          uVar3 = *param_1;
          piVar5 = (int *)(param_5 + (longlong)(iVar1 * (int)uVar10) * 4);
          puVar7 = (undefined2 *)(param_2 + (lVar6 * *param_4 + (longlong)iVar2) * 2);
          lVar8 = (longlong)iVar1;
          do {
            iVar4 = *piVar5;
            piVar5 = piVar5 + 1;
            *puVar7 = (short)(iVar4 >> ((byte)uVar3 & 0x1f));
            lVar8 = lVar8 + -1;
            puVar7 = puVar7 + 1;
          } while (lVar8 != 0);
        }
        uVar9 = (int)uVar10 + 1;
        uVar10 = (ulonglong)uVar9;
        lVar6 = lVar6 + 1;
      } while ((int)uVar9 <= local_38[0]);
    }
    _vcomp_for_static_end();
  }
  return;
}

