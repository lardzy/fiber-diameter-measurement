// ev_72d2b2d155adc2f67ce49493d33a83b4a101058f5d3b390da948ede931568198

void FUN_18000b940(longlong param_1,longlong param_2,int *param_3,int *param_4,int *param_5,
                  int *param_6,int *param_7)

{
  undefined2 uVar1;
  int iVar2;
  longlong lVar3;
  undefined2 *puVar4;
  int iVar5;
  longlong lVar6;
  int iVar7;
  undefined2 *puVar8;
  int iVar9;
  int iVar10;
  int local_28 [4];
  
  if (0 < *param_7) {
    _vcomp_for_static_simple_init(0,*param_7 + -1,1,1,&param_7,local_28);
    if ((int)param_7 <= local_28[0]) {
      iVar2 = *param_3;
      iVar5 = *param_4;
      iVar10 = (int)param_7;
      do {
        iVar9 = 0;
        iVar7 = *param_5 * iVar10 + iVar2;
        if (0 < iVar5) {
          lVar6 = (longlong)(*param_6 + iVar7);
          puVar4 = (undefined2 *)(param_2 + -2 + lVar6 * 2);
          lVar3 = (iVar7 - lVar6) * 2;
          puVar8 = (undefined2 *)(param_1 + lVar6 * 2);
          do {
            iVar9 = iVar9 + 1;
            *(undefined2 *)((lVar3 - param_2) + param_1 + (longlong)puVar4) =
                 *(undefined2 *)((lVar3 - param_1) + param_2 + (longlong)puVar8);
            uVar1 = *puVar4;
            puVar4 = puVar4 + -1;
            *puVar8 = uVar1;
            iVar5 = *param_4;
            puVar8 = puVar8 + 1;
          } while (iVar9 < iVar5);
        }
        iVar10 = iVar10 + 1;
      } while (iVar10 <= local_28[0]);
    }
    _vcomp_for_static_end();
  }
  return;
}

