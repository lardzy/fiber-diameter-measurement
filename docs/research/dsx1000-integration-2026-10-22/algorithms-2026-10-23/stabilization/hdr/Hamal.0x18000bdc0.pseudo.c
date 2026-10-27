// ev_c5ff996316077e91a6a75d0c80eefa07c718c6bbe99a6b25b3f25395cf9c8b41

void FUN_18000bdc0(longlong param_1,longlong param_2,int *param_3,int *param_4,int *param_5,
                  int *param_6)

{
  int *piVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  longlong lVar5;
  longlong lVar6;
  uint uVar7;
  ulonglong uVar8;
  longlong lVar9;
  int local_28 [4];
  
  piVar1 = param_6;
  if (0 < *param_6) {
    _vcomp_for_static_simple_init(0,*param_6 + -1,1,1,&param_6,local_28);
    uVar8 = (ulonglong)(int)param_6;
    if ((int)param_6 <= local_28[0]) {
      iVar4 = *param_5;
      lVar9 = (longlong)*param_4 + uVar8;
      do {
        iVar2 = 0;
        if (0 < iVar4) {
          lVar5 = (longlong)(*param_3 + (int)uVar8);
          do {
            iVar4 = *piVar1;
            iVar3 = iVar4 * iVar2;
            iVar2 = iVar2 + 1;
            lVar6 = (longlong)iVar3;
            *(undefined2 *)(param_1 + ((lVar9 - iVar4) - lVar6) * 2) =
                 *(undefined2 *)(param_2 + (lVar9 + lVar6) * 2);
            *(undefined2 *)(param_1 + (lVar6 + lVar5) * 2) =
                 *(undefined2 *)(param_2 + ((lVar5 - *piVar1) - lVar6) * 2);
            iVar4 = *param_5;
          } while (iVar2 < iVar4);
        }
        uVar7 = (int)uVar8 + 1;
        uVar8 = (ulonglong)uVar7;
        lVar9 = lVar9 + 1;
      } while ((int)uVar7 <= local_28[0]);
    }
    _vcomp_for_static_end();
  }
  return;
}

