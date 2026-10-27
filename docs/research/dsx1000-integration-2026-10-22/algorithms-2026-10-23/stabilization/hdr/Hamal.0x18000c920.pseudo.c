// ev_38c0d3b9eceef37ece6857aad4712e43cf694faefdc14efc03722cb631c1bbfe

void FUN_18000c920(longlong param_1,int *param_2,int *param_3)

{
  int iVar1;
  int *piVar2;
  int iVar3;
  uint uVar4;
  longlong lVar5;
  ulonglong uVar7;
  ulonglong uVar8;
  int local_res18 [2];
  int local_res20 [2];
  ulonglong uVar6;
  
  iVar1 = *param_3;
  uVar7 = (ulonglong)iVar1;
  if (0 < iVar1) {
    _vcomp_for_static_simple_init(0,iVar1 + -1,1,1,local_res18,local_res20);
    lVar5 = (longlong)local_res18[0];
    if (local_res18[0] <= local_res20[0]) {
      iVar3 = *param_2;
      uVar8 = (ulonglong)((local_res20[0] - local_res18[0]) + 1);
      do {
        if (iVar1 < iVar3) {
          piVar2 = (int *)(param_1 + (lVar5 + uVar7) * 4);
          uVar6 = uVar7;
          do {
            uVar4 = (int)uVar6 + iVar1;
            uVar6 = (ulonglong)uVar4;
            *piVar2 = *piVar2 + piVar2[-uVar7];
            iVar3 = *param_2;
            piVar2 = piVar2 + uVar7;
          } while ((int)uVar4 < iVar3);
        }
        lVar5 = lVar5 + 1;
        uVar8 = uVar8 - 1;
      } while (uVar8 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

