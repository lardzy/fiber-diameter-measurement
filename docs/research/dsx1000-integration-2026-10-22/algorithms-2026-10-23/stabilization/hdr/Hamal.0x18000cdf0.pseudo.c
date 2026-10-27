// ev_2d5ef5526b7e564ae2668168171bed241478b5ae23285e6252856e553f43f9af

void FUN_18000cdf0(int *param_1,longlong param_2,longlong param_3)

{
  int iVar1;
  short *psVar2;
  int iVar3;
  int iVar4;
  int *piVar5;
  ulonglong uVar6;
  int local_res8 [2];
  int local_res20 [2];
  
  if (0 < *param_1) {
    _vcomp_for_static_simple_init(0,*param_1 + -1,1,1,local_res8,local_res20);
    if (local_res8[0] <= local_res20[0]) {
      piVar5 = (int *)(param_3 + (longlong)local_res8[0] * 4);
      uVar6 = (ulonglong)((local_res20[0] - local_res8[0]) + 1);
      psVar2 = (short *)(param_2 + ((longlong)local_res8[0] * 3 + 1) * 2);
      do {
        iVar3 = (int)psVar2[-1];
        iVar4 = (int)*psVar2;
        iVar1 = (int)psVar2[1];
        if (iVar4 < iVar3) {
          if (iVar1 < iVar3) {
            iVar1 = iVar3;
          }
        }
        else if (iVar1 < iVar4) {
          iVar1 = iVar4;
        }
        *piVar5 = iVar1;
        piVar5 = piVar5 + 1;
        psVar2 = psVar2 + 3;
        uVar6 = uVar6 - 1;
      } while (uVar6 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

