// ev_12f4996b91d57ebc6cd62cc9a154943858df28088d971ef66aaac1ebe6785368

void FUN_18000aed0(int *param_1,longlong param_2,longlong param_3,longlong param_4)

{
  ushort uVar1;
  int *piVar2;
  longlong lVar3;
  int iVar4;
  int *piVar5;
  ulonglong uVar6;
  int local_res8 [2];
  int local_18 [4];
  
  if (0 < *param_1) {
    _vcomp_for_static_simple_init(0,*param_1 + -1,1,1,local_res8,local_18);
    if (local_res8[0] <= local_18[0]) {
      uVar6 = (ulonglong)((local_18[0] - local_res8[0]) + 1);
      piVar5 = (int *)(param_4 + (longlong)local_res8[0] * 4);
      do {
        uVar1 = *(ushort *)((param_2 - param_4) + (longlong)piVar5);
        lVar3 = 1;
        piVar2 = (int *)(param_3 + 0x1130);
        do {
          if ((int)(uint)uVar1 < piVar2[-1]) break;
          if ((int)(uint)uVar1 < *piVar2) {
            lVar3 = lVar3 + 1;
            break;
          }
          lVar3 = lVar3 + 2;
          piVar2 = piVar2 + 2;
        } while (lVar3 < 0xf);
        iVar4 = (int)(((uint)uVar1 - *(int *)(param_3 + 0x1124 + lVar3 * 4)) *
                      *(int *)(param_3 + 0x1160 + lVar3 * 4) + 0x20) >> 6;
        if (0x1fff < iVar4) {
          iVar4 = 0x1fff;
        }
        iVar4 = iVar4 + *(int *)(param_3 + 0x119c + lVar3 * 4);
        if (0x1fff < iVar4) {
          iVar4 = 0x1fff;
        }
        *piVar5 = iVar4;
        uVar6 = uVar6 - 1;
        piVar5 = piVar5 + 1;
      } while (uVar6 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

