// ev_51b80c38f42489d98855207e921350b9125ab4e4be72f3f78f30309dd1a0d475

void FUN_18000afc0(int *param_1,longlong param_2,longlong param_3)

{
  int iVar1;
  longlong lVar2;
  int *piVar3;
  int iVar4;
  int *piVar5;
  ulonglong uVar6;
  int local_res8 [2];
  int local_res20 [2];
  
  if (0 < *param_1) {
    _vcomp_for_static_simple_init(0,*param_1 + -1,1,1,local_res8,local_res20);
    if (local_res8[0] <= local_res20[0]) {
      piVar5 = (int *)(param_2 + (longlong)local_res8[0] * 4);
      uVar6 = (ulonglong)((local_res20[0] - local_res8[0]) + 1);
      do {
        iVar1 = *piVar5;
        if (iVar1 < *(int *)(param_3 + 0x10)) {
          iVar1 = *(int *)(param_3 + 0x30) * iVar1 + 0x100 >> 9;
          if (iVar1 < 0x1fff) {
            if (iVar1 < 0) {
              iVar1 = 0;
            }
            *piVar5 = iVar1;
          }
          else {
            *piVar5 = 0x1ffe;
          }
        }
        else {
          lVar2 = 1;
          piVar3 = (int *)(param_3 + 0x14);
          iVar4 = 1;
          do {
            if (iVar1 < *piVar3) {
              if (iVar4 == 0) {
                iVar1 = iVar1 * *(int *)(param_3 + 0x30 + lVar2 * 4);
              }
              else {
                iVar1 = (iVar1 - *(int *)(param_3 + 0xc + lVar2 * 4)) *
                        *(int *)(param_3 + 0x30 + lVar2 * 4);
              }
              iVar1 = iVar1 + 0x100 >> 9;
              if (iVar1 < 0x400) {
                if (iVar1 < 0) {
                  iVar1 = 0;
                }
              }
              else {
                iVar1 = 0x3ff;
              }
              iVar1 = iVar4 * 0x400 + iVar1;
              break;
            }
            lVar2 = lVar2 + 1;
            iVar4 = iVar4 + 1;
            piVar3 = piVar3 + 1;
          } while (lVar2 < 7);
          if (iVar4 == 7) {
            iVar1 = (iVar1 - *(int *)(param_3 + 0x28)) * *(int *)(param_3 + 0x4c) + 0x100 >> 9;
            if (iVar1 < 0x400) {
              if (iVar1 < 0) {
                iVar1 = 0;
              }
            }
            else {
              iVar1 = 0x3ff;
            }
            iVar1 = iVar1 + 0x1c00;
          }
          *piVar5 = iVar1;
        }
        piVar5 = piVar5 + 1;
        uVar6 = uVar6 - 1;
      } while (uVar6 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

