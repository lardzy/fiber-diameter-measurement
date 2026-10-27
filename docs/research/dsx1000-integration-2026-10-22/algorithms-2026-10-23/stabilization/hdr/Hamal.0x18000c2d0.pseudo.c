// ev_4f47271bd667cde54bf525458da840b3a8420b87140a0f81282821b482c2ff47

void FUN_18000c2d0(longlong param_1,int *param_2,int *param_3,longlong param_4,longlong param_5)

{
  ushort uVar1;
  int iVar2;
  int *piVar3;
  longlong lVar4;
  int iVar5;
  ushort *puVar6;
  ulonglong uVar7;
  int local_res18 [2];
  int local_28 [4];
  
  if (0 < *param_3) {
    _vcomp_for_static_simple_init(0,*param_3 + -1,1,1,local_res18,local_28);
    if (local_res18[0] <= local_28[0]) {
      puVar6 = (ushort *)(param_4 + (longlong)local_res18[0] * 4);
      uVar7 = (ulonglong)((local_28[0] - local_res18[0]) + 1);
      do {
        uVar1 = *puVar6;
        lVar4 = 1;
        piVar3 = (int *)(param_5 + 0x1130);
        do {
          if ((int)(uint)uVar1 < piVar3[-1]) break;
          if ((int)(uint)uVar1 < *piVar3) {
            lVar4 = lVar4 + 1;
            break;
          }
          lVar4 = lVar4 + 2;
          piVar3 = piVar3 + 2;
        } while (lVar4 < 0xf);
        iVar2 = *param_2;
        iVar5 = (int)(((uint)uVar1 - *(int *)(param_5 + 0x1124 + lVar4 * 4)) *
                      *(int *)(param_5 + 0x1160 + lVar4 * 4) + 0x20) >> 6;
        if (iVar2 < iVar5) {
          iVar5 = iVar2;
        }
        iVar5 = iVar5 + *(int *)(param_5 + 0x119c + lVar4 * 4);
        if (iVar2 < iVar5) {
          iVar5 = iVar2;
        }
        _vcomp_atomic_add_i4(param_1 + ((longlong)iVar5 >> 10) * 4,1);
        puVar6 = puVar6 + 2;
        uVar7 = uVar7 - 1;
      } while (uVar7 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

