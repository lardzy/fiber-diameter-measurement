// ev_2412843d3f07882f5711bacd7a02a261db8cb108e8af8bdda57aa02d0b8b89a8

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_18000d100(int *param_1,longlong param_2,longlong param_3,uint *param_4)

{
  uint uVar1;
  int iVar2;
  int iVar3;
  longlong lVar4;
  int *piVar5;
  int *piVar6;
  int *piVar7;
  int *piVar8;
  int iVar9;
  ulonglong uVar10;
  int iVar11;
  ulonglong uVar12;
  int aiStack_127c [1131];
  undefined8 uStack_d0;
  undefined1 auStack_c8 [32];
  undefined8 *local_a8;
  uint *local_a0;
  uint local_98 [2];
  undefined8 local_90;
  int local_88;
  int local_80 [16];
  ulonglong local_40;
  
  local_40 = DAT_180019000 ^ (ulonglong)auStack_c8;
  iVar11 = *param_1;
  if (0 < iVar11) {
    local_a0 = local_98;
    local_a8 = &local_90;
    uStack_d0 = 0x18000d161;
    local_88 = iVar11;
    _vcomp_for_static_simple_init(0,iVar11 + -1,1);
    if ((int)local_90 <= (int)local_98[0]) {
      uVar1 = *param_4;
      uVar10 = (ulonglong)((local_98[0] - (int)local_90) + 1);
      piVar7 = (int *)(param_2 + (longlong)(int)local_90 * 4);
      do {
        iVar9 = 0xfff;
        if (0 < (int)uVar1) {
          uVar12 = (ulonglong)uVar1;
          piVar6 = (int *)(param_3 + 0x11fc);
          piVar8 = piVar7;
          do {
            piVar5 = (int *)(param_3 + 0x1240);
            lVar4 = 1;
            iVar3 = piVar6[-8] * *piVar8 + 8 >> 4;
            if (0x3000 < iVar3) {
              iVar3 = 0x3000;
            }
            do {
              if (iVar3 < *piVar5) break;
              lVar4 = lVar4 + 1;
              piVar5 = piVar5 + 1;
            } while (lVar4 < 7);
            iVar3 = (iVar3 - *(int *)(param_3 + 0x1238 + lVar4 * 4)) *
                    *(int *)(param_3 + 0x1254 + lVar4 * 4) + 0x20 >> 6;
            if (0xfff < iVar3) {
              iVar3 = 0xfff;
            }
            iVar3 = iVar3 + *(int *)(param_3 + 0x1270 + lVar4 * 4);
            if (0xfff < iVar3) {
              iVar3 = 0xfff;
            }
            iVar3 = *piVar6 * iVar3 + 0x80 >> 8;
            if (0xfff < iVar3) {
              iVar3 = 0xfff;
            }
            piVar6 = piVar6 + 1;
            piVar8 = piVar8 + iVar11;
            *(int *)((longlong)aiStack_127c + (-4 - param_3) + (longlong)piVar6) = iVar3;
            iVar9 = iVar9 - iVar3;
            uVar12 = uVar12 - 1;
          } while (uVar12 != 0);
          if (iVar9 < 0) {
            iVar9 = 0;
          }
          iVar11 = local_88;
          if (0 < (int)uVar1) {
            uVar12 = (ulonglong)uVar1;
            piVar6 = local_80;
            piVar8 = piVar7;
            do {
              iVar3 = *(int *)((param_3 - (longlong)local_80) + 0x121c + (longlong)piVar6) * iVar9 +
                      0x200 >> 10;
              if (0xfff < iVar3) {
                iVar3 = 0xfff;
              }
              iVar2 = iVar3 + *piVar6;
              if (0xfff < iVar3 + *piVar6) {
                iVar2 = 0xfff;
              }
              *piVar8 = iVar2;
              piVar8 = piVar8 + local_88;
              *piVar6 = iVar2;
              uVar12 = uVar12 - 1;
              piVar6 = piVar6 + 1;
            } while (uVar12 != 0);
          }
        }
        piVar7 = piVar7 + 1;
        uVar10 = uVar10 - 1;
      } while (uVar10 != 0);
      local_90 = 0;
      local_98[0] = uVar1;
    }
    uStack_d0 = 0x18000d32e;
    _vcomp_for_static_end();
  }
  return;
}

