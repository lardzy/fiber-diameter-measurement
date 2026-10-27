// ev_c4c68c2461f4a773ff500404b73aea8c86f1abf8867e9103480047deef5e66a6

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_18000ba60(int *param_1,longlong param_2,int *param_3,int *param_4,longlong param_5,
                  int *param_6,int *param_7,uint *param_8,longlong param_9)

{
  int iVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  int *piVar7;
  longlong lVar8;
  int iVar9;
  int *piVar10;
  short *psVar11;
  longlong lVar12;
  ulonglong uVar13;
  short *psVar14;
  longlong lVar15;
  undefined1 auStack_c8 [32];
  int *local_a8;
  int *local_a0;
  int local_98;
  int local_94;
  int *local_90;
  longlong local_88;
  int *local_80;
  longlong local_78;
  int *local_70;
  int local_68;
  undefined8 local_64;
  undefined8 local_5c;
  undefined8 local_54;
  undefined4 local_4c;
  ulonglong local_48;
  
  local_48 = DAT_180019000 ^ (ulonglong)auStack_c8;
  local_78 = param_9;
  if (0 < *param_7) {
    local_a0 = &local_98;
    local_a8 = &local_94;
    local_90 = param_3;
    local_88 = param_2;
    local_80 = param_4;
    local_70 = param_1;
    _vcomp_for_static_simple_init(0,*param_7 + -1,1);
    for (; local_94 <= local_98; local_94 = local_94 + 1) {
      iVar1 = *param_6;
      iVar9 = 0;
      lVar8 = 0;
      if (0 < iVar1) {
        uVar2 = *param_8;
        do {
          piVar10 = (int *)(param_5 + 0x24);
          iVar3 = *param_3;
          local_64 = 0;
          local_5c = 0;
          local_54 = 0;
          local_4c = 0;
          local_68 = 0;
          lVar15 = 0x21;
          psVar14 = (short *)(local_88 + (iVar3 * local_94 + lVar8) * 2);
          do {
            lVar12 = 0x21;
            psVar11 = psVar14;
            do {
              iVar4 = *piVar10;
              piVar10 = piVar10 + 1;
              (&local_68)[*psVar11] = (&local_68)[*psVar11] + iVar4;
              psVar11 = psVar11 + 1;
              lVar12 = lVar12 + -1;
            } while (lVar12 != 0);
            psVar14 = psVar14 + iVar3;
            lVar15 = lVar15 + -1;
          } while (lVar15 != 0);
          if (0 < (int)uVar2) {
            piVar7 = &local_68;
            iVar3 = *local_70;
            iVar4 = *local_80;
            piVar10 = (int *)(local_78 + (longlong)(iVar9 + iVar1 * local_94) * 4);
            uVar13 = (ulonglong)uVar2;
            do {
              iVar5 = *(int *)(param_5 + 0x18);
              iVar6 = *piVar7;
              if (iVar5 < *piVar7) {
                *piVar7 = iVar5;
                iVar6 = iVar5;
              }
              piVar7 = piVar7 + 1;
              *piVar10 = iVar3 * iVar6 >> ((byte)*(undefined4 *)(param_5 + 0x20) & 0x1f);
              piVar10 = piVar10 + iVar4;
              uVar13 = uVar13 - 1;
            } while (uVar13 != 0);
          }
          iVar9 = iVar9 + 1;
          lVar8 = lVar8 + 1;
          param_3 = local_90;
        } while (iVar9 < *param_6);
      }
    }
    _vcomp_for_static_end();
  }
  return;
}

