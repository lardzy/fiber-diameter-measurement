// ev_f3246d9a1e8670d236fe3555c409b317f75b6682eac86d0f714a034d381030f1

void FUN_18000c9e0(int *param_1,int *param_2,longlong param_3,int *param_4,int *param_5,int *param_6
                  ,longlong param_7,int *param_8,longlong param_9,longlong param_10)

{
  int iVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  int iVar7;
  longlong lVar8;
  int iVar9;
  int iVar10;
  int iVar11;
  int iVar12;
  longlong lVar13;
  int iVar14;
  int local_res8 [2];
  int local_res10;
  longlong local_res18;
  int *local_res20;
  int local_94;
  int local_90;
  int local_8c;
  int local_88;
  int local_84;
  int local_80;
  longlong local_78;
  longlong local_70;
  longlong local_68;
  longlong local_60;
  longlong local_58;
  
  iVar12 = *param_2;
  if (iVar12 < *param_1) {
    local_res18 = param_3;
    local_res20 = param_4;
    local_80 = iVar12;
    _vcomp_for_static_simple_init(iVar12,*param_1 + -1,1,1,local_res8,&local_94);
    if (local_res8[0] <= local_94) {
      iVar9 = *param_8;
      local_90 = *param_6 + -1;
      iVar11 = local_res8[0];
      do {
        local_res8[0] = 0;
        local_88 = *(int *)(param_10 + 0xc) * iVar11;
        iVar4 = local_88 >> 0xd;
        local_88 = local_88 + iVar4 * -0x2000;
        iVar1 = iVar4 + 1;
        iVar5 = local_90;
        if ((iVar4 <= local_90) && (iVar5 = iVar4, iVar4 < 0)) {
          iVar5 = 0;
        }
        iVar4 = local_90;
        if ((iVar1 <= local_90) && (iVar4 = iVar1, iVar1 < 0)) {
          iVar4 = 0;
        }
        local_res10 = 0;
        if (local_88 < 0) {
          local_88 = 0;
        }
        if (0 < iVar9) {
          local_84 = *param_5;
          local_58 = (longlong)iVar12;
          iVar1 = local_84 + -1;
          local_8c = local_84 * iVar5;
          local_84 = local_84 * iVar4;
          lVar13 = (longlong)(iVar9 * iVar11);
          do {
            iVar4 = local_res8[0] >> 0xd;
            iVar12 = iVar4 + 1;
            iVar14 = local_res8[0] + iVar4 * -0x2000;
            iVar5 = iVar1;
            if ((iVar4 <= iVar1) && (iVar5 = iVar4, iVar4 < 0)) {
              iVar5 = 0;
            }
            iVar4 = iVar1;
            if ((iVar12 <= iVar1) && (iVar4 = iVar12, iVar12 < 0)) {
              iVar4 = 0;
            }
            if (iVar14 < 0) {
              iVar14 = 0;
            }
            iVar12 = *(int *)(param_7 + (lVar13 - iVar9 * local_58) * 4);
            if (param_8[4] == 0) {
              iVar12 = iVar12 >> 1;
            }
            iVar10 = iVar12 >> 9;
            iVar12 = iVar12 + iVar10 * -0x200;
            local_78 = (longlong)(local_8c + iVar5);
            lVar8 = (longlong)(*param_4 * iVar10);
            iVar2 = *(int *)(param_3 + (local_78 + lVar8) * 4);
            local_60 = (longlong)(local_8c + iVar4);
            iVar3 = *(int *)(param_3 + (local_60 + lVar8) * 4);
            local_70 = (longlong)(iVar5 + local_84);
            iVar5 = *(int *)(param_3 + (local_70 + lVar8) * 4);
            local_68 = (longlong)(local_84 + iVar4);
            iVar4 = *(int *)(local_res18 + (local_68 + lVar8) * 4);
            if (iVar10 == 0) {
              iVar10 = iVar2 * iVar12 + 0x100 >> 9;
              iVar6 = iVar3 * iVar12 + 0x100 >> 9;
              iVar7 = iVar5 * iVar12 + 0x100 >> 9;
              iVar12 = iVar4 * iVar12 + 0x100 >> 9;
            }
            else {
              lVar8 = (longlong)(*param_4 * iVar10 - *local_res20);
              iVar10 = *(int *)(local_res18 + (local_78 + lVar8) * 4);
              iVar6 = *(int *)(local_res18 + (local_60 + lVar8) * 4);
              iVar7 = *(int *)(local_res18 + (local_70 + lVar8) * 4);
              iVar10 = ((iVar2 - iVar10) * iVar12 + 0x100 >> 9) + iVar10;
              iVar6 = ((iVar3 - iVar6) * iVar12 + 0x100 >> 9) + iVar6;
              iVar7 = ((iVar5 - iVar7) * iVar12 + 0x100 >> 9) + iVar7;
              iVar12 = ((iVar4 - *(int *)(local_res18 + (local_68 + lVar8) * 4)) * iVar12 + 0x100 >>
                       9) + *(int *)(local_res18 + (local_68 + lVar8) * 4);
            }
            if (0xfff < iVar10) {
              iVar10 = 0xfff;
            }
            if (0xfff < iVar6) {
              iVar6 = 0xfff;
            }
            if (0xfff < iVar7) {
              iVar7 = 0xfff;
            }
            if (0xfff < iVar12) {
              iVar12 = 0xfff;
            }
            iVar10 = ((iVar6 - iVar10) * iVar14 + 0x1000 >> 0xd) + iVar10;
            iVar10 = (((((iVar12 - iVar7) * iVar14 + 0x1000 >> 0xd) - iVar10) + iVar7) * local_88 +
                      0x1000 >> 0xd) + iVar10;
            if (0xfff < iVar10) {
              iVar10 = 0xfff;
            }
            *(int *)(param_9 + (lVar13 - iVar9 * local_58) * 4) = iVar10;
            iVar9 = *param_8;
            local_res8[0] = local_res8[0] + *(int *)(param_10 + 8);
            local_res10 = local_res10 + 1;
            param_3 = local_res18;
            param_4 = local_res20;
            lVar13 = lVar13 + 1;
            iVar12 = local_80;
          } while (local_res10 < iVar9);
        }
        iVar11 = iVar11 + 1;
      } while (iVar11 <= local_94);
    }
    _vcomp_for_static_end();
  }
  return;
}

