// ev_8466e68820612e0b3fd609ed7d582b2c57cb6667dc3dbf35274da031e51d331b

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

ulonglong FUN_180004d30(longlong param_1,undefined8 param_2,uint *param_3,undefined8 param_4)

{
  uint uVar1;
  int iVar2;
  ulonglong uVar3;
  undefined8 uVar4;
  undefined8 uVar5;
  ulonglong uVar6;
  undefined8 uVar7;
  longlong *plVar8;
  wchar_t *pwVar9;
  uint *puVar10;
  longlong lVar11;
  undefined1 auStackY_c8 [32];
  longlong *local_88;
  longlong *local_80;
  longlong *local_78;
  longlong local_70;
  uint *local_68;
  undefined8 local_60;
  undefined8 local_58;
  undefined4 uStack_50;
  undefined4 uStack_4c;
  undefined8 local_48;
  undefined8 local_40;
  ulonglong local_38;
  
  local_60 = 0xfffffffffffffffe;
  local_38 = DAT_180019000 ^ (ulonglong)auStackY_c8;
  uVar1 = *param_3;
  iVar2 = *(int *)(param_1 + 0x58);
  local_80 = (longlong *)0x0;
  local_78 = (longlong *)0x0;
  local_88 = (longlong *)0x0;
  local_58._0_4_ = 0x7618f959;
  local_58._4_4_ = 0x47d56895;
  uStack_50 = 0x5b823598;
  uStack_4c = 0xd6e805c;
  local_70 = param_1;
  local_68 = param_3;
  Ordinal_5(&local_58,&local_80);
  lVar11 = (longlong)(int)(uVar1 * iVar2);
  uVar3 = (**(code **)(*local_80 + 0x10))(local_80,lVar11);
  if ((int)uVar3 != 0) {
    if (local_88 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_88 + 8))();
    }
    if (local_78 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_78 + 8))();
    }
    if (local_80 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_80 + 8))();
    }
    return uVar3 & 0xffffffffffffff00;
  }
  uVar4 = (**(code **)(*local_80 + 0x28))();
  FUN_180001ad0(param_1,param_2,(int *)param_3,uVar4);
  local_58._0_4_ = 0x7618f959;
  local_58._4_4_ = 0x47d56895;
  uStack_50 = 0x5b823598;
  uStack_4c = 0xd6e805c;
  Ordinal_5(&local_58,&local_78);
  uVar3 = (**(code **)(*local_78 + 0x10))(local_78,lVar11);
  if ((int)uVar3 != 0) {
    if (local_88 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_88 + 8))();
    }
    if (local_78 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_78 + 8))();
    }
    if (local_80 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_80 + 8))();
    }
    return uVar3 & 0xffffffffffffff00;
  }
  plVar8 = local_78;
  uVar5 = (**(code **)(*local_78 + 0x28))();
  uVar1 = *param_3;
  uVar3 = (ulonglong)uVar1;
  puVar10 = param_3;
  if (param_3[5] == 0) {
    FUN_180005fb0(plVar8,uVar4,param_3,uVar1,*(int *)(param_1 + 0x58),uVar5);
  }
  else {
    FUN_180003fe0(param_1,uVar4,param_3,uVar1,*(int *)(param_1 + 0x58),uVar5);
  }
  local_58._0_4_ = 0x7618f959;
  local_58._4_4_ = 0x47d56895;
  uStack_50 = 0x5b823598;
  uStack_4c = 0xd6e805c;
  Ordinal_5(&local_58,&local_88);
  uVar6 = (**(code **)(*local_88 + 0x10))(local_88,lVar11);
  if ((int)uVar6 != 0) {
    if (local_88 != (longlong *)0x0) {
      uVar6 = (**(code **)(*local_88 + 8))();
    }
    if (local_78 != (longlong *)0x0) {
      uVar6 = (**(code **)(*local_78 + 8))();
    }
    if (local_80 != (longlong *)0x0) {
      uVar6 = (**(code **)(*local_80 + 8))();
    }
    return uVar6 & 0xffffffffffffff00;
  }
  uVar7 = (**(code **)(*local_88 + 0x28))();
  if (param_3[4] == 0) {
    uVar5 = FUN_1800025e0(param_1,*(undefined8 *)(param_1 + 0x60),puVar10,uVar3,uVar5,
                          (longlong)param_3,8,uVar7);
    if ((char)uVar5 != '\x01') {
      uVar3 = 0x31;
      pwVar9 = L"Error when run MakeOutExt_mod8bit function(pnLPF)";
      goto LAB_180004f77;
    }
LAB_18000502d:
    plVar8 = local_78;
    if (local_78 != (longlong *)0x0) {
      local_78 = (longlong *)0x0;
      (**(code **)(*plVar8 + 8))();
    }
    uVar4 = FUN_1800024e0(param_1,uVar4,uVar7,(int *)param_3,param_4);
    plVar8 = local_80;
    if ((char)uVar4 == '\x01') {
      if (local_80 != (longlong *)0x0) {
        local_80 = (longlong *)0x0;
        (**(code **)(*plVar8 + 8))();
      }
      plVar8 = local_88;
      if (local_88 != (longlong *)0x0) {
        local_88 = (longlong *)0x0;
        (**(code **)(*plVar8 + 8))();
      }
      uVar3 = (ulonglong)*(uint *)(param_1 + 0x54);
      if (*(uint *)(param_1 + 0x50) == *(uint *)(param_1 + 0x54)) {
        if (param_3[4] == 0) {
          plVar8 = *(longlong **)(param_1 + 0x68);
        }
        else {
          plVar8 = *(longlong **)(param_1 + 0x70);
        }
        uVar3 = (**(code **)(*plVar8 + 8))();
      }
      if (local_88 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_88 + 8))();
      }
      if (local_78 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_78 + 8))();
      }
      if (local_80 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_80 + 8))();
      }
      return CONCAT71((int7)(uVar3 >> 8),1);
    }
    local_40 = 7;
    local_48 = 0;
    local_58._0_4_ = (uint)local_58 & 0xffff0000;
    FUN_180006100(&local_58,L"Error when run MakeAmp8bit2",0x1b);
    uVar3 = FUN_180009e10((longlong *)(param_1 + 0x78),4000,(wchar_t *)&local_58,0);
    plVar8 = local_80;
    if (local_80 != (longlong *)0x0) {
      local_80 = (longlong *)0x0;
      uVar3 = (**(code **)(*plVar8 + 8))(plVar8);
    }
    plVar8 = local_88;
    if (local_88 != (longlong *)0x0) {
      local_88 = (longlong *)0x0;
      uVar3 = (**(code **)(*plVar8 + 8))();
    }
    if (local_88 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_88 + 8))();
    }
    if (local_78 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_78 + 8))();
    }
    if (local_80 == (longlong *)0x0) goto LAB_1800051dc;
    lVar11 = *local_80;
  }
  else {
    uVar5 = FUN_1800025e0(param_1,*(undefined8 *)(param_1 + 0x60),puVar10,uVar3,uVar5,
                          (longlong)param_3,0x10,uVar7);
    if ((char)uVar5 == '\x01') goto LAB_18000502d;
    uVar3 = 0x32;
    pwVar9 = L"Error when run MakeOutExt_mod8bit function(pnLPF2)";
LAB_180004f77:
    local_40 = 7;
    local_48 = 0;
    local_58._0_4_ = (uint)local_58 & 0xffff0000;
    FUN_180006100(&local_58,pwVar9,uVar3);
    FUN_180009e10((longlong *)(param_1 + 0x78),4000,(wchar_t *)&local_58,0);
    plVar8 = local_78;
    if (local_78 != (longlong *)0x0) {
      local_78 = (longlong *)0x0;
      (**(code **)(*plVar8 + 8))();
    }
    plVar8 = local_88;
    if (local_88 != (longlong *)0x0) {
      local_88 = (longlong *)0x0;
      (**(code **)(*plVar8 + 8))();
    }
    if (param_3[4] == 0) {
      plVar8 = *(longlong **)(param_1 + 0x68);
    }
    else {
      plVar8 = *(longlong **)(param_1 + 0x70);
    }
    uVar3 = (**(code **)(*plVar8 + 8))();
    if (local_88 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_88 + 8))();
    }
    if (local_78 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_78 + 8))();
    }
    if (local_80 == (longlong *)0x0) goto LAB_1800051dc;
    lVar11 = *local_80;
  }
  uVar3 = (**(code **)(lVar11 + 8))(local_80);
LAB_1800051dc:
  return uVar3 & 0xffffffffffffff00;
}

