// ev_234979406b657494c894ff95da2717e7a9266c898a2d9a34105816e88bd78267

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_180004200(longlong param_1,longlong param_2,uint *param_3,undefined4 param_4,undefined4 param_5,
             int param_6)

{
  uint uVar1;
  longlong *plVar2;
  ulonglong uVar3;
  undefined8 uVar4;
  undefined8 uVar5;
  longlong lVar6;
  longlong *plVar7;
  undefined1 auStack_b8 [32];
  longlong *local_88;
  longlong *local_80;
  longlong local_78;
  undefined8 local_70;
  undefined8 local_68;
  undefined4 uStack_60;
  undefined4 uStack_5c;
  undefined8 local_58;
  undefined8 local_50;
  ulonglong local_48;
  
  local_70 = 0xfffffffffffffffe;
  local_48 = DAT_180019000 ^ (ulonglong)auStack_b8;
  *(undefined4 *)(param_1 + 0x54) = param_4;
  *(undefined4 *)(param_1 + 0x50) = param_5;
  uVar3 = (ulonglong)(uint)param_6;
  *(int *)(param_1 + 0x58) = param_6;
  uVar1 = *param_3;
  plVar7 = (longlong *)0x0;
  local_88 = (longlong *)0x0;
  local_80 = (longlong *)0x0;
  local_78 = param_1;
  if (param_3[0x4b6] != 0) {
    local_68._0_4_ = 0x7618f959;
    local_68._4_4_ = 0x47d56895;
    uStack_60 = 0x5b823598;
    uStack_5c = 0xd6e805c;
    Ordinal_5(&local_68,&local_88);
    uVar3 = (**(code **)(*local_88 + 0x10))(local_88,(longlong)(int)(uVar1 * param_6));
    if ((int)uVar3 != 0) {
      if (local_80 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_80 + 8))();
      }
      if (local_88 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_88 + 8))();
      }
      return uVar3 & 0xffffffffffffff00;
    }
    uVar4 = (**(code **)(*local_88 + 0x28))();
    uVar5 = FUN_180004d30(param_1,param_2,param_3,uVar4);
    if ((char)uVar5 == '\0') {
      local_50 = 7;
      local_58 = 0;
      local_68._0_4_ = (uint)local_68 & 0xffff0000;
      FUN_180006100(&local_68,L"Error IridixThumScaleLpfMix8bit",0x1f);
      uVar3 = FUN_180009e10((longlong *)(param_1 + 0x78),4000,(wchar_t *)&local_68,0);
      plVar7 = local_80;
      if (local_80 != (longlong *)0x0) {
        local_80 = (longlong *)0x0;
        uVar3 = (**(code **)(*plVar7 + 8))(plVar7);
      }
      plVar7 = local_88;
      if (local_88 != (longlong *)0x0) {
        local_88 = (longlong *)0x0;
        uVar3 = (**(code **)(*plVar7 + 8))();
      }
      if (local_80 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_80 + 8))(local_80);
      }
      if (local_88 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_88 + 8))();
      }
      return uVar3 & 0xffffffffffffff00;
    }
    local_68._0_4_ = 0x78ed2692;
    local_68._4_4_ = 0x45195b13;
    uStack_60 = 0xcdc34086;
    uStack_5c = 0x5b6aa0a6;
    Ordinal_5(&local_68,&local_80);
    uVar3 = (**(code **)(*local_80 + 0x10))(local_80,(longlong)(int)(uVar1 * param_6 * 3));
    if ((int)uVar3 != 0) {
      if (local_80 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_80 + 8))();
      }
      if (local_88 != (longlong *)0x0) {
        uVar3 = (**(code **)(*local_88 + 8))();
      }
      return uVar3 & 0xffffffffffffff00;
    }
    lVar6 = (**(code **)(*local_80 + 0x28))();
    FUN_1800017a0(param_1,param_2,(int *)param_3,lVar6,8);
    FUN_1800019f0(param_1,lVar6,uVar4,(int *)param_3);
    plVar7 = local_88;
    if (local_88 != (longlong *)0x0) {
      local_88 = (longlong *)0x0;
      (**(code **)(*plVar7 + 8))(plVar7);
    }
    plVar7 = local_88;
    uVar3 = FUN_180001810(param_1,lVar6,(int *)param_3,param_2,8);
  }
  plVar2 = local_80;
  if (local_80 != (longlong *)0x0) {
    local_80 = (longlong *)0x0;
    uVar3 = (**(code **)(*plVar2 + 8))();
    plVar7 = local_88;
  }
  if (local_80 != (longlong *)0x0) {
    uVar3 = (**(code **)(*local_80 + 8))();
    plVar7 = local_88;
  }
  if (plVar7 != (longlong *)0x0) {
    uVar3 = (**(code **)(*plVar7 + 8))(plVar7);
  }
  return CONCAT71((int7)(uVar3 >> 8),1);
}

