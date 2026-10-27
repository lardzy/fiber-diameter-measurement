// ev_7c5c769b43c0dcee0c304ffef04c16b0dcd5d957393cd26956a533dae7883812

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */
/* WARNING: Type propagation algorithm not settling */

ulonglong FUN_1800025e0(longlong param_1,undefined8 param_2,undefined8 param_3,undefined8 param_4,
                       undefined8 param_5,longlong param_6,int param_7,undefined8 param_8)

{
  longlong *plVar1;
  ulonglong uVar2;
  undefined8 uVar3;
  undefined1 auStack_c8 [32];
  int *local_a8;
  int *local_a0;
  int *local_98;
  undefined4 *local_90;
  int *local_88;
  undefined8 local_80;
  longlong local_78;
  undefined8 local_70;
  longlong local_68;
  longlong *local_58;
  int local_50;
  int local_4c;
  int local_48 [4];
  undefined4 local_38 [4];
  undefined8 local_28;
  undefined4 local_20;
  undefined4 uStack_1c;
  undefined4 uStack_18;
  undefined4 uStack_14;
  ulonglong local_10;
  
  local_28 = 0xfffffffffffffffe;
  local_10 = DAT_180019000 ^ (ulonglong)auStack_c8;
  local_48[2] = 0x3c;
  local_38[0] = 0x50;
  local_48[1] = 0x12c0;
  local_50 = param_7 * 0x12c0;
  local_58 = (longlong *)0x0;
  local_20 = 0x7618f959;
  uStack_1c = 0x47d56895;
  uStack_18 = 0x5b823598;
  uStack_14 = 0xd6e805c;
  Ordinal_5(&local_20,&local_58);
  uVar2 = (**(code **)(*local_58 + 0x10))(local_58,(longlong)local_50);
  if ((int)uVar2 == 0) {
    uVar3 = (**(code **)(*local_58 + 0x28))();
    local_a8 = &local_50;
    local_a0 = (int *)param_2;
    _vcomp_fork(1,3,FUN_18000c3f0,uVar3);
    local_a0 = local_48 + 1;
    local_a8 = &local_50;
    _vcomp_fork(1,3,FUN_18000c920,uVar3);
    local_48[0] = *(int *)(param_1 + 0x50);
    if (local_48[0] == *(int *)(param_1 + 0x54)) {
      local_48[0] = *(int *)(param_6 + 4);
      local_4c = local_48[0] - *(int *)(param_1 + 0x58);
    }
    else {
      local_4c = (local_48[0] + -1) * *(int *)(param_1 + 0x58);
      local_48[0] = *(int *)(param_1 + 0x58) * local_48[0];
    }
    local_70 = param_8;
    local_78 = param_6;
    local_80 = param_5;
    local_88 = local_48 + 2;
    local_90 = local_38;
    local_98 = local_48 + 1;
    local_a8 = &local_4c;
    local_a0 = (int *)uVar3;
    local_68 = param_1;
    uVar3 = _vcomp_fork(1,10,FUN_18000c9e0,local_48);
    plVar1 = local_58;
    if (local_58 != (longlong *)0x0) {
      local_58 = (longlong *)0x0;
      uVar3 = (**(code **)(*plVar1 + 8))();
    }
    if (local_58 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_58 + 8))();
    }
    uVar2 = CONCAT71((int7)((ulonglong)uVar3 >> 8),1);
  }
  else {
    if (local_58 != (longlong *)0x0) {
      uVar2 = (**(code **)(*local_58 + 8))();
    }
    uVar2 = uVar2 & 0xffffffffffffff00;
  }
  return uVar2;
}

