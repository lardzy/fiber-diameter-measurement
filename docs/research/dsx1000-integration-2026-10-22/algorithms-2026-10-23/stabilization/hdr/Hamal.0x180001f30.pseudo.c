// ev_22054630b65979110a572ba17ae460f540549681abb1fe004820f83266c4d549

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */
/* WARNING: Type propagation algorithm not settling */

ulonglong FUN_180001f30(undefined8 param_1,undefined8 param_2,longlong param_3)

{
  longlong *plVar1;
  ulonglong uVar2;
  undefined8 uVar3;
  undefined1 auStack_e8 [32];
  undefined8 local_c8;
  int *local_c0;
  int *local_b8;
  int *local_b0;
  int *local_a8;
  int *local_a0;
  int *local_98;
  int local_88;
  int local_84;
  int local_80 [2];
  longlong *local_78;
  int local_70 [2];
  undefined4 local_68 [4];
  int local_58 [4];
  undefined4 local_48;
  int local_44 [11];
  ulonglong local_18;
  
  local_44[1] = -2;
  local_44[2] = -1;
  local_18 = DAT_180019000 ^ (ulonglong)auStack_e8;
  local_58[2] = 8;
  local_80[0] = 0x3c;
  local_68[0] = 0x50;
  local_48 = 0x12c0;
  local_78 = (longlong *)0x0;
  local_84 = 0x10;
  local_88 = 0x70;
  local_58[0] = 0x700;
  local_44[3] = 0x78ed2692;
  local_44[4] = 0x45195b13;
  local_44[5] = 0xcdc34086;
  local_44[6] = 0x5b6aa0a6;
  Ordinal_5(local_44 + 3,&local_78);
  uVar2 = (**(code **)(*local_78 + 0x10))(local_78,(longlong)(local_88 * 0x5c));
  if ((int)uVar2 == 0) {
    uVar3 = (**(code **)(*local_78 + 0x28))();
    local_58[1] = 10;
    local_a0 = local_80;
    local_a8 = local_68;
    local_b8 = &local_88;
    local_c0 = &local_84;
    local_c8 = uVar3;
    local_b0 = (int *)param_2;
    _vcomp_fork(1,7,FUN_18000bcb0,local_58 + 1);
    local_70[0] = local_84 + local_58[0];
    local_a0 = local_80;
    local_a8 = local_68;
    local_b0 = &local_88;
    local_b8 = &local_84;
    local_c0 = local_70;
    local_c8 = uVar3;
    _vcomp_fork(1,7,FUN_18000b940,uVar3);
    local_70[0] = (local_84 + local_80[0]) * local_88;
    local_a8 = &local_88;
    local_b0 = &local_84;
    local_b8 = local_58;
    local_c0 = local_70;
    local_c8 = uVar3;
    _vcomp_fork(1,6,FUN_18000bdc0,uVar3);
    local_44[0] = *(int *)(param_3 + 0x1c) << 0xc;
    local_98 = local_58 + 2;
    local_a0 = local_80;
    local_a8 = local_68;
    local_b8 = &local_48;
    local_c0 = &local_88;
    local_c8 = uVar3;
    local_b0 = (int *)param_3;
    uVar3 = _vcomp_fork(1,9,FUN_18000ba60,local_44);
    plVar1 = local_78;
    if (local_78 != (longlong *)0x0) {
      local_78 = (longlong *)0x0;
      uVar3 = (**(code **)(*plVar1 + 8))();
    }
    if (local_78 != (longlong *)0x0) {
      uVar3 = (**(code **)(*local_78 + 8))();
    }
    uVar2 = CONCAT71((int7)((ulonglong)uVar3 >> 8),1);
  }
  else {
    if (local_78 != (longlong *)0x0) {
      uVar2 = (**(code **)(*local_78 + 8))();
    }
    uVar2 = uVar2 & 0xffffffffffffff00;
  }
  return uVar2;
}

