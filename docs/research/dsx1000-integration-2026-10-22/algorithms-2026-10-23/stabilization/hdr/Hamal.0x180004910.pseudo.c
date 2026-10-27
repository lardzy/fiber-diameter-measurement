// ev_3266da6551cef6164c1b8722f6794f3f5d78ae2ccde398adefb0c6362c0e9763

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

ulonglong FUN_180004910(longlong param_1,longlong param_2,int *param_3)

{
  ulonglong uVar1;
  longlong lVar2;
  undefined8 uVar3;
  undefined8 uVar4;
  longlong *plVar5;
  longlong lVar6;
  undefined1 auStackY_b8 [32];
  undefined8 in_stack_ffffffffffffff68;
  longlong *local_78;
  longlong *local_70;
  longlong local_68;
  undefined8 local_60;
  ulonglong local_58 [6];
  
  local_60 = 0xfffffffffffffffe;
  local_58[4] = DAT_180019000 ^ (ulonglong)auStackY_b8;
  local_70 = (longlong *)0x0;
  local_78 = (longlong *)0x0;
  local_58[0] = 0x47d568957618f959;
  local_58[1] = 0xd6e805c5b823598;
  local_68 = param_1;
  Ordinal_5(local_58,&local_70);
  uVar1 = (**(code **)(*local_70 + 0x10))(local_70,0x12c0);
  if ((int)uVar1 == 0) {
    lVar2 = (**(code **)(*local_70 + 0x28))();
    *(int *)(param_1 + 8) = (int)((79.0 / (double)(*param_3 + -2)) * 8192.0 + 0.5);
    *(int *)(param_1 + 0xc) = (int)((59.0 / (double)(param_3[1] + -2)) * 8192.0 + 0.5);
    lVar6 = lVar2;
    FUN_180001d30(param_1,param_2,param_3,lVar2);
    if (param_3[5] != 0) {
      FUN_180002a70(param_1,lVar2,param_3);
    }
    local_58[0] = 0x47d568957618f959;
    local_58[1] = 0xd6e805c5b823598;
    Ordinal_5(local_58,&local_78);
    uVar1 = (**(code **)(*local_78 + 0x10))(local_78,0x12c0);
    if ((int)uVar1 == 0) {
      plVar5 = local_78;
      uVar3 = (**(code **)(*local_78 + 0x28))();
      if (param_3[5] == 0) {
        FUN_180003190(plVar5,lVar2,param_3,lVar6,in_stack_ffffffffffffff68,uVar3);
      }
      else {
        FUN_180003230(param_1,lVar2,param_3,lVar6,in_stack_ffffffffffffff68,uVar3);
      }
      plVar5 = local_70;
      if (local_70 != (longlong *)0x0) {
        local_70 = (longlong *)0x0;
        (**(code **)(*plVar5 + 8))();
      }
      local_58[0] = 0x47d568957618f959;
      local_58[1] = 0xd6e805c5b823598;
      Ordinal_5(local_58,param_1 + 0x68);
      uVar1 = (**(code **)(**(longlong **)(param_1 + 0x68) + 0x10))
                        (*(longlong **)(param_1 + 0x68),0x9600);
      if ((int)uVar1 == 0) {
        uVar4 = (**(code **)(**(longlong **)(param_1 + 0x68) + 0x28))();
        uVar3 = FUN_180001f30(param_1,uVar3,(longlong)param_3);
        plVar5 = local_78;
        if ((char)uVar3 == '\x01') {
          if (local_78 != (longlong *)0x0) {
            local_78 = (longlong *)0x0;
            (**(code **)(*plVar5 + 8))();
          }
          uVar3 = FUN_180001b10(plVar5,uVar4,param_3);
          if ((char)uVar3 == '\x01') {
            if (param_3[4] == 0) {
              *(undefined8 *)(param_1 + 0x60) = uVar4;
            }
            else {
              FUN_1800060c0(param_1 + 0x70);
              uVar1 = (**(code **)(**(longlong **)(param_1 + 0x70) + 0x10))
                                (*(longlong **)(param_1 + 0x70),0x12c00);
              if ((int)uVar1 != 0) {
                if (local_78 != (longlong *)0x0) {
                  uVar1 = (**(code **)(*local_78 + 8))();
                }
                if (local_70 != (longlong *)0x0) {
                  uVar1 = (**(code **)(*local_70 + 8))();
                }
                return uVar1 & 0xffffffffffffff00;
              }
              plVar5 = *(longlong **)(param_1 + 0x70);
              uVar3 = (**(code **)(*plVar5 + 0x28))();
              *(undefined8 *)(param_1 + 0x60) = uVar3;
              FUN_180001bd0(plVar5,uVar4);
              uVar3 = (**(code **)(**(longlong **)(param_1 + 0x68) + 8))();
            }
            if (local_78 != (longlong *)0x0) {
              uVar3 = (**(code **)(*local_78 + 8))();
            }
            if (local_70 != (longlong *)0x0) {
              uVar3 = (**(code **)(*local_70 + 8))();
            }
            return CONCAT71((int7)((ulonglong)uVar3 >> 8),1);
          }
          local_58[3] = 7;
          local_58[2] = 0;
          local_58[0] = local_58[0] & 0xffffffffffff0000;
          FUN_180006100(local_58,L"Error when run NoiseSuppression8bit function",0x2c);
          FUN_180009e10((longlong *)(param_1 + 0x78),4000,(wchar_t *)local_58,0);
          uVar1 = (**(code **)(**(longlong **)(param_1 + 0x68) + 8))();
        }
        else {
          local_58[3] = 7;
          local_58[2] = 0;
          local_58[0] = local_58[0] & 0xffffffffffff0000;
          FUN_180006100(local_58,L"Error when run LpfProcFilter8bit function",0x29);
          FUN_180009e10((longlong *)(param_1 + 0x78),4000,(wchar_t *)local_58,0);
          uVar1 = (**(code **)(**(longlong **)(param_1 + 0x68) + 8))();
          plVar5 = local_78;
          if (local_78 != (longlong *)0x0) {
            local_78 = (longlong *)0x0;
            uVar1 = (**(code **)(*plVar5 + 8))();
          }
        }
        if (local_78 != (longlong *)0x0) {
          uVar1 = (**(code **)(*local_78 + 8))();
        }
        if (local_70 != (longlong *)0x0) {
          uVar1 = (**(code **)(*local_70 + 8))();
        }
        uVar1 = uVar1 & 0xffffffffffffff00;
      }
      else {
        if (local_78 != (longlong *)0x0) {
          uVar1 = (**(code **)(*local_78 + 8))();
        }
        if (local_70 != (longlong *)0x0) {
          uVar1 = (**(code **)(*local_70 + 8))();
        }
        uVar1 = uVar1 & 0xffffffffffffff00;
      }
    }
    else {
      if (local_78 != (longlong *)0x0) {
        uVar1 = (**(code **)(*local_78 + 8))();
      }
      if (local_70 != (longlong *)0x0) {
        uVar1 = (**(code **)(*local_70 + 8))();
      }
      uVar1 = uVar1 & 0xffffffffffffff00;
    }
  }
  else {
    if (local_78 != (longlong *)0x0) {
      uVar1 = (**(code **)(*local_78 + 8))();
    }
    if (local_70 != (longlong *)0x0) {
      uVar1 = (**(code **)(*local_70 + 8))();
    }
    uVar1 = uVar1 & 0xffffffffffffff00;
  }
  return uVar1;
}

