// ev_1068a2449982f243613568e4dd689f27f92bfbc503d9fc17e495fca7a5ee3265

undefined8 FUN_180009ca0(undefined8 param_1,longlong *param_2,undefined8 *param_3)

{
  undefined8 uVar1;
  undefined8 *puVar2;
  wchar_t *pwVar3;
  
  if (param_3 == (undefined8 *)0x0) {
    uVar1 = 0x80070057;
  }
  else {
    if ((*param_2 == 0x4e37dd17ccf7bb59) && (param_2[1] == 0x747b2ffe378985b4)) {
      puVar2 = operator_new(0x68);
      if (puVar2 != (undefined8 *)0x0) {
        *puVar2 = CHalationRejectionCmd::vftable;
        puVar2[1] = CHalationRejectionCmd::vftable;
        *(undefined4 *)(puVar2 + 0xc) = 1;
        puVar2[2] = 0;
        puVar2[3] = 0;
        puVar2[4] = 0;
        puVar2[6] = 0;
        puVar2[0xb] = 0;
      }
      *param_3 = puVar2;
      pwVar3 = 
      L"CHalationRejectionService::CreateCmdInstance(REFIID = IID_IHalationRejectionCmd . ppv =%p)\n"
      ;
    }
    else {
      if ((*param_2 != 0x42b26d6ab99a52e6) || (param_2[1] != -0x5bda695211429d49)) {
        *param_3 = 0;
        return 0x80004002;
      }
      puVar2 = operator_new(0x60);
      if (puVar2 != (undefined8 *)0x0) {
        *puVar2 = CCustomHalationRejectionCmd::vftable;
        puVar2[1] = CCustomHalationRejectionCmd::vftable;
        *(undefined4 *)(puVar2 + 0xb) = 1;
        puVar2[2] = 0;
        puVar2[3] = 0;
        puVar2[4] = 0;
        puVar2[6] = 0;
      }
      *param_3 = puVar2;
      pwVar3 = 
      L"CHalationRejectionService::CreateCmdInstance(REFIID = IID_ICustomHalationRejectionCmd . ppv =%p)\n"
      ;
    }
    LogTraceV(pwVar3,param_3);
    uVar1 = 0;
  }
  return uVar1;
}

