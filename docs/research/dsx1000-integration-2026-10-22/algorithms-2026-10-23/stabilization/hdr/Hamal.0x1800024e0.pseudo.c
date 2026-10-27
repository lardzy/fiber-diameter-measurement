// ev_72d0985aa18df11abdcd14ed46610cca5cb43f69a05a4785a8203f639e091aef

undefined8
FUN_1800024e0(longlong param_1,undefined8 param_2,undefined8 param_3,int *param_4,undefined8 param_5
             )

{
  undefined8 uVar1;
  int local_res8 [6];
  int local_res20 [2];
  int local_18 [6];
  
  local_res8[0] = *(int *)(param_1 + 0x58) * *param_4;
  local_res20[0] = param_4[2];
  local_18[0] = (0x40 - local_res20[0]) * 4;
  uVar1 = _vcomp_fork(1,7,FUN_18000c040,local_18,local_res20,local_res8,param_2,param_3,param_5,
                      param_1);
  return CONCAT71((int7)((ulonglong)uVar1 >> 8),1);
}

