// ev_f62059232aa366bb55540ca3a0c2bd415716316083e5a0364eb851dfc4b2e1f9

void FUN_1800019f0(longlong param_1,undefined8 param_2,undefined8 param_3,int *param_4)

{
  int local_res8 [6];
  int local_res20 [2];
  undefined4 uStack0000000000000028;
  int local_18 [6];
  
  local_res8[0] = param_4[3];
  uStack0000000000000028 = 0x80;
  local_res20[0] = 0x40 - local_res8[0];
  local_18[0] = *(int *)(param_1 + 0x58) * *param_4;
  _vcomp_fork(1,6,FUN_18000b7a0,local_18,local_res20,local_res8,&stack0x00000028,param_2,param_3);
  return;
}

