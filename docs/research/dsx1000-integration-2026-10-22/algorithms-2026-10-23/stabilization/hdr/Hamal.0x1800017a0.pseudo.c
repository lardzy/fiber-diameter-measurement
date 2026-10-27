// ev_e890d2f3b891ce5cc6ae879c667de03f5ac00eebdd7d41215a3736a940d91866

void FUN_1800017a0(longlong param_1,undefined8 param_2,int *param_3,undefined8 param_4,byte param_5)

{
  int local_res8 [4];
  int local_res18 [4];
  
  local_res8[0] = *(int *)(param_1 + 0x58) * *param_3 * 3;
  local_res18[0] = 1 << (param_5 - 1 & 0x1f);
  _param_5 = (1 << (param_5 & 0x1f)) + -1;
  _vcomp_fork(1,6,FUN_18000b200,local_res18,&param_5,local_res8,param_2,param_3,param_4);
  return;
}

