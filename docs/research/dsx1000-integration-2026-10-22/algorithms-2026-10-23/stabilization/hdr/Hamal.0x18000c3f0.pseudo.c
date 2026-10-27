// ev_ccfd9e2f3025f2c8bb65d5a35023e13261b3b0d15aabb841ba553d2b4781c3d9

void FUN_18000c3f0(longlong param_1,int *param_2,longlong param_3)

{
  undefined4 *puVar1;
  ulonglong uVar2;
  int local_res10 [4];
  int local_res20 [2];
  
  if (0 < *param_2) {
    _vcomp_for_static_simple_init(0,*param_2 + -1,1,1,local_res10,local_res20);
    if (local_res10[0] <= local_res20[0]) {
      uVar2 = (ulonglong)((local_res20[0] - local_res10[0]) + 1);
      puVar1 = (undefined4 *)(param_1 + (longlong)local_res10[0] * 4);
      do {
        *puVar1 = *(undefined4 *)((param_3 - param_1) + (longlong)puVar1);
        uVar2 = uVar2 - 1;
        puVar1 = puVar1 + 1;
      } while (uVar2 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

