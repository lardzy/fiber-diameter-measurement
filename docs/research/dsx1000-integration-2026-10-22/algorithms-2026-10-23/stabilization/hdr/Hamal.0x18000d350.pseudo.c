// ev_5b4f5bb183a9201ab3ff53ab86639ba4a2e4d18d1cfb08111a8af6d96293febf

void FUN_18000d350(int *param_1,int *param_2,char *param_3,longlong param_4)

{
  longlong lVar1;
  longlong lVar2;
  ushort uVar3;
  ulonglong uVar4;
  int local_res8 [2];
  int local_18 [4];
  
  if (0 < *param_1) {
    _vcomp_for_static_simple_init(0,*param_1 + -1,1,1,local_res8,local_18);
    if (local_res8[0] <= local_18[0]) {
      lVar2 = (longlong)local_res8[0] * 2;
      uVar4 = (ulonglong)((local_18[0] - local_res8[0]) + 1);
      do {
        lVar1 = *(longlong *)(param_4 + 0x18);
        if (*(int *)(lVar1 + 0xc) == 8) {
          if (*param_3 == '\x01') {
            *(undefined2 *)(lVar2 + *(longlong *)(lVar1 + 0x20)) =
                 *(undefined2 *)(lVar2 + *(longlong *)(param_4 + 0x40));
          }
        }
        else {
          if (*param_2 < 0) {
            uVar3 = *(ushort *)(lVar2 + *(longlong *)(param_4 + 0x40)) >> (-(char)*param_2 & 0x1fU);
          }
          else {
            if ((*(int *)(lVar1 + 0xc) == 8) || (*param_2 < 0)) goto LAB_18000d414;
            uVar3 = *(short *)(*(longlong *)(param_4 + 0x40) + lVar2) << ((byte)*param_2 & 0x1f);
          }
          *(ushort *)(lVar2 + *(longlong *)(lVar1 + 0x20)) = uVar3;
        }
LAB_18000d414:
        lVar2 = lVar2 + 2;
        uVar4 = uVar4 - 1;
      } while (uVar4 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

