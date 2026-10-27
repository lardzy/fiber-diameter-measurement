// ev_446eef3200d7411a48acccc3b40977d1313fb79cc2a3255be23c492345456a65

void FUN_18000c040(int *param_1,int *param_2,int *param_3,longlong param_4,longlong param_5,
                  longlong param_6)

{
  longlong lVar1;
  int iVar2;
  uint uVar3;
  uint uVar4;
  int iVar5;
  uint uVar6;
  int iVar7;
  char cVar8;
  int *piVar9;
  ulonglong uVar10;
  int local_res18 [4];
  int local_38 [4];
  
  if (0 < *param_3) {
    _vcomp_for_static_simple_init(0,*param_3 + -1,1,1,local_res18,local_38);
    if (local_res18[0] <= local_38[0]) {
      iVar2 = *param_1;
      piVar9 = (int *)(param_5 + (longlong)local_res18[0] * 4);
      uVar10 = (ulonglong)((local_38[0] - local_res18[0]) + 1);
      do {
        uVar3 = *(uint *)((param_4 - param_5) + (longlong)piVar9);
        iVar7 = iVar2;
        if (uVar3 != 0) {
          cVar8 = '\0';
          uVar6 = uVar3;
          if (uVar3 < 0x2000) {
            uVar4 = 0x2000;
            cVar8 = '\0';
            do {
              uVar4 = uVar4 >> 1 | (uint)((uVar4 & 1) != 0) << 0x1f;
              uVar6 = uVar6 * 2;
              cVar8 = cVar8 + '\x01';
            } while (uVar3 < uVar4);
          }
          iVar7 = uVar6 - (uVar6 & 0xffffffc0);
          lVar1 = (ulonglong)((uVar6 & 0xffffffc0) - 0x2000 >> 6) * 4;
          iVar7 = (int)((ulonglong)
                        (uint)(iVar7 * *(int *)(&DAT_180014804 + lVar1) +
                              *(int *)(&DAT_180014800 + lVar1) * (0x40 - iVar7)) *
                        (ulonglong)(uint)(*piVar9 << 4) >> (0x26U - cVar8 & 0x3f));
          if (0x7fff < iVar7) {
            iVar7 = 0x7fff;
          }
          iVar7 = (*param_2 * iVar7 + 0x20 >> 6) + iVar2;
        }
        iVar5 = 0x7fff;
        if ((iVar7 < 0x8000) && (iVar5 = iVar7, iVar7 < 0)) {
          iVar5 = 0;
        }
        *(int *)((param_6 - param_5) + (longlong)piVar9) = iVar5;
        piVar9 = piVar9 + 1;
        uVar10 = uVar10 - 1;
      } while (uVar10 != 0);
    }
    _vcomp_for_static_end();
  }
  return;
}

