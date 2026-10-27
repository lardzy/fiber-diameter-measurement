// ev_80d438b84b2311c215f041eb32d8336831bd47070a453591d724677dd75f43d5

void FUN_180001d30(longlong param_1,longlong param_2,int *param_3,longlong param_4)

{
  byte bVar1;
  byte bVar2;
  byte bVar3;
  int iVar4;
  uint uVar5;
  longlong lVar6;
  int iVar7;
  int iVar8;
  int iVar9;
  int iVar10;
  int iVar11;
  int iVar12;
  
  iVar8 = 0;
  iVar12 = -1;
  if (0 < param_3[1]) {
    do {
      iVar11 = -1;
      iVar7 = *(int *)(param_1 + 0xc) * iVar8 + (*(int *)(param_1 + 0xc) >> 1) >> 0xd;
      if (iVar7 != iVar12) {
        iVar4 = *param_3;
        iVar9 = 0;
        iVar10 = *(int *)(param_1 + 8) >> 1;
        if (0 < iVar4) {
          do {
            if (iVar10 >> 0xd != iVar11) {
              iVar4 = (iVar4 * iVar8 + iVar9) * 3;
              bVar1 = *(byte *)(iVar4 + param_2);
              lVar6 = (longlong)(iVar4 + 1);
              bVar2 = *(byte *)(lVar6 + param_2);
              bVar3 = *(byte *)(lVar6 + 1 + param_2);
              if ((uint)bVar2 < (uint)bVar1) {
                uVar5 = (uint)bVar3;
                if ((uint)bVar3 < (uint)bVar1) {
                  uVar5 = (uint)bVar1;
                }
              }
              else {
                uVar5 = (uint)bVar3;
                if (bVar2 != uVar5 && uVar5 <= bVar2) {
                  uVar5 = (uint)bVar2;
                }
              }
              iVar11 = iVar11 + 1;
              *(uint *)(param_4 + ((longlong)iVar7 * 0x50 + (longlong)(iVar10 >> 0xd)) * 4) = uVar5;
            }
            iVar4 = *param_3;
            iVar10 = iVar10 + *(int *)(param_1 + 8);
            iVar9 = iVar9 + 1;
          } while (iVar9 < iVar4);
        }
        iVar12 = iVar12 + 1;
      }
      iVar8 = iVar8 + 1;
    } while (iVar8 < param_3[1]);
  }
  return;
}

