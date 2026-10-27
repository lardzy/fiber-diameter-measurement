// ev_56f95d71117fcf7a9b354f841e2b97367ccf69b54c3ba5644178bd7cca7d6bc6
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

undefined4 FUN_180002820(uint param_1)

{
  int iVar1;
  int iVar2;
  int iVar3;
  
  iVar3 = 0;
  iVar2 = 0xef;
  if (param_1 < 0x10000) {
    do {
      iVar1 = iVar2 + iVar3 >> 1;
      if (*(int *)(&DAT_180016520 + (longlong)iVar1 * 4) < (int)param_1) {
        iVar3 = iVar1 + 1;
        iVar1 = iVar2;
      }
      iVar2 = iVar1;
    } while (iVar3 < iVar2);
    return *(undefined4 *)(&DAT_180016520 + (longlong)iVar2 * 4);
  }
  return 0xffffffff;
}

