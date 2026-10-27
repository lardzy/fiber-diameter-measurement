// ev_8bee991df19cff034cc65a0abfbf3565e80a1f50f87f923029d1123f74ba1779
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

undefined8 FUN_1800034f0(int *param_1,undefined8 *param_2,longlong *param_3)

{
  double *pdVar1;
  longlong lVar2;
  longlong lVar3;
  int iVar4;
  longlong lVar5;
  double dVar6;
  double dVar7;
  double dVar8;
  double dVar9;
  
  iVar4 = 0;
  if (0 < param_1[0xb]) {
    pdVar1 = (double *)*param_2;
    lVar2 = *param_3;
    lVar5 = 0;
    do {
      dVar8 = *(double *)(param_1 + 2);
      lVar3 = *(longlong *)(param_1 + 8);
      dVar7 = *(double *)(lVar5 + *(longlong *)(param_1 + 6)) + pdVar1[1];
      dVar9 = *(double *)(*(longlong *)(param_1 + 4) + lVar5) + *pdVar1;
      dVar6 = sin(dVar7 * dVar8 * 3.141592653589793);
      dVar7 = sin((dVar7 * 3.141592653589793) / (double)param_1[1]);
      dVar8 = sin(dVar9 * dVar8 * 3.141592653589793);
      dVar9 = sin((dVar9 * 3.141592653589793) / (double)*param_1);
      iVar4 = iVar4 + 1;
      *(double *)(lVar5 + lVar2) = *(double *)(lVar5 + lVar3) - (dVar6 / dVar7) * (dVar8 / dVar9);
      lVar5 = lVar5 + 8;
    } while (iVar4 < param_1[0xb]);
  }
  return 0;
}

