// ev_5f2a6ca4ad0d1c930c6f09745f467efd8f2326baa1405ad071fa339a97a88dc8

void FUN_18000b3e0(uint *param_1,longlong param_2,int *param_3,longlong param_4)

{
  int iVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  int *piVar5;
  longlong lVar6;
  ulonglong uVar7;
  ulonglong uVar8;
  longlong lVar9;
  int *piVar10;
  int iVar11;
  int local_res18 [2];
  int local_38 [4];
  
  iVar1 = *param_3;
  if (0 < iVar1) {
    _vcomp_for_static_simple_init(0,iVar1 + -1,1,1,local_res18,local_38);
    if (local_res18[0] <= local_38[0]) {
      uVar2 = *param_1;
      uVar8 = (ulonglong)uVar2;
      do {
        iVar11 = uVar2 * local_res18[0];
        lVar9 = (longlong)(iVar11 * 2);
        if (local_res18[0] == 0) {
          if (0 < (int)uVar2) {
            lVar6 = (longlong)(int)uVar2;
            uVar7 = uVar8;
            piVar5 = (int *)(param_4 + (lVar9 + lVar6) * 4);
            piVar10 = (int *)(param_2 + (longlong)(int)(uVar2 + iVar11) * 4);
            do {
              iVar3 = *(int *)((((iVar11 - lVar9) - lVar6) * 4 - param_4) + param_2 +
                              (longlong)piVar5);
              iVar4 = *piVar10;
              (piVar5 + 1)[-1 - lVar6] = iVar3 >> 1;
              *piVar5 = iVar3 * 3 + iVar4 >> 3;
              uVar7 = uVar7 - 1;
              piVar5 = piVar5 + 1;
              piVar10 = piVar10 + 1;
            } while (uVar7 != 0);
          }
        }
        else if (local_res18[0] == iVar1 + -1) {
          if (0 < (int)uVar2) {
            lVar6 = (longlong)(int)(iVar11 - uVar2);
            piVar5 = (int *)(param_2 + lVar6 * 4);
            uVar7 = uVar8;
            do {
              iVar3 = piVar5[iVar11 - lVar6];
              piVar10 = piVar5 + 1;
              *(int *)(((lVar9 - lVar6) * 4 - param_2) + param_4 + -4 + (longlong)piVar10) =
                   iVar3 * 3 + *piVar5 >> 3;
              *(int *)((((longlong)(int)uVar2 + (lVar9 - lVar6)) * 4 - param_2) + param_4 + -4 +
                      (longlong)piVar10) = iVar3 >> 1;
              uVar7 = uVar7 - 1;
              piVar5 = piVar10;
            } while (uVar7 != 0);
          }
        }
        else if (0 < (int)uVar2) {
          lVar6 = (longlong)(int)(iVar11 - uVar2);
          piVar5 = (int *)(param_2 + (longlong)(int)(uVar2 + iVar11) * 4);
          uVar7 = uVar8;
          piVar10 = (int *)(param_2 + lVar6 * 4);
          do {
            iVar3 = piVar10[iVar11 - lVar6];
            iVar4 = *piVar5;
            piVar5 = piVar5 + 1;
            *(int *)((longlong)piVar10 + ((lVar9 - lVar6) * 4 - param_2) + param_4) =
                 iVar3 * 3 + *piVar10 >> 3;
            *(int *)((((longlong)(int)uVar2 + (lVar9 - lVar6)) * 4 - param_2) + param_4 + -4 +
                    (longlong)(piVar10 + 1)) = iVar4 + iVar3 * 3 >> 3;
            uVar7 = uVar7 - 1;
            piVar10 = piVar10 + 1;
          } while (uVar7 != 0);
        }
        local_res18[0] = local_res18[0] + 1;
      } while (local_res18[0] <= local_38[0]);
    }
    _vcomp_for_static_end();
  }
  return;
}

