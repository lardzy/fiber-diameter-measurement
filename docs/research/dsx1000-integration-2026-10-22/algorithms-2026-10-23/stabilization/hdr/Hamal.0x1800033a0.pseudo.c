// ev_39ab63f3fb4b3f69cc6cc1a985288fb25f3ba7c5c92f9e837b102c8287e751c2

ulonglong FUN_1800033a0(longlong param_1,longlong param_2)

{
  int *piVar1;
  uint *puVar2;
  ulonglong uVar3;
  uint uVar4;
  uint uVar5;
  wchar_t local_28 [8];
  undefined8 local_18;
  undefined8 local_10;
  
  uVar4 = 0;
  if (*(uint *)(param_2 + 0x14) < 2) {
    if (*(uint *)(param_2 + 0x10) < 2) {
      if (*(uint *)(param_2 + 8) < 0x100) {
        if (*(uint *)(param_2 + 0xc) < 0x100) {
          if (*(int *)(param_2 + 0x18) - 1U < 0x1dc7) {
            if (*(int *)(param_2 + 0x1c) - 0x22U < 0x3ffde) {
              if (*(int *)(param_2 + 0x20) == 0x12) {
                if (*(uint *)(param_2 + 0x12d8) < 2) {
                  piVar1 = (int *)(param_2 + 0x11a0);
                  uVar5 = uVar4;
                  do {
                    if ((0xff < piVar1[-0x1e]) || (piVar1[-0x1e] < 0)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"AsymmetryIn parameter invalid: AsymmetryIn[%d] = %d",0x33);
                      goto LAB_180003805;
                    }
                    if ((0x1fff < *piVar1) || (*piVar1 < 0)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"AsymmetryOut parameter invalid: AsymmetryOut[%d] = %d",0x35);
                      goto LAB_180003805;
                    }
                    if (0x1ffff < (uint)piVar1[-0xf]) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"AsymmetrySlp parameter invalid: AsymmetrySlp[%d] = %d",0x35);
                      goto LAB_180003805;
                    }
                    uVar5 = uVar5 + 1;
                    piVar1 = piVar1 + 1;
                  } while ((int)uVar5 < 0xf);
                  piVar1 = (int *)(param_2 + 0x11fc);
                  uVar5 = uVar4;
                  do {
                    if ((0x7fff < piVar1[-8]) || (piVar1[-8] < 0)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"NoiseSupSlp1 parameter invalid: NoiseSupSlp1[%d] = %d",0x35);
                      goto LAB_180003805;
                    }
                    if ((0x1fff < *piVar1) || (*piVar1 < 0)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"NoiseSupSlp2 parameter invalid: NoiseSupSlp2[%d] = %d",0x35);
                      goto LAB_180003805;
                    }
                    if (0x3ff < (uint)piVar1[8]) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"NoiseSupBin parameter invalid: NoiseSupBin[%d] = %d",0x33);
                      goto LAB_180003805;
                    }
                    uVar5 = uVar5 + 1;
                    piVar1 = piVar1 + 1;
                  } while ((int)uVar5 < 8);
                  piVar1 = (int *)(param_2 + 0x1258);
                  uVar5 = uVar4;
                  do {
                    if ((0x3fff < piVar1[-7]) || (piVar1[-7] < 0)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"NoiseSupTanhIn parameter invalid: NoiseSupTanhIn[%d] = %d",
                                    0x39);
                      goto LAB_180003805;
                    }
                    if ((0x3f < *piVar1) || (*piVar1 < 0)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"NoiseSupTanhSlp parameter invalid: NoiseSupTanhSlp[%d] = %d",
                                    0x3b);
                      goto LAB_180003805;
                    }
                    if (0xfff < (uint)piVar1[7]) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"NoiseSupTanhOut parameter invalid: NoiseSupTanhOut[%d] = %d",
                                    0x3b);
                      goto LAB_180003805;
                    }
                    uVar5 = uVar5 + 1;
                    piVar1 = piVar1 + 1;
                  } while ((int)uVar5 < 7);
                  piVar1 = (int *)(param_2 + 0x12b4);
                  uVar5 = uVar4;
                  do {
                    if ((0x3ff < piVar1[-9]) || (piVar1[-9] < -0x3ff)) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"Rgb2YCbCr colour matrix  parameter invalid: Rgb2YCbCr[%d] = %d"
                                    ,0x3e);
                      goto LAB_180003805;
                    }
                    if (0xffe < *piVar1 + 0x7ffU) {
                      local_10 = 7;
                      local_18 = 0;
                      local_28[0] = L'\0';
                      FUN_180006100((ulonglong *)local_28,
                                    L"YCbCr2Rgb colour matrix  parameter invalid: YCbCr2Rgb[%d] = %d"
                                    ,0x3e);
                      goto LAB_180003805;
                    }
                    uVar5 = uVar5 + 1;
                    piVar1 = piVar1 + 1;
                  } while ((int)uVar5 < 9);
                  puVar2 = (uint *)(param_2 + 0x24);
                  uVar5 = uVar4;
                  while (*puVar2 < 8) {
                    uVar5 = uVar5 + 1;
                    puVar2 = puVar2 + 1;
                    if (0x440 < (int)uVar5) {
                      return CONCAT71((int7)((ulonglong)puVar2 >> 8),1);
                    }
                  }
                  local_10 = 7;
                  local_18 = 0;
                  local_28[0] = L'\0';
                  FUN_180006100((ulonglong *)local_28,
                                L"FilterCoef parameter invalid: FilterCoef[%d] = %d",0x31);
LAB_180003805:
                  uVar3 = FUN_180009e10((longlong *)(param_1 + 0x78),4000,local_28,(ulonglong)uVar5)
                  ;
                  return uVar3 & 0xffffffffffffff00;
                }
                local_10 = 7;
                local_18 = 0;
                local_28[0] = L'\0';
                FUN_180006100((ulonglong *)local_28,L"ExecSwitch parameter invalid: ExecSwitch = %d"
                              ,0x2d);
                uVar3 = (ulonglong)*(uint *)(param_2 + 0x12d8);
              }
              else {
                local_10 = 7;
                local_18 = 0;
                local_28[0] = L'\0';
                FUN_180006100((ulonglong *)local_28,
                              L"LpfCoefSumDivBit parameter invalid: LfpCoefSumDivBit = %d",0x39);
                uVar3 = (ulonglong)*(uint *)(param_2 + 0x20);
              }
            }
            else {
              local_10 = 7;
              local_18 = 0;
              local_28[0] = L'\0';
              FUN_180006100((ulonglong *)local_28,
                            L"LpfCoefSumDiv parameter invalid: LfpCoefSumDiv = %d",0x33);
              uVar3 = (ulonglong)*(uint *)(param_2 + 0x1c);
            }
          }
          else {
            local_10 = 7;
            local_18 = 0;
            local_28[0] = L'\0';
            FUN_180006100((ulonglong *)local_28,L"LpfCoefSum parameter invalid: LfpCoefSum = %d",
                          0x2d);
            uVar3 = (ulonglong)*(uint *)(param_2 + 0x18);
          }
        }
        else {
          local_10 = 7;
          local_18 = 0;
          local_28[0] = L'\0';
          FUN_180006100((ulonglong *)local_28,L"Saturation parameter invalid: Saturation = %d",0x2d)
          ;
          uVar3 = (ulonglong)*(uint *)(param_2 + 0xc);
        }
      }
      else {
        local_10 = 7;
        local_18 = 0;
        local_28[0] = L'\0';
        FUN_180006100((ulonglong *)local_28,L"Strength parameter invalid: Strength = %d",0x29);
        uVar3 = (ulonglong)*(uint *)(param_2 + 8);
      }
    }
    else {
      local_10 = 7;
      local_18 = 0;
      local_28[0] = L'\0';
      FUN_180006100((ulonglong *)local_28,L"BlendMode parameter invalid: BlendMode = %d",0x2b);
      uVar3 = (ulonglong)*(uint *)(param_2 + 0x10);
    }
  }
  else {
    local_10 = 7;
    local_18 = 0;
    local_28[0] = L'\0';
    FUN_180006100((ulonglong *)local_28,L"AsymMode parameter invalid: AsymMode = %d",0x29);
    uVar3 = (ulonglong)*(uint *)(param_2 + 0x14);
  }
  uVar3 = FUN_180009e10((longlong *)(param_1 + 0x78),4000,local_28,uVar3);
  return uVar3 & 0xffffffffffffff00;
}

