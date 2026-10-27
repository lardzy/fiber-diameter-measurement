# sg64-evidence.json — Ghidra 输出摘录

这是反编译伪码，不是原始源码；变量类型可能需要汇编核对。

## {'procedure': '0x180014730'}
Evidence: `ev_0799f14d755a466d2e9a367d59d872400de124472efa12b697f09c9285e8418f`

```c

void FUN_180014730(int param_1,int param_2,undefined8 param_3,longlong *param_4,float param_5,
                  float param_6)

{
  int iVar1;
  longlong lVar2;
  int iVar3;
  int iVar4;
  
  iVar4 = 0;
  if (0 < param_2) {
    do {
      iVar3 = 0;
      if (3 < param_1) {
        do {
          lVar2 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar4 + iVar3);
          *(float *)(*param_4 + lVar2 * 4) =
               (float)(int)*(float *)(*param_4 + lVar2 * 4) * param_6 + param_5;
          lVar2 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar4 + 1 + iVar3);
          *(float *)(*param_4 + lVar2 * 4) =
               (float)(int)*(float *)(*param_4 + lVar2 * 4) * param_6 + param_5;
          lVar2 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar4 + 2 + iVar3);
          *(float *)(*param_4 + lVar2 * 4) =
               (float)(int)*(float *)(*param_4 + lVar2 * 4) * param_6 + param_5;
          iVar1 = (int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar4 + 3 + iVar3;
          iVar3 = iVar3 + 4;
          lVar2 = (longlong)iVar1;
          *(float *)(*param_4 + lVar2 * 4) =
               (float)(int)*(float *)(*param_4 + lVar2 * 4) * param_6 + param_5;
        } while (iVar3 < param_1 + -3);
      }
      for (; iVar3 < param_1; iVar3 = iVar3 + 1) {
        lVar2 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar4 + iVar3);
        *(float *)(*param_4 + lVar2 * 4) =
             (float)(int)*(float *)(*param_4 + lVar2 * 4) * param_6 + param_5;
      }
      iVar4 = iVar4 + 1;
    } while (iVar4 < param_2);
  }
  return;
}


```

## {'procedure': '0x180016f70'}
Evidence: `ev_0a01c3e8c99f3d852739ea8620d43e6b924f4a4f08623d87a2dc3822290e4f9a`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180016f70(longlong param_1,longlong param_2,longlong param_3,int param_4,int *param_5,
                  longlong param_6,longlong param_7,longlong param_8)

{
  int *piVar1;
  longlong lVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  ulonglong uVar7;
  longlong lVar8;
  longlong lVar9;
  uint uVar10;
  ulonglong uVar11;
  longlong lVar12;
  int *piVar13;
  Mat *this;
  longlong lVar14;
  int *piVar15;
  longlong lVar16;
  ulonglong uVar17;
  undefined1 auStackY_2a8 [32];
  ulonglong local_268;
  undefined4 local_238 [2];
  longlong local_230;
  undefined8 local_228;
  undefined4 local_220 [2];
  longlong local_218;
  undefined8 local_210;
  undefined4 local_208 [2];
  longlong local_200;
  undefined8 local_1f8;
  undefined4 local_1f0 [2];
  longlong local_1e8;
  undefined8 local_1e0;
  undefined4 local_1d8 [2];
  longlong local_1d0;
  undefined8 local_1c8;
  undefined4 local_1c0 [2];
  longlong local_1b8;
  undefined8 local_1b0;
  longlong local_1a8;
  undefined4 uStack_1a0;
  undefined4 uStack_19c;
  longlong local_198;
  undefined4 uStack_190;
  undefined4 uStack_18c;
  longlong local_188;
  undefined4 uStack_180;
  undefined4 uStack_17c;
  longlong local_178;
  undefined8 uStack_170;
  longlong local_168;
  undefined8 uStack_160;
  undefined8 local_158;
  longlong local_148;
  undefined8 uStack_140;
  undefined8 local_138;
  undefined8 uStack_130;
  undefined8 local_128;
  undefined8 uStack_120;
  undefined8 local_118;
  undefined8 uStack_110;
  undefined8 local_108;
  longlong lStack_100;
  undefined8 *local_f8;
  undefined8 *local_f0;
  undefined8 local_e8;
  undefined8 local_e0;
  undefined4 local_d8;
  undefined8 local_c8;
  undefined8 uStack_c0;
  undefined8 local_b8;
  undefined8 uStack_b0;
  undefined8 local_a8;
  undefined8 uStack_a0;
  undefined8 local_98;
  longlong lStack_90;
  undefined8 *local_88;
  undefined8 *local_80;
  undefined8 local_78;
  undefined8 local_70;
  undefined4 local_68;
  ulonglong local_58;
  
  local_158 = 0xfffffffffffffffe;
  local_58 = DAT_18008b080 ^ (ulonglong)auStackY_2a8;
  if (*(char *)(param_1 + 0x24) == '\0') {
    iVar3 = *(int *)(param_1 + 0x78);
    lVar16 = (longlong)iVar3;
    iVar4 = *(int *)(param_1 + 0x4dc);
    iVar5 = *(int *)(param_1 + 0x4e0);
    if (param_4 < iVar3) {
      piVar13 = (int *)(param_1 + 0x244);
      local_268 = 0;
      uVar7 = (ulonglong)((longlong)(piVar13 + *(int *)(param_1 + 0x254)) + (3 - (longlong)piVar13))
              >> 2;
      if (piVar13 + *(int *)(param_1 + 0x254) < piVar13) {
        uVar7 = 0;
      }
      if (uVar7 != 0) {
        lVar16 = (longlong)param_4 * 0x68;
        do {
          local_228 = 0;
          local_238[0] = 0x2010000;
          lVar8 = (longlong)*piVar13 * 8;
          local_230 = *(longlong *)(lVar8 + param_6) + lVar16;
          lVar12 = (longlong)*piVar13 * 0x68;
          this = (Mat *)(param_2 + lVar12);
          cv::Mat::copyTo(this,(_OutputArray *)local_238);
          local_210 = 0;
          local_220[0] = 0x2010000;
          local_218 = *(longlong *)(lVar8 + param_7) + lVar16;
          cv::Mat::copyTo(this,(_OutputArray *)local_220);
          local_1e0 = 0;
          local_1f0[0] = 0x2010000;
          local_1e8 = *(longlong *)(lVar8 + param_8) + lVar16;
          cv::Mat::copyTo((Mat *)(param_3 + lVar12),(_OutputArray *)local_1f0);
          piVar13 = piVar13 + 1;
          local_268 = local_268 + 1;
        } while (local_268 != uVar7);
      }
      param_5[param_4] = param_4;
    }
    else {
      iVar6 = *param_5;
      if (0 < (int)(iVar3 - 1U)) {
        piVar15 = param_5;
        piVar13 = param_5;
        for (uVar7 = (ulonglong)(iVar3 - 1U); piVar13 = piVar13 + 1, uVar7 != 0; uVar7 = uVar7 - 1)
        {
          *piVar15 = *piVar13;
          piVar15 = piVar15 + 1;
        }
      }
      param_5[lVar16 + -1] = iVar6;
      piVar13 = (int *)(param_1 + 0x244);
      piVar15 = piVar13 + *(int *)(param_1 + 0x254);
      if (piVar13 != piVar15) {
        lVar8 = (longlong)iVar6 * 0x68;
        do {
          lVar12 = 0;
          lVar2 = (longlong)*piVar13 * 8;
          local_1b8 = lVar8 + *(longlong *)(lVar2 + param_7);
          local_1b0 = 0;
          local_1c0[0] = 0x2010000;
          lVar14 = (longlong)*piVar13 * 0x68;
          cv::Mat::copyTo((Mat *)(param_2 + lVar14),(_OutputArray *)local_1c0);
          local_1d0 = lVar8 + *(longlong *)(lVar2 + param_8);
          local_1c8 = 0;
          local_1d8[0] = 0x2010000;
          cv::Mat::copyTo((Mat *)(param_3 + lVar14),(_OutputArray *)local_1d8);
          memset(&local_c8,0,0x68);
          local_c8 = 0x42ff0000;
          uStack_c0 = 0;
          local_b8 = 0;
          uStack_b0 = 0;
          local_a8 = 0;
          uStack_a0 = 0;
          local_98 = 0;
          lStack_90 = 0;
          local_88 = &uStack_c0;
          local_80 = &local_78;
          local_70 = 0;
          local_78 = 0;
          local_68 = 0;
          memset(&local_138,0,0x68);
          local_138 = 0x42ff0000;
          uStack_130 = 0;
          local_128 = 0;
          uStack_120 = 0;
          local_118 = 0;
          uStack_110 = 0;
          local_108 = 0;
          lStack_100 = 0;
          local_f8 = &uStack_130;
          local_f0 = &local_e8;
          local_e0 = 0;
          local_e8 = 0;
          local_d8 = 0;
          lVar14 = lVar12;
          if (0 < lVar16) {
            do {
              lVar9 = (longlong)param_5[iVar3 / 2] * 0x68 + *(longlong *)(lVar2 + param_8);
              local_198 = *(longlong *)(lVar9 + 0x10);
              uStack_190 = (undefined4)*(undefined8 *)(lVar9 + 0x50);
              uStack_170 = CONCAT44(uStack_18c,uStack_190);
              local_188 = *(longlong *)(*(longlong *)(lVar2 + param_6) + 0x10 + lVar12);
              uStack_180 = (undefined4)
                           *(undefined8 *)(*(longlong *)(lVar2 + param_6) + 0x50 + lVar12);
              uStack_140 = CONCAT44(uStack_17c,uStack_180);
              lVar9 = (longlong)param_5[lVar14] * 0x68 + *(longlong *)(lVar2 + param_7);
              local_1a8 = *(longlong *)(lVar9 + 0x10);
              uStack_1a0 = (undefined4)*(undefined8 *)(lVar9 + 0x50);
              uStack_160 = CONCAT44(uStack_19c,uStack_1a0);
              local_178 = local_198;
              local_168 = local_1a8;
              local_148 = local_188;
              FUN_180016dd0(iVar4,iVar5,&local_168,&local_148,&local_178);
              lVar14 = lVar14 + 1;
              lVar12 = lVar12 + 0x68;
            } while (lVar14 < lVar16);
            if (lStack_100 != 0) {
              LOCK();
              piVar1 = (int *)(lStack_100 + 0x14);
              iVar6 = *piVar1;
              *piVar1 = *piVar1 + -1;
              UNLOCK();
              if (iVar6 == 1) {
                cv::Mat::deallocate((Mat *)&local_138);
              }
            }
          }
          uVar7 = 0;
          lStack_100 = 0;
          local_128 = 0;
          uStack_110 = 0;
          local_118 = 0;
          uStack_120 = 0;
          uVar17 = uVar7;
          uVar11 = uVar7;
          if (0 < local_138._4_4_) {
            do {
              *(undefined4 *)(uVar17 + (longlong)local_f8) = 0;
              uVar10 = (int)uVar11 + 1;
              uVar17 = uVar17 + 4;
              uVar11 = (ulonglong)uVar10;
            } while ((int)uVar10 < local_138._4_4_);
          }
          if (local_f0 != &local_e8) {
            cv::fastFree(local_f0);
          }
          if (lStack_90 != 0) {
            LOCK();
            piVar1 = (int *)(lStack_90 + 0x14);
            iVar6 = *piVar1;
            *piVar1 = *piVar1 + -1;
            UNLOCK();
            if (iVar6 == 1) {
              cv::Mat::deallocate((Mat *)&local_c8);
            }
          }
          lStack_90 = 0;
          local_b8 = 0;
          uStack_a0 = 0;
          local_a8 = 0;
          uStack_b0 = 0;
          uVar17 = uVar7;
          if (0 < local_c8._4_4_) {
            do {
              *(undefined4 *)(uVar7 + (longlong)local_88) = 0;
              uVar10 = (int)uVar17 + 1;
              uVar7 = uVar7 + 4;
              uVar17 = (ulonglong)uVar10;
            } while ((int)uVar10 < local_c8._4_4_);
          }
          if (local_80 != &local_78) {
            cv::fastFree(local_80);
          }
          piVar13 = piVar13 + 1;
        } while (piVar13 != piVar15);
      }
    }
  }
  else {
    piVar13 = (int *)(param_1 + 0x244);
    uVar17 = 0;
    uVar7 = (ulonglong)((longlong)(piVar13 + *(int *)(param_1 + 0x254)) + (3 - (longlong)piVar13))
            >> 2;
    if (piVar13 + *(int *)(param_1 + 0x254) < piVar13) {
      uVar7 = uVar17;
    }
    if (uVar7 != 0) {
      do {
        local_1f8 = 0;
        local_208[0] = 0x2010000;
        local_200 = *(longlong *)(param_6 + (longlong)*piVar13 * 8) + (longlong)param_4 * 0x68;
        cv::Mat::copyTo((Mat *)((longlong)*piVar13 * 0x68 + param_2),(_OutputArray *)local_208);
        piVar13 = piVar13 + 1;
        uVar17 = uVar17 + 1;
      } while (uVar17 != uVar7);
    }
  }
  return;
}


```

## {'procedure': '0x18001c480'}
Evidence: `ev_0a0f49acf4d588b10168a58666899c7a3c58fdfb83a596d2533e4cb5010d65c8`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_18001c480(longlong param_1,longlong *param_2,longlong param_3,Mat *param_4,int param_5)

{
  int *piVar1;
  int iVar2;
  undefined8 uVar3;
  char *pcVar4;
  _InputOutputArray *p_Var5;
  Mat *pMVar6;
  undefined1 *puVar7;
  uint uVar8;
  ulonglong uVar9;
  ulonglong uVar10;
  ulonglong uVar11;
  undefined4 uVar12;
  undefined4 uVar13;
  undefined1 auStackY_a58 [32];
  undefined4 uVar15;
  undefined8 uVar14;
  undefined8 local_a18;
  undefined8 local_a10;
  undefined8 local_a08;
  double local_a00;
  double local_9f8;
  undefined4 local_9f0 [2];
  undefined8 *local_9e8;
  undefined8 local_9e0;
  undefined4 local_9d8 [2];
  Mat *local_9d0;
  undefined8 local_9c8;
  undefined4 local_9c0 [2];
  undefined8 *local_9b8;
  undefined8 local_9b0;
  undefined4 local_9a8 [2];
  double *local_9a0;
  undefined4 local_998;
  undefined4 local_994;
  undefined4 local_990 [2];
  Mat *local_988;
  undefined8 local_980;
  undefined4 local_978 [2];
  undefined8 *local_970;
  undefined8 local_968;
  undefined4 local_960 [2];
  undefined8 *local_958;
  undefined8 local_950;
  undefined4 local_948 [2];
  Mat *local_940;
  undefined8 local_938;
  undefined4 local_930 [2];
  double *local_928;
  undefined4 local_920;
  undefined4 local_91c;
  undefined4 local_918 [2];
  Mat *local_910;
  undefined8 local_908;
  undefined4 local_900 [2];
  Mat *local_8f8;
  undefined8 local_8f0;
  undefined4 local_8e8 [2];
  undefined8 *local_8e0;
  undefined8 local_8d8;
  undefined4 local_8d0 [2];
  Mat *local_8c8;
  undefined8 local_8c0;
  undefined4 local_8b8 [2];
  double *local_8b0;
  undefined4 local_8a8;
  undefined4 local_8a4;
  undefined4 local_8a0 [2];
  double *local_898;
  undefined4 local_890;
  undefined4 local_88c;
  Mat local_888 [4];
  int local_884;
  undefined8 local_878;
  undefined8 local_870;
  undefined8 local_868;
  undefined8 local_860;
  longlong local_850;
  longlong local_848;
  undefined1 *local_840;
  undefined1 local_838 [16];
  Mat local_828 [4];
  int local_824;
  undefined8 local_818;
  undefined8 local_810;
  undefined8 local_808;
  undefined8 local_800;
  longlong local_7f0;
  longlong local_7e8;
  undefined1 *local_7e0;
  undefined1 local_7d8 [16];
  undefined8 local_7c8;
  double local_7c0;
  double local_7b8 [2];
  undefined8 local_7a8;
  undefined8 uStack_7a0;
  undefined8 local_798;
  undefined8 uStack_790;
  undefined8 local_788;
  undefined8 uStack_780;
  undefined8 local_778;
  longlong lStack_770;
  undefined8 *local_768;
  undefined8 *local_760;
  undefined8 local_758;
  undefined8 local_750;
  undefined4 local_748;
  undefined8 local_738;
  undefined8 uStack_730;
  undefined8 local_728;
  undefined8 uStack_720;
  undefined8 local_718;
  undefined8 uStack_710;
  undefined8 local_708;
  longlong lStack_700;
  undefined8 *local_6f8;
  undefined8 *local_6f0;
  undefined8 local_6e8;
  undefined8 local_6e0;
  undefined4 local_6d8;
  longlong *local_6c8;
  undefined8 local_6c0;
  longlong local_6b8;
  longlong local_6b0;
  undefined8 local_6a8;
  Mat local_698 [312];
  undefined1 local_560 [312];
  undefined1 local_428 [312];
  undefined1 local_2f0 [632];
  ulonglong local_78;
  
  local_7c8 = 0xfffffffffffffffe;
  local_78 = DAT_18008b080 ^ (ulonglong)auStackY_a58;
  uVar3 = *(undefined8 *)(param_1 + 0xb0);
  if (param_5 == 0) {
    uVar8 = *(uint *)(param_1 + 0xcc);
    uVar12 = (undefined4)*(undefined8 *)(param_1 + 0xd0);
    uVar13 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0xd0) >> 0x20);
  }
  else {
    if (param_5 != 1) {
      local_a10 = 0;
      local_a08 = 0;
      pcVar4 = cv::String::allocate((String *)&local_a10,0x31);
      builtin_strncpy(pcVar4,"nDepthType == DEPTH_3D || nDepthType == DEPTH_EFI",0x31);
      cv::error(-0xd7,(String *)&local_a10,"sg::cpu::FastBilateralFilterCPU","SweepGuideCpu.cpp",
                0xe43);
      cv::String::deallocate((String *)&local_a10);
    }
    uVar8 = *(uint *)(param_1 + 0xec);
    uVar12 = (undefined4)*(undefined8 *)(param_1 + 0xf0);
    uVar13 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0xf0) >> 0x20);
  }
  if (0 < (int)uVar8) {
    FUN_18001b920((longlong)local_698);
    memset(&local_7a8,0,0x68);
    local_7a8 = 0x42ff0000;
    uStack_7a0 = 0;
    local_798 = 0;
    uStack_790 = 0;
    local_788 = 0;
    uStack_780 = 0;
    local_778 = 0;
    lStack_770 = 0;
    local_768 = &uStack_7a0;
    local_760 = &local_758;
    local_750 = 0;
    local_758 = 0;
    local_748 = 0;
    memset(&local_738,0,0x68);
    local_738 = 0x42ff0000;
    uStack_730 = 0;
    local_728 = 0;
    uStack_720 = 0;
    local_718 = 0;
    uStack_710 = 0;
    local_708 = 0;
    lStack_700 = 0;
    local_6f8 = &uStack_730;
    local_6f0 = &local_6e8;
    local_6e0 = 0;
    local_6e8 = 0;
    local_6d8 = 0;
    local_6a8 = 0;
    local_6c0 = 0x20;
    local_6b8 = 0;
    local_6b0 = param_2[1] + *param_2;
    local_6c8 = param_2;
    FUN_180026760((longlong *)&local_6c8,param_1,uVar8,local_698,&local_7a8,(Mat *)&local_738);
    local_980 = 0;
    local_990[0] = 0x1010000;
    local_988 = param_4;
    p_Var5 = cv::noArray();
    uVar15 = 0;
    cv::minMaxLoc((_InputArray *)local_990,&local_7c0,local_7b8,(Point_<int> *)0x0,
                  (Point_<int> *)0x0,(_InputArray *)p_Var5);
    local_7c0 = (double)**(float **)(param_3 + 0x10);
    if (local_7b8[0] - local_7c0 == 0.0) {
      local_6c8[1] = local_6c8[1] - local_6b8;
      if (lStack_700 != 0) {
        LOCK();
        piVar1 = (int *)(lStack_700 + 0x14);
        iVar2 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar2 == 1) {
          cv::Mat::deallocate((Mat *)&local_738);
        }
      }
      uVar9 = 0;
      lStack_700 = 0;
      local_728 = 0;
      uStack_710 = 0;
      local_718 = 0;
      uStack_720 = 0;
      uVar11 = uVar9;
      uVar10 = uVar9;
      if (0 < local_738._4_4_) {
        do {
          *(undefined4 *)(uVar10 + (longlong)local_6f8) = 0;
          uVar8 = (int)uVar11 + 1;
          uVar11 = (ulonglong)uVar8;
          uVar10 = uVar10 + 4;
        } while ((int)uVar8 < local_738._4_4_);
      }
      if (local_6f0 != &local_6e8) {
        cv::fastFree(local_6f0);
      }
      if (lStack_770 != 0) {
        LOCK();
        piVar1 = (int *)(lStack_770 + 0x14);
        iVar2 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar2 == 1) {
          cv::Mat::deallocate((Mat *)&local_7a8);
        }
      }
      lStack_770 = 0;
      local_798 = 0;
      uStack_780 = 0;
      local_788 = 0;
      uStack_790 = 0;
      uVar11 = uVar9;
      if (0 < local_7a8._4_4_) {
        do {
          *(undefined4 *)(uVar11 + (longlong)local_768) = 0;
          uVar8 = (int)uVar9 + 1;
          uVar9 = (ulonglong)uVar8;
          uVar11 = uVar11 + 4;
        } while ((int)uVar8 < local_7a8._4_4_);
      }
      if (local_760 != &local_758) {
        cv::fastFree(local_760);
      }
      _eh_vector_destructor_iterator_(local_2f0,0x68,6,FUN_180001750);
      _eh_vector_destructor_iterator_(local_428,0x68,3,FUN_180001750);
      _eh_vector_destructor_iterator_(local_560,0x68,3,FUN_180001750);
      _eh_vector_destructor_iterator_(local_698,0x68,3,FUN_180001750);
      return 0x18;
    }
    local_968 = 0;
    local_978[0] = 0x2010000;
    local_970 = &local_7a8;
    local_8b8[0] = 0xc1020006;
    local_8b0 = &local_7c0;
    local_8a8 = 1;
    local_8a4 = 1;
    local_938 = 0;
    local_948[0] = 0x1010000;
    local_940 = param_4;
    p_Var5 = cv::noArray();
    cv::subtract((_InputArray *)local_948,(_InputArray *)local_8b8,(_OutputArray *)local_978,
                 (_InputArray *)p_Var5,-1);
    local_8d8 = 0;
    local_8e8[0] = 0x2010000;
    local_8e0 = &local_7a8;
    local_a00 = 1.0 / (local_7b8[0] - local_7c0);
    local_9a8[0] = 0xc1020006;
    local_9a0 = &local_a00;
    local_998 = 1;
    local_994 = 1;
    local_9e0 = 0;
    local_9f0[0] = 0x1010000;
    local_9e8 = &local_7a8;
    uVar14 = CONCAT44(uVar15,0xffffffff);
    cv::multiply((_InputArray *)local_9f0,(_InputArray *)local_9a8,(_OutputArray *)local_8e8,1.0,-1)
    ;
    FUN_180019240(param_1,param_2,(Mat *)&local_7a8,(longlong)local_698,uVar14,uVar8,
                  (double)CONCAT44(uVar13,uVar12));
    local_9b0 = 0;
    local_9c0[0] = 0x2010000;
    local_9b8 = &local_738;
    local_950 = 0;
    local_960[0] = 0x1010000;
    local_958 = &local_7a8;
    local_a18 = 0;
    cv::resize(local_960,local_9c0,&local_a18,uVar3);
    FUN_18001a680(param_1,param_2,(Mat *)&local_738,(longlong)local_698,(longlong)param_4,uVar8,1);
    uVar11 = 0;
    local_9c8 = 0;
    local_9d8[0] = 0x2010000;
    local_9f8 = local_7b8[0] - local_7c0;
    local_930[0] = 0xc1020006;
    local_928 = &local_9f8;
    local_920 = 1;
    local_91c = 1;
    local_8f0 = 0;
    local_900[0] = 0x1010000;
    local_9d0 = param_4;
    local_8f8 = param_4;
    cv::multiply((_InputArray *)local_900,(_InputArray *)local_930,(_OutputArray *)local_9d8,1.0,-1)
    ;
    local_8c0 = 0;
    local_8d0[0] = 0x2010000;
    local_8a0[0] = 0xc1020006;
    local_898 = &local_7c0;
    local_890 = 1;
    local_88c = 1;
    local_908 = 0;
    local_918[0] = 0x1010000;
    local_910 = param_4;
    local_8c8 = param_4;
    p_Var5 = cv::noArray();
    cv::add((_InputArray *)local_918,(_InputArray *)local_8a0,(_OutputArray *)local_8d0,
            (_InputArray *)p_Var5,-1);
    local_6c8[1] = local_6c8[1] - local_6b8;
    if (lStack_700 != 0) {
      LOCK();
      piVar1 = (int *)(lStack_700 + 0x14);
      iVar2 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar2 == 1) {
        cv::Mat::deallocate((Mat *)&local_738);
      }
    }
    lStack_700 = 0;
    local_728 = 0;
    uStack_710 = 0;
    local_718 = 0;
    uStack_720 = 0;
    uVar10 = uVar11;
    uVar9 = uVar11;
    if (0 < local_738._4_4_) {
      do {
        *(undefined4 *)(uVar10 + (longlong)local_6f8) = 0;
        uVar8 = (int)uVar9 + 1;
        uVar10 = uVar10 + 4;
        uVar9 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_738._4_4_);
    }
    if (local_6f0 != &local_6e8) {
      cv::fastFree(local_6f0);
    }
    if (lStack_770 != 0) {
      LOCK();
      piVar1 = (int *)(lStack_770 + 0x14);
      iVar2 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar2 == 1) {
        cv::Mat::deallocate((Mat *)&local_7a8);
      }
    }
    lStack_770 = 0;
    local_798 = 0;
    uStack_780 = 0;
    local_788 = 0;
    uStack_790 = 0;
    uVar10 = uVar11;
    if (0 < local_7a8._4_4_) {
      do {
        *(undefined4 *)(uVar11 + (longlong)local_768) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar11 = uVar11 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_7a8._4_4_);
    }
    if (local_760 != &local_758) {
      cv::fastFree(local_760);
    }
    _eh_vector_destructor_iterator_(local_2f0,0x68,6,FUN_180001750);
    _eh_vector_destructor_iterator_(local_428,0x68,3,FUN_180001750);
    _eh_vector_destructor_iterator_(local_560,0x68,3,FUN_180001750);
    _eh_vector_destructor_iterator_(local_698,0x68,3,FUN_180001750);
  }
  uVar11 = 0;
  if (*(char *)(param_1 + 0x11c) == '\x01') {
    if (param_5 == 0) {
      pMVar6 = FUN_1800013f0(local_828,param_4);
      FUN_180004b70(param_1,param_3,pMVar6,7);
      if (local_7f0 != 0) {
        LOCK();
        piVar1 = (int *)(local_7f0 + 0x14);
        iVar2 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar2 == 1) {
          cv::Mat::deallocate(local_828);
        }
      }
      local_7f0 = 0;
      local_818 = 0;
      local_800 = 0;
      local_808 = 0;
      local_810 = 0;
      uVar10 = uVar11;
      if (0 < local_824) {
        do {
          *(undefined4 *)(uVar10 + local_7e8) = 0;
          uVar8 = (int)uVar11 + 1;
          uVar11 = (ulonglong)uVar8;
          uVar10 = uVar10 + 4;
        } while ((int)uVar8 < local_824);
      }
      puVar7 = local_7d8;
    }
    else {
      if (param_5 != 1) {
        return 0;
      }
      pMVar6 = FUN_1800013f0(local_888,param_4);
      FUN_180004b70(param_1,param_3,pMVar6,0xc);
      if (local_850 != 0) {
        LOCK();
        piVar1 = (int *)(local_850 + 0x14);
        iVar2 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar2 == 1) {
          cv::Mat::deallocate(local_888);
        }
      }
      local_850 = 0;
      local_878 = 0;
      local_860 = 0;
      local_868 = 0;
      local_870 = 0;
      uVar10 = uVar11;
      if (0 < local_884) {
        do {
          *(undefined4 *)(uVar10 + local_848) = 0;
          uVar8 = (int)uVar11 + 1;
          uVar11 = (ulonglong)uVar8;
          uVar10 = uVar10 + 4;
        } while ((int)uVar8 < local_884);
      }
      puVar7 = local_838;
      local_7e0 = local_840;
    }
    if (local_7e0 != puVar7) {
      cv::fastFree(local_7e0);
    }
  }
  return 0;
}


```

## {'procedure': 'EFI_InputImage'}
Evidence: `ev_1a5e701b88ea0a6bcbaa53ee714f107ef37e4511c61a60ee8da335709c7904ed`

```c

undefined8 EFI_InputImage(undefined8 *param_1,longlong param_2)

{
  undefined8 uVar1;
  
                    /* 0x65f0  6  EFI_InputImage */
  if ((param_1 == (undefined8 *)0x0) || (param_2 == 0)) {
    uVar1 = 0x11;
  }
  else {
    uVar1 = (**(code **)*param_1)();
  }
  return uVar1;
}


```

## {'procedure': '0x180023a90'}
Evidence: `ev_21faf21d88f3bf576e8cee16a896f3a85ae7e0b1d3196392de22edb05c3432b4`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_180023a90(undefined8 *param_1,undefined8 *param_2,ulonglong param_3,undefined8 param_4,
             undefined8 param_5,undefined8 *param_6)

{
  int iVar1;
  int iVar2;
  undefined8 uVar3;
  int *piVar4;
  int iVar5;
  int *piVar6;
  undefined8 **ppuVar7;
  undefined8 *puVar8;
  longlong lVar9;
  ulonglong uVar10;
  longlong *plVar11;
  undefined8 *puVar12;
  ulonglong uVar13;
  ulonglong uVar14;
  undefined1 auStackY_348 [32];
  undefined4 in_stack_fffffffffffffcdc;
  undefined8 *local_308;
  undefined8 uStack_300;
  ulonglong local_2f8;
  undefined8 *local_2e8;
  undefined8 *local_2e0;
  undefined8 *local_2d8;
  ulonglong uStack_2d0;
  ulonglong local_2c8;
  undefined8 *local_2b8;
  longlong local_2b0;
  undefined8 *local_2a8;
  ulonglong uStack_2a0;
  ulonglong local_298;
  undefined8 **local_290;
  undefined8 local_288;
  ulonglong local_280;
  undefined8 *local_278;
  undefined8 local_270;
  int local_268;
  int iStack_264;
  int iStack_254;
  int iStack_250;
  int iStack_24c;
  int local_248;
  ulonglong local_38;
  
  local_38 = DAT_18008b080 ^ (ulonglong)auStackY_348;
  lVar9 = 4;
  piVar4 = &local_268;
  do {
    puVar12 = param_1;
    piVar6 = piVar4;
    uVar3 = puVar12[1];
    *(undefined8 *)piVar6 = *puVar12;
    *(undefined8 *)(piVar6 + 2) = uVar3;
    uVar3 = puVar12[3];
    *(undefined8 *)(piVar6 + 4) = puVar12[2];
    *(undefined8 *)(piVar6 + 6) = uVar3;
    uVar3 = puVar12[5];
    *(undefined8 *)(piVar6 + 8) = puVar12[4];
    *(undefined8 *)(piVar6 + 10) = uVar3;
    uVar3 = puVar12[7];
    *(undefined8 *)(piVar6 + 0xc) = puVar12[6];
    *(undefined8 *)(piVar6 + 0xe) = uVar3;
    uVar3 = puVar12[9];
    *(undefined8 *)(piVar6 + 0x10) = puVar12[8];
    *(undefined8 *)(piVar6 + 0x12) = uVar3;
    uVar3 = puVar12[0xb];
    *(undefined8 *)(piVar6 + 0x14) = puVar12[10];
    *(undefined8 *)(piVar6 + 0x16) = uVar3;
    uVar3 = puVar12[0xd];
    *(undefined8 *)(piVar6 + 0x18) = puVar12[0xc];
    *(undefined8 *)(piVar6 + 0x1a) = uVar3;
    uVar3 = puVar12[0xf];
    *(undefined8 *)(piVar6 + 0x1c) = puVar12[0xe];
    *(undefined8 *)(piVar6 + 0x1e) = uVar3;
    lVar9 = lVar9 + -1;
    piVar4 = piVar6 + 0x20;
    param_1 = puVar12 + 0x10;
  } while (lVar9 != 0);
  uVar3 = puVar12[0x11];
  *(undefined8 *)(piVar6 + 0x20) = puVar12[0x10];
  *(undefined8 *)(piVar6 + 0x22) = uVar3;
  iVar5 = *(int *)((longlong)puVar12 + 0x94);
  iVar1 = *(int *)(puVar12 + 0x13);
  iVar2 = *(int *)((longlong)puVar12 + 0x9c);
  piVar6[0x24] = *(int *)(puVar12 + 0x12);
  piVar6[0x25] = iVar5;
  piVar6[0x26] = iVar1;
  piVar6[0x27] = iVar2;
  iVar5 = *(int *)((longlong)puVar12 + 0xa4);
  iVar1 = *(int *)(puVar12 + 0x15);
  iVar2 = *(int *)((longlong)puVar12 + 0xac);
  piVar6[0x28] = *(int *)(puVar12 + 0x14);
  piVar6[0x29] = iVar5;
  piVar6[0x2a] = iVar1;
  piVar6[0x2b] = iVar2;
  iVar5 = omp_get_max_threads();
  uVar14 = (ulonglong)iVar5;
  puVar8 = (undefined8 *)0x0;
  local_248 = 0;
  local_2d8 = (undefined8 *)0x0;
  uStack_2d0 = 0;
  local_2b8 = (undefined8 *)0x0;
  local_2b0 = 0;
  local_2e8 = (undefined8 *)0x0;
  local_2e0 = (undefined8 *)0x0;
  FUN_18001d3c0(&local_268,(ulonglong *)&local_2d8,(longlong *)&local_2b8,(ulonglong *)&local_2e8,
                iVar5);
  puVar12 = local_2e0;
  uVar13 = (longlong)local_2e0 + uStack_2d0 + local_2b0;
  if (uVar13 <= param_3) {
    local_2f8 = param_3 - uVar13;
    local_2c8 = uVar13 - (longlong)local_2e0;
    local_2e0 = (undefined8 *)0x0;
    iStack_254 = iStack_254 * local_268;
    if (iStack_250 != 8) {
      iStack_254 = iStack_254 * 2;
    }
    local_2e8 = (undefined8 *)((longlong)(iStack_264 * iStack_254) + 0x1fU & 0xffffffffffffffe0);
    ppuVar7 = &local_2e8;
    if (local_2e8 == (undefined8 *)0x0) {
      ppuVar7 = &local_2e0;
    }
    uVar10 = local_2f8 % (ulonglong)*ppuVar7;
    local_248 = (int)(local_2f8 / (ulonglong)*ppuVar7);
    if (iStack_24c < local_248) {
      local_248 = iStack_24c;
    }
    if (0 < local_248) {
      local_308 = (undefined8 *)(uVar13 + (longlong)param_2);
      uStack_2d0 = 0;
      local_270 = 0;
      local_290 = &local_2a8;
      local_288 = 0x20;
      local_280 = 0xd60;
      uStack_2a0 = 0xd60;
      local_2d8 = local_308;
      local_298 = local_2c8;
      local_2a8 = param_2;
      local_278 = param_2;
      if (local_2c8 < 0xd60) {
        local_2c8 = local_2f8;
                    /* WARNING: Subroutine does not return */
        FUN_180001b90("Allocating memory exceeding the limit ...",uVar10,local_298,uVar13);
      }
      uStack_2a0 = (uVar14 * 0x18 + 0x1f & 0xffffffffffffffe0) + 0xd60;
      local_2b8 = param_2;
      if (uStack_2a0 <= local_2c8) {
        local_280 = 0;
        if (0 < (longlong)uVar14) {
          iVar1 = (int)((ulonglong)puVar12 / uVar14);
          plVar11 = param_2 + 0x1ac;
          puVar12 = puVar8;
          do {
            if (plVar11 != (longlong *)0x0) {
              *plVar11 = (longlong)param_2 + (longlong)(int)puVar12 + local_2c8 + 0x40;
              plVar11[1] = 0;
              plVar11[2] = (longlong)(iVar1 + -0x40);
            }
            puVar12 = (undefined8 *)(ulonglong)(uint)((int)puVar12 + iVar1);
            plVar11 = plVar11 + 3;
            uVar14 = uVar14 - 1;
          } while (uVar14 != 0);
        }
        if (param_2 != (undefined8 *)0x0) {
          uStack_300 = 0;
          local_2e8 = param_2;
          local_2d8 = param_2;
          uStack_2d0 = uStack_2a0;
          puVar8 = FUN_18001df60(param_2,(undefined8 *)&local_268,(longlong *)&local_2d8,
                                 (char *)&local_308,CONCAT44(in_stack_fffffffffffffcdc,iVar5),
                                 param_2 + 0x1ac);
        }
        *param_6 = puVar8;
        return 0;
      }
      local_2c8 = local_2f8;
      local_280 = uStack_2a0;
                    /* WARNING: Subroutine does not return */
      FUN_180001b90("Allocating memory exceeding the limit ...",uVar10,local_298,uVar14);
    }
  }
  return 0x13;
}


```

## {'procedure': '0x18001e6a0'}
Evidence: `ev_2b287a6e081d1cf79ebf3e7440c135f9c22d1c37813f615873c3c01194805dbd`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

ulonglong FUN_18001e6a0(longlong param_1,longlong param_2,longlong param_3,undefined *param_4,
                       undefined8 param_5)

{
  uint *puVar1;
  int *piVar2;
  longlong *plVar3;
  uint uVar4;
  code *pcVar5;
  int iVar6;
  uint uVar7;
  _InputOutputArray *p_Var8;
  undefined8 uVar9;
  uint uVar10;
  ulonglong uVar11;
  ulonglong uVar12;
  ulonglong uVar13;
  ulonglong local_res8;
  undefined1 auStackY_888 [32];
  code *local_848;
  undefined8 local_840;
  longlong local_838;
  undefined4 uStack_830;
  undefined4 uStack_82c;
  longlong local_828;
  undefined4 uStack_820;
  undefined4 uStack_81c;
  undefined4 local_818 [2];
  undefined8 *local_810;
  undefined8 local_808;
  undefined4 local_800 [2];
  Mat *local_7f8;
  undefined8 local_7f0;
  undefined4 local_7e8 [2];
  double *local_7e0;
  undefined4 local_7d8;
  undefined4 local_7d4;
  undefined4 local_7d0 [2];
  Mat *local_7c8;
  undefined8 local_7c0;
  undefined4 local_7b8 [2];
  Mat *local_7b0;
  undefined8 local_7a8;
  undefined4 local_7a0 [2];
  undefined8 *local_798;
  undefined8 local_790;
  double local_788;
  double dStack_780;
  double local_778;
  double dStack_770;
  longlong local_768;
  undefined8 uStack_760;
  undefined8 local_758;
  longlong local_748;
  undefined8 uStack_740;
  longlong local_738;
  undefined8 uStack_730;
  undefined8 local_728;
  undefined8 uStack_720;
  undefined4 local_718;
  int local_714;
  undefined4 local_710;
  undefined4 local_70c;
  undefined8 local_708;
  undefined8 local_700;
  undefined8 local_6f8;
  undefined8 local_6f0;
  undefined8 local_6e8;
  longlong local_6e0;
  undefined4 *local_6d8;
  undefined8 *local_6d0;
  undefined8 local_6c8;
  undefined8 local_6c0;
  undefined4 local_6b8;
  undefined4 local_6a8;
  int local_6a4;
  undefined4 local_6a0;
  undefined4 local_69c;
  undefined8 local_698;
  undefined8 local_690;
  undefined8 local_688;
  undefined8 local_680;
  undefined8 local_678;
  longlong local_670;
  undefined4 *local_668;
  undefined8 *local_660;
  undefined8 local_658;
  undefined8 local_650;
  undefined4 local_648;
  undefined8 local_638;
  undefined8 uStack_630;
  undefined8 local_628;
  undefined8 uStack_620;
  undefined8 local_618;
  undefined8 uStack_610;
  undefined8 local_608;
  longlong lStack_600;
  undefined8 *local_5f8;
  undefined8 *local_5f0;
  undefined8 local_5e8;
  undefined8 local_5e0;
  undefined4 local_5d8;
  longlong *local_5c8;
  undefined8 local_5c0;
  longlong local_5b8;
  longlong local_5b0;
  undefined8 local_5a8;
  longlong *local_5a0;
  undefined8 local_598;
  longlong local_590;
  longlong local_588;
  undefined8 local_580;
  undefined8 local_578;
  undefined8 uStack_570;
  longlong local_568;
  undefined8 uStack_560;
  undefined8 local_558;
  undefined8 uStack_550;
  undefined8 local_548;
  undefined8 uStack_540;
  undefined8 *local_538;
  undefined8 *local_530;
  undefined8 local_528;
  undefined8 local_520;
  undefined4 local_518;
  undefined4 local_508;
  int local_504;
  undefined4 local_500;
  undefined4 local_4fc;
  undefined8 local_4f8;
  undefined8 local_4f0;
  undefined8 local_4e8;
  undefined8 local_4e0;
  undefined8 local_4d8;
  longlong local_4d0;
  undefined8 *local_4c0;
  undefined8 local_4b8 [2];
  undefined4 local_4a8;
  undefined8 local_498;
  undefined8 uStack_490;
  longlong local_488;
  undefined8 uStack_480;
  undefined8 local_478;
  undefined8 uStack_470;
  undefined8 local_468;
  undefined8 uStack_460;
  undefined8 *local_458;
  undefined8 *local_450;
  undefined8 local_448;
  undefined8 local_440;
  undefined4 local_438;
  undefined8 local_428;
  undefined8 uStack_420;
  undefined8 local_418;
  undefined8 uStack_410;
  undefined8 local_408;
  undefined8 uStack_400;
  undefined8 local_3f8;
  undefined8 uStack_3f0;
  undefined8 *local_3e8;
  undefined8 *local_3e0;
  undefined8 local_3d8;
  undefined8 local_3d0;
  undefined4 local_3c8;
  undefined8 local_3b8;
  undefined8 uStack_3b0;
  undefined8 local_3a8;
  undefined8 uStack_3a0;
  undefined8 local_398;
  undefined8 uStack_390;
  undefined8 local_388;
  undefined8 uStack_380;
  undefined8 *local_378;
  undefined8 *local_370;
  undefined8 local_368;
  undefined8 local_360;
  undefined4 local_358;
  undefined8 local_348;
  undefined8 uStack_340;
  undefined8 local_338;
  undefined8 uStack_330;
  undefined8 local_328;
  undefined8 uStack_320;
  undefined8 local_318;
  undefined8 uStack_310;
  undefined8 *local_308;
  undefined8 *local_300;
  undefined8 local_2f8;
  undefined8 local_2f0;
  undefined4 local_2e8;
  Mat local_2d8 [72];
  undefined1 *local_290;
  undefined1 local_288 [16];
  undefined4 local_278;
  Mat local_268 [72];
  undefined1 *local_220;
  undefined1 local_218 [16];
  undefined4 local_208;
  Mat local_1f8 [72];
  undefined1 *local_1b0;
  undefined1 local_1a8 [16];
  undefined4 local_198;
  Mat local_188 [72];
  undefined1 *local_140;
  undefined1 local_138 [16];
  undefined4 local_128;
  Mat local_118 [72];
  undefined1 *local_d0;
  undefined1 local_c8 [16];
  Mat local_b8 [72];
  undefined1 *local_70;
  undefined1 local_68 [16];
  ulonglong local_58;
  
  local_758 = 0xfffffffffffffffe;
  local_58 = DAT_18008b080 ^ (ulonglong)auStackY_888;
  local_838 = param_2;
  local_828 = param_3;
  if (DAT_18008c518 != '\0') {
    if (DAT_18008c538 == (void *)0x0) {
      DAT_18008c538 = malloc(0x280000);
    }
    DAT_18008c530 = 0;
    _DAT_18008c520 = cv::getTickCount();
    _DAT_18008c528 = _DAT_18008c520;
    FUN_180001d80(&DAT_18007b328,0,0,0,0);
  }
  puVar1 = (uint *)(param_1 + 8);
  piVar2 = (int *)(param_1 + 0x840);
  if ((*piVar2 != 0) &&
     ((local_840 = param_5, local_848 = (code *)param_4, param_4 == (undefined *)0x0 ||
      (iVar6 = (*(code *)param_4)(param_5), iVar6 == 0)))) {
    uVar10 = *puVar1;
    uVar4 = *(uint *)(param_1 + 0xc);
    memset(local_b8,0,0x60);
    FUN_1800014c0((uint *)local_b8,uVar4,uVar10,5,local_828,(longlong)*(int *)(param_1 + 0x18));
    memset(local_118,0,0x60);
    if (*(int *)(param_1 + 0x1c) == 1) {
      uVar13 = 2;
      uVar7 = 0;
    }
    else {
      uVar7 = 0x10;
      uVar13 = 0x12;
    }
    if (*(int *)(param_1 + 0x20) == 8) {
      uVar13 = (ulonglong)uVar7;
    }
    FUN_1800014c0((uint *)local_118,uVar4,uVar10,(uint)uVar13,local_838,
                  (longlong)*(int *)(param_1 + 0x14));
    plVar3 = (longlong *)(param_1 + 0x808);
    FUN_180002e50((longlong)puVar1,plVar3,piVar2,uVar13);
    if (*(int *)(param_1 + 0x844) == 1) {
      local_7c0 = 0;
      local_7d0[0] = 0x2010000;
      local_7c8 = local_118;
      cv::Mat::copyTo((Mat *)(param_1 + 0x878),(_OutputArray *)local_7d0);
      local_788 = (double)**(float **)(param_1 + 0x848);
      local_7e8[0] = 0xc1020006;
      local_7d8 = 1;
      local_7d4 = 4;
      local_7e0 = &local_788;
      dStack_780 = local_788;
      local_778 = local_788;
      dStack_770 = local_788;
      p_Var8 = cv::noArray();
      cv::Mat::setTo(local_b8,(_InputArray *)local_7e8,(_InputArray *)p_Var8);
      if (param_4 != (undefined *)0x0) {
        (*(code *)param_4)(param_5,0);
      }
    }
    else {
      *(undefined8 *)(param_1 + 0x810) = *(undefined8 *)(param_1 + 0xd40);
      memset(local_1f8,0,0x68);
      FUN_1800013f0(local_1f8,(Mat *)(param_1 + 0x8e0));
      local_198 = *(undefined4 *)(param_1 + 0x940);
      memset(local_188,0,0x68);
      FUN_1800013f0(local_188,(Mat *)(param_1 + 0x948));
      local_128 = *(undefined4 *)(param_1 + 0x9a8);
      memset(local_268,0,0x68);
      FUN_1800013f0(local_268,(Mat *)(param_1 + 0x9b0));
      local_208 = *(undefined4 *)(param_1 + 0xa10);
      memset(local_2d8,0,0x68);
      FUN_1800013f0(local_2d8,(Mat *)(param_1 + 0xa18));
      local_278 = *(undefined4 *)(param_1 + 0xa78);
      memset(&local_508,0,0x68);
      FUN_1800013f0((Mat *)&local_508,(Mat *)(param_1 + 0xa80));
      local_4a8 = *(undefined4 *)(param_1 + 0xae0);
      FUN_180020450(param_1,local_1f8,local_188,local_268,local_2d8,(Mat *)&local_508);
      FUN_180001d80("& GetDepthAndFastEFI",0,0,0,0);
      if ((param_4 == (undefined *)0x0) || (iVar6 = (*(code *)param_4)(param_5), iVar6 == 0)) {
        memset(&local_638,0,0x68);
        local_638 = 0x42ff0000;
        uStack_630 = 0;
        local_628 = 0;
        uStack_620 = 0;
        local_618 = 0;
        uStack_610 = 0;
        local_608 = 0;
        lStack_600 = 0;
        local_5f8 = &uStack_630;
        local_5f0 = &local_5e8;
        uVar13 = 0;
        local_5e0 = 0;
        local_5e8 = 0;
        local_5d8 = 0;
        memset(&local_578,0,0x68);
        local_578 = 0x42ff0000;
        uStack_570 = 0;
        local_568 = 0;
        uStack_560 = 0;
        local_558 = 0;
        uStack_550 = 0;
        local_548 = 0;
        uStack_540 = 0;
        local_538 = &uStack_570;
        local_530 = &local_528;
        local_520 = 0;
        local_528 = 0;
        local_518 = 0;
        memset(&local_3b8,0,0x68);
        local_3b8 = 0x42ff0000;
        uStack_3b0 = 0;
        local_3a8 = 0;
        uStack_3a0 = 0;
        local_398 = 0;
        uStack_390 = 0;
        local_388 = 0;
        uStack_380 = 0;
        local_378 = &uStack_3b0;
        local_370 = &local_368;
        local_360 = 0;
        local_368 = 0;
        local_358 = 0;
        memset(&local_498,0,0x68);
        local_498 = 0x42ff0000;
        uStack_490 = 0;
        local_488 = 0;
        uStack_480 = 0;
        local_478 = 0;
        uStack_470 = 0;
        local_468 = 0;
        uStack_460 = 0;
        local_458 = &uStack_490;
        local_450 = &local_448;
        local_440 = 0;
        local_448 = 0;
        local_438 = 0;
        *(undefined8 *)(param_1 + 0x810) = *(undefined8 *)(param_1 + 0xd48);
        local_5a8 = 0;
        local_5c0 = 0x20;
        local_5b8 = 0;
        local_5b0 = *(longlong *)(param_1 + 0x810) + *plVar3;
        local_5c8 = plVar3;
        FUN_18002a880((longlong *)&local_5c8,(longlong)puVar1,(Mat *)&local_638,(Mat *)&local_578,
                      (Mat *)&local_3b8,(Mat *)&local_498);
        FUN_180003550((longlong)puVar1,plVar3,local_268,&local_508,(longlong)local_2d8,
                      (Mat *)&local_638,3);
        FUN_180001d80("& BlendEFI",0,0,0,0);
        local_808 = 0;
        local_818[0] = 0x2010000;
        local_810 = &local_578;
        cv::Mat::copyTo(local_1f8,(_OutputArray *)local_818);
        FUN_180012b70((longlong)puVar1,plVar3,(longlong)piVar2,(Mat *)&local_578,(longlong)local_188
                      ,0);
        FUN_180001d80("& InterpTextureLess",0,0,0,0);
        uVar9 = FUN_18001aa20((longlong)puVar1,plVar3,(longlong)piVar2,&local_848,(Mat *)&local_578,
                              (longlong)&local_638,0);
        if ((char)uVar9 == '\0') {
          uVar9 = FUN_18001c480((longlong)puVar1,plVar3,(longlong)piVar2,(Mat *)&local_578,0);
          FUN_180001d80("& FastBilateralFilter",0,0,0,0);
          if ((int)uVar9 == 0) {
            local_790 = 0;
            local_7a0[0] = 0x2010000;
            local_798 = &local_3b8;
            cv::Mat::copyTo(local_1f8,(_OutputArray *)local_7a0);
            FUN_180012b70((longlong)puVar1,plVar3,(longlong)piVar2,(Mat *)&local_3b8,
                          (longlong)local_188,1);
            FUN_180001d80("& InterpTextureLess",0,0,0,0);
            uVar9 = FUN_18001aa20((longlong)puVar1,plVar3,(longlong)piVar2,&local_848,
                                  (Mat *)&local_3b8,(longlong)&local_638,1);
            if ((char)uVar9 == '\0') {
              uVar9 = FUN_18001c480((longlong)puVar1,plVar3,(longlong)piVar2,(Mat *)&local_3b8,1);
              FUN_180001d80("& FastBilateralFilter",0,0,0,0);
              if ((int)uVar9 == 0) {
                memset(&local_348,0,0x68);
                local_348 = 0x42ff0000;
                uStack_340 = 0;
                local_338 = 0;
                uStack_330 = 0;
                local_328 = 0;
                uStack_320 = 0;
                local_318 = 0;
                uStack_310 = 0;
                local_308 = &uStack_340;
                local_300 = &local_2f8;
                local_2f0 = 0;
                local_2f8 = 0;
                local_2e8 = 0;
                memset(&local_428,0,0x68);
                local_428 = 0x42ff0000;
                uStack_420 = 0;
                local_418 = 0;
                uStack_410 = 0;
                local_408 = 0;
                uStack_400 = 0;
                local_3f8 = 0;
                uStack_3f0 = 0;
                local_3e8 = &uStack_420;
                local_3e0 = &local_3d8;
                local_3d0 = 0;
                local_3d8 = 0;
                local_3c8 = 0;
                memset(&local_718,0,0x68);
                local_718 = local_508;
                local_714 = local_504;
                local_710 = local_500;
                local_70c = local_4fc;
                local_708 = local_4f8;
                local_700 = local_4f0;
                local_6f8 = local_4e8;
                local_6f0 = local_4e0;
                local_6e8 = local_4d8;
                local_6e0 = local_4d0;
                local_6d8 = &local_710;
                local_6d0 = &local_6c8;
                local_6c0 = 0;
                local_6c8 = 0;
                if (local_4d0 != 0) {
                  LOCK();
                  *(int *)(local_4d0 + 0x14) = *(int *)(local_4d0 + 0x14) + 1;
                  UNLOCK();
                }
                if (local_504 < 3) {
                  local_6c8 = *local_4c0;
                  local_6c0 = local_4c0[1];
                }
                else {
                  local_714 = 0;
                  cv::Mat::copySize((Mat *)&local_718,(Mat *)&local_508);
                }
                local_6b8 = local_4a8;
                local_580 = 0;
                local_598 = 0x20;
                local_590 = 0;
                local_588 = *(longlong *)(param_1 + 0x810) + *plVar3;
                local_5a0 = plVar3;
                FUN_18002abf0((longlong *)&local_5a0,(longlong)puVar1,(Mat *)&local_348,
                              (Mat *)&local_428);
                FUN_180004050((int *)puVar1,plVar3,(longlong)piVar2,(longlong)&local_3b8,
                              (longlong)&local_348,(longlong)&local_428);
                if (*(int *)(param_1 + 0x28) < *piVar2) {
                  *(undefined4 *)(param_1 + 0xb20) = 0;
                  *(int *)(param_1 + 0xb24) = *piVar2;
                  *(undefined8 *)(param_1 + 0xb28) = *(undefined8 *)(param_1 + 0x810);
                  if ((Mat *)(param_1 + 0xb38) != (Mat *)&local_578) {
                    FUN_180001780((Mat *)(param_1 + 0xb38),(Mat *)&local_578);
                  }
                  *(undefined4 *)(param_1 + 0xb98) = local_518;
                  if ((Mat *)(param_1 + 0xba0) != (Mat *)&local_498) {
                    FUN_180001780((Mat *)(param_1 + 0xba0),(Mat *)&local_498);
                  }
                  *(undefined4 *)(param_1 + 0xc00) = local_438;
                  if ((Mat *)(param_1 + 0xc08) != (Mat *)&local_348) {
                    FUN_180001780((Mat *)(param_1 + 0xc08),(Mat *)&local_348);
                  }
                  *(undefined4 *)(param_1 + 0xc68) = local_2e8;
                  if ((Mat *)(param_1 + 0xc70) != (Mat *)&local_428) {
                    FUN_180001780((Mat *)(param_1 + 0xc70),(Mat *)&local_428);
                  }
                  *(undefined4 *)(param_1 + 0xcd0) = local_3c8;
                  if ((Mat *)(param_1 + 0xcd8) != (Mat *)&local_718) {
                    FUN_180001780((Mat *)(param_1 + 0xcd8),(Mat *)&local_718);
                  }
                  *(undefined4 *)(param_1 + 0xd38) = local_6b8;
                  if (local_848 != (code *)0x0) {
                    (*local_848)(local_840,0);
                  }
                }
                else {
                  memset(&local_6a8,0,0x68);
                  local_6a8 = (undefined4)local_638;
                  local_6a4 = local_638._4_4_;
                  local_6a0 = (undefined4)uStack_630;
                  local_69c = uStack_630._4_4_;
                  local_698 = local_628;
                  local_690 = uStack_620;
                  local_688 = local_618;
                  local_680 = uStack_610;
                  local_678 = local_608;
                  local_670 = lStack_600;
                  local_668 = &local_6a0;
                  local_660 = &local_658;
                  local_650 = 0;
                  local_658 = 0;
                  if (lStack_600 != 0) {
                    LOCK();
                    *(int *)(lStack_600 + 0x14) = *(int *)(lStack_600 + 0x14) + 1;
                    UNLOCK();
                  }
                  if (local_638._4_4_ < 3) {
                    local_658 = *local_5f0;
                    local_650 = local_5f0[1];
                  }
                  else {
                    local_6a4 = 0;
                    cv::Mat::copySize((Mat *)&local_6a8,(Mat *)&local_638);
                  }
                  local_648 = local_5d8;
                  FUN_1800043f0(puVar1,plVar3,piVar2,param_1 + 0x878,
                                *(int *)(param_1 + 0x4e0) *
                                ((*(int *)(param_1 + 0x20) != 8) + 1) * *(int *)(param_1 + 0x4dc) *
                                *(int *)(param_1 + 0x1c),&local_348,&local_428,(Mat *)&local_718);
                  FUN_180001d80("& CompFineEFI",0,0,0,0);
                  uVar9 = local_840;
                  pcVar5 = local_848;
                  if ((local_848 == (code *)0x0) || (iVar6 = (*local_848)(local_840,0), iVar6 == 0))
                  {
                    FUN_180003550((longlong)puVar1,plVar3,local_268,&local_718,(longlong)local_2d8,
                                  (Mat *)&local_6a8,4);
                    FUN_180001d80("& BlendEFI",0,0,0,0);
                    local_7f0 = 0;
                    local_800[0] = 0x2010000;
                    local_7f8 = local_118;
                    iVar6 = 2;
                    if (*(int *)(param_1 + 0x20) == 8) {
                      iVar6 = 0;
                    }
                    cv::Mat::convertTo((Mat *)&local_6a8,(_OutputArray *)local_800,iVar6,1.0,0.0);
                    local_838 = local_488;
                    uStack_830 = (undefined4)local_448;
                    uStack_760 = CONCAT44(uStack_82c,(undefined4)local_448);
                    local_768 = local_488;
                    local_828 = local_568;
                    uStack_820 = (undefined4)local_528;
                    uStack_740 = CONCAT44(uStack_81c,(undefined4)local_528);
                    local_748 = local_568;
                    local_738 = *(longlong *)(param_1 + 0x588);
                    uStack_730 = *(undefined8 *)(param_1 + 0x590);
                    local_728 = *(undefined8 *)(param_1 + 0x598);
                    uStack_720 = *(undefined8 *)(param_1 + 0x5a0);
                    FUN_18001ba70(*puVar1,*(int *)(param_1 + 0xc),&local_738,&local_748,&local_768);
                    local_7a8 = 0;
                    local_7b8[0] = 0x2010000;
                    local_7b0 = local_b8;
                    cv::Mat::copyTo((Mat *)&local_498,(_OutputArray *)local_7b8);
                    FUN_180001d80("& Construct Finished",0,0,0,0);
                    if (pcVar5 != (code *)0x0) {
                      (*pcVar5)(uVar9,0);
                    }
                  }
                  if (local_670 != 0) {
                    LOCK();
                    piVar2 = (int *)(local_670 + 0x14);
                    iVar6 = *piVar2;
                    *piVar2 = *piVar2 + -1;
                    UNLOCK();
                    if (iVar6 == 1) {
                      cv::Mat::deallocate((Mat *)&local_6a8);
                    }
                  }
                  local_670 = 0;
                  local_698 = 0;
                  local_680 = 0;
                  local_688 = 0;
                  local_690 = 0;
                  uVar11 = uVar13;
                  uVar12 = uVar13;
                  if (0 < local_6a4) {
                    do {
                      *(undefined4 *)(uVar12 + (longlong)local_668) = 0;
                      uVar10 = (int)uVar11 + 1;
                      uVar11 = (ulonglong)uVar10;
                      uVar12 = uVar12 + 4;
                    } while ((int)uVar10 < local_6a4);
                  }
                  if (local_660 != &local_658) {
                    cv::fastFree(local_660);
                  }
                }
                local_5a0[1] = local_5a0[1] - local_590;
                if (local_6e0 != 0) {
                  LOCK();
                  piVar2 = (int *)(local_6e0 + 0x14);
                  iVar6 = *piVar2;
                  *piVar2 = *piVar2 + -1;
                  UNLOCK();
                  if (iVar6 == 1) {
                    cv::Mat::deallocate((Mat *)&local_718);
                  }
                }
                local_6e0 = 0;
                local_708 = 0;
                local_6f0 = 0;
                local_6f8 = 0;
                local_700 = 0;
                uVar11 = uVar13;
                if (0 < local_714) {
                  do {
                    *(undefined4 *)(uVar11 + (longlong)local_6d8) = 0;
                    uVar10 = (int)uVar13 + 1;
                    uVar13 = (ulonglong)uVar10;
                    uVar11 = uVar11 + 4;
                  } while ((int)uVar10 < local_714);
                }
                if (local_6d0 != &local_6c8) {
                  cv::fastFree(local_6d0);
                }
                FUN_180001900((Mat *)&local_428);
                if (local_3e0 != &local_3d8) {
                  cv::fastFree(local_3e0);
                }
                FUN_180001900((Mat *)&local_348);
                if (local_300 != &local_2f8) {
                  cv::fastFree(local_300);
                }
              }
            }
          }
        }
        local_5c8[1] = local_5c8[1] - local_5b8;
        FUN_180001900((Mat *)&local_498);
        if (local_450 != &local_448) {
          cv::fastFree(local_450);
        }
        FUN_180001900((Mat *)&local_3b8);
        if (local_370 != &local_368) {
          cv::fastFree(local_370);
        }
        FUN_180001900((Mat *)&local_578);
        if (local_530 != &local_528) {
          cv::fastFree(local_530);
        }
        FUN_180001900((Mat *)&local_638);
        if (local_5f0 != &local_5e8) {
          cv::fastFree(local_5f0);
        }
      }
      FUN_180001900((Mat *)&local_508);
      if (local_4c0 != local_4b8) {
        cv::fastFree(local_4c0);
      }
      FUN_180001900(local_2d8);
      if (local_290 != local_288) {
        cv::fastFree(local_290);
      }
      FUN_180001900(local_268);
      if (local_220 != local_218) {
        cv::fastFree(local_220);
      }
      FUN_180001900(local_188);
      if (local_140 != local_138) {
        cv::fastFree(local_140);
      }
      FUN_180001900(local_1f8);
      if (local_1b0 != local_1a8) {
        cv::fastFree(local_1b0);
      }
    }
    FUN_180001900(local_118);
    if (local_d0 != local_c8) {
      cv::fastFree(local_d0);
    }
    FUN_180001900(local_b8);
    if (local_70 != local_68) {
      cv::fastFree(local_70);
    }
  }
  return local_res8;
}


```

## {'procedure': '0x1800033d0'}
Evidence: `ev_2fc389a03d9d3e986eba3a170b1562b137e9a5d46a60d41a2ff353e5b5df69bc`

```c

void FUN_1800033d0(int param_1,int param_2,longlong *param_3,float param_4,float param_5,
                  longlong *param_6)

{
  longlong lVar1;
  longlong lVar2;
  longlong lVar3;
  longlong lVar4;
  float fVar5;
  
  if (0 < param_2) {
    lVar3 = (longlong)param_1;
    lVar4 = 0;
    do {
      lVar2 = 0;
      if (3 < lVar3) {
        do {
          lVar1 = lVar2 * 4;
          fVar5 = param_4 * *(float *)((int)param_3[1] * lVar4 + *param_3 + lVar1);
          if (param_5 < fVar5) {
            fVar5 = param_5;
          }
          *(float *)((int)param_6[1] * lVar4 + lVar1 + *param_6) = fVar5;
          fVar5 = param_4 * *(float *)((int)param_3[1] * lVar4 + *param_3 + 4 + lVar1);
          if (param_5 < fVar5) {
            fVar5 = param_5;
          }
          *(float *)((int)param_6[1] * lVar4 + lVar1 + 4 + *param_6) = fVar5;
          fVar5 = param_4 * *(float *)((int)param_3[1] * lVar4 + lVar1 + 8 + *param_3);
          if (param_5 < fVar5) {
            fVar5 = param_5;
          }
          *(float *)((int)param_6[1] * lVar4 + lVar1 + 8 + *param_6) = fVar5;
          fVar5 = param_4 * *(float *)((int)param_3[1] * lVar4 + lVar1 + 0xc + *param_3);
          if (param_5 < fVar5) {
            fVar5 = param_5;
          }
          lVar2 = lVar2 + 4;
          *(float *)((int)param_6[1] * lVar4 + lVar1 + 0xc + *param_6) = fVar5;
        } while (lVar2 < lVar3 + -3);
      }
      for (; lVar2 < lVar3; lVar2 = lVar2 + 1) {
        fVar5 = param_4 * *(float *)((int)param_3[1] * lVar4 + *param_3 + lVar2 * 4);
        if (param_5 < fVar5) {
          fVar5 = param_5;
        }
        *(float *)((int)param_6[1] * lVar4 + lVar2 * 4 + *param_6) = fVar5;
      }
      lVar4 = lVar4 + 1;
    } while (lVar4 < param_2);
  }
  return;
}


```

## {'procedure': '0x18001e660'}
Evidence: `ev_3d349d7919cd6d508a6c67c80a34ecb7d49afa84a872fca44b8c57e9be7a1bd6`

```c

void FUN_18001e660(longlong param_1,longlong param_2,longlong param_3,undefined *param_4,
                  undefined8 param_5)

{
  int extraout_EAX;
  ulonglong uVar1;
  
  uVar1 = FUN_18001e6a0(param_1,param_2,param_3,param_4,param_5);
  if (extraout_EAX != 1) {
    *(undefined8 *)(uVar1 + 0x840) = 0;
    *(undefined1 *)(uVar1 + 0x868) = 0;
  }
  return;
}


```

## {'procedure': '0x1800251b0'}
Evidence: `ev_3dbfc5a9562c7d4471cbee5bacb7046aa5c7a5231a7b602fe645654395f0581f`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_1800251b0(longlong param_1,int param_2,float param_3,undefined8 param_4,longlong param_5,
                  longlong param_6,int param_7,longlong param_8,longlong param_9,longlong param_10,
                  longlong param_11,longlong param_12,longlong param_13,undefined8 param_14,
                  longlong *param_15,longlong *param_16)

{
  ushort uVar1;
  ushort uVar2;
  longlong lVar3;
  int iVar4;
  uint uVar5;
  float *pfVar6;
  float *pfVar7;
  uint uVar8;
  longlong lVar9;
  float *pfVar10;
  longlong lVar11;
  float *pfVar12;
  float *pfVar13;
  int iVar14;
  longlong lVar15;
  ushort *puVar16;
  short sVar17;
  uint uVar18;
  longlong lVar19;
  longlong lVar20;
  ulonglong uVar21;
  longlong lVar22;
  float fVar23;
  float fVar24;
  undefined1 auStackY_128 [32];
  int local_f8;
  int local_f4;
  undefined4 uStack_9c;
  longlong local_88;
  undefined8 uStack_80;
  float local_78 [2];
  longlong *local_70;
  undefined8 local_68;
  ulonglong local_60;
  float *local_58;
  undefined8 local_50;
  ulonglong local_48;
  
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_128;
  iVar14 = (int)param_4;
  lVar20 = (longlong)iVar14;
  local_70 = (longlong *)0x0;
  local_68 = 0;
  local_60 = 0;
  local_58 = (float *)0x0;
  local_50 = 0;
  iVar4 = omp_get_thread_num();
  local_70 = (longlong *)(*(longlong *)(param_1 + 0x30) + (longlong)iVar4 * 0x18);
  local_68 = CONCAT44(local_68._4_4_,0x20);
  pfVar10 = (float *)(*local_70 + local_70[1]);
  local_60 = (longlong)(iVar14 * 4) * 4 + 0x1fU & 0xffffffffffffffe0;
  lVar3 = local_70[1];
  local_70[1] = local_60 + lVar3;
  local_58 = pfVar10;
  if (local_60 + lVar3 <= (ulonglong)local_70[2]) {
    local_f8 = 0;
    fVar24 = param_3;
    if (0 < (int)param_3) {
      do {
        local_f4 = 0;
        if (0 < param_2) {
          do {
            lVar11 = (longlong)
                     ((int)((ulonglong)(longlong)*(int *)(param_5 + 8) >> 2) * local_f8 + local_f4);
            lVar3 = lVar11 * 4;
            fVar23 = (float)rintf();
            lVar22 = (longlong)(int)fVar23;
            uVar1 = *(ushort *)(param_8 + lVar22 * 2);
            local_78[0] = (float)(uint)uVar1;
            lVar9 = 0;
            if (3 < lVar20) {
              lVar19 = (longlong)param_7;
              pfVar7 = pfVar10 + 2;
              lVar11 = lVar19 + lVar11 * 2;
              pfVar12 = (float *)(param_6 + lVar11 * 2);
              lVar15 = (lVar20 - 4U >> 2) + 1;
              lVar9 = lVar15 * 4;
              do {
                pfVar7[-2] = *(float *)(lVar3 + lVar11 * -2 + (longlong)pfVar12);
                pfVar7[-1] = *(float *)((longlong)pfVar12 - lVar19);
                *pfVar7 = *pfVar12;
                pfVar7[1] = *(float *)(lVar19 + (longlong)pfVar12);
                pfVar12 = pfVar12 + lVar19;
                pfVar7 = pfVar7 + 4;
                lVar15 = lVar15 + -1;
              } while (lVar15 != 0);
            }
            if (lVar9 < lVar20) {
              pfVar7 = (float *)(param_7 * lVar9 + lVar3 + param_6);
              do {
                pfVar10[lVar9] = *pfVar7;
                lVar9 = lVar9 + 1;
                pfVar7 = (float *)((longlong)pfVar7 + (longlong)param_7);
              } while (lVar9 < lVar20);
            }
            uVar5 = (uint)*(ushort *)(param_9 + lVar22 * 2);
            uVar2 = *(ushort *)(param_10 + lVar22 * 2);
            if (uVar2 == 0) {
              local_78[0] = *pfVar10;
              lVar9 = 1;
              if (1 < lVar20) {
                if (3 < lVar20 + -1) {
                  pfVar7 = pfVar10 + 3;
                  lVar11 = (lVar20 - 5U >> 2) + 1;
                  lVar9 = lVar11 * 4 + 1;
                  do {
                    pfVar12 = local_78;
                    if (local_78[0] < pfVar7[-2]) {
                      pfVar12 = pfVar7 + -2;
                    }
                    local_78[0] = *pfVar12;
                    pfVar6 = local_78;
                    if (*pfVar12 < pfVar7[-1]) {
                      pfVar6 = pfVar7 + -1;
                    }
                    local_78[0] = *pfVar6;
                    pfVar12 = local_78;
                    if (*pfVar6 < *pfVar7) {
                      pfVar12 = pfVar7;
                    }
                    local_78[0] = *pfVar12;
                    pfVar6 = local_78;
                    if (*pfVar12 < pfVar7[1]) {
                      pfVar6 = pfVar7 + 1;
                    }
                    local_78[0] = *pfVar6;
                    pfVar7 = pfVar7 + 4;
                    lVar11 = lVar11 + -1;
                  } while (lVar11 != 0);
                }
                if (lVar9 < lVar20) {
                  pfVar7 = pfVar10 + lVar9;
                  lVar9 = lVar20 - lVar9;
                  do {
                    pfVar12 = local_78;
                    if (local_78[0] < *pfVar7) {
                      pfVar12 = pfVar7;
                    }
                    local_78[0] = *pfVar12;
                    pfVar7 = pfVar7 + 1;
                    lVar9 = lVar9 + -1;
                  } while (lVar9 != 0);
                }
              }
              sVar17 = 0;
            }
            else {
              uVar21 = (ulonglong)uVar5;
              pfVar7 = pfVar10 + lVar20;
              pfVar12 = pfVar10 + lVar20 * 2;
              pfVar6 = pfVar10 + lVar20 * 3;
              uStack_80 = CONCAT44(uStack_9c,(uint)uVar1);
              local_88 = param_11;
              FUN_180013fd0(iVar14,&local_88,(longlong)pfVar10,pfVar7,pfVar12,(longlong)pfVar6);
              iVar4 = (uint)*(ushort *)(param_12 + uVar21 * 2) - (int)local_78[0];
              fVar24 = *(float *)(param_13 + uVar21 * 4);
              if (iVar4 < 0) {
                iVar4 = 0;
                fVar24 = 0.0;
              }
              fVar24 = ((fVar24 * pfVar6[iVar4] + pfVar12[iVar4]) * fVar24 + pfVar7[iVar4]) * fVar24
                       + pfVar10[iVar4];
              uVar18 = 0;
              sVar17 = 0;
              uVar8 = 1;
              if (1 < uVar2) {
                pfVar13 = (float *)(param_13 + 4 + uVar21 * 4);
                puVar16 = (ushort *)(param_12 + uVar21 * 2);
                do {
                  puVar16 = puVar16 + 1;
                  iVar4 = (uint)*puVar16 - (int)local_78[0];
                  fVar23 = *pfVar13;
                  fVar23 = ((fVar23 * pfVar6[iVar4] + pfVar12[iVar4]) * fVar23 + pfVar7[iVar4]) *
                           fVar23 + pfVar10[iVar4];
                  if (fVar24 < fVar23) {
                    uVar18 = uVar8 & 0xffff;
                    fVar24 = fVar23;
                  }
                  sVar17 = (short)uVar18;
                  uVar8 = uVar8 + 1;
                  pfVar13 = pfVar13 + 1;
                } while ((int)uVar8 < (int)(uint)uVar2);
              }
            }
            *(float *)(lVar3 + *param_15) = (float)(int)((int)sVar17 + uVar5);
            *(float *)(lVar3 + *param_16) = fVar24;
            local_f4 = local_f4 + 1;
          } while (local_f4 < param_2);
        }
        local_f8 = local_f8 + 1;
      } while (local_f8 < (int)param_3);
    }
    local_70[1] = local_70[1] - local_60;
    return;
  }
                    /* WARNING: Subroutine does not return */
  FUN_180001b90("Allocating memory exceeding the limit ...",(longlong)iVar4 * 3,local_70,param_4);
}


```

## {'procedure': '0x180009040'}
Evidence: `ev_48a28f53ec9bee8d160ba8b19ed40db26b90c33610c7e87c609ce9d363781012`

```c

void FUN_180009040(undefined8 param_1,longlong param_2,longlong param_3,longlong param_4,
                  longlong param_5,longlong param_6)

{
  longlong local_68;
  undefined8 uStack_60;
  longlong local_58;
  undefined8 uStack_50;
  longlong local_48;
  undefined8 uStack_40;
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  undefined8 local_18;
  undefined8 uStack_10;
  
  local_58 = *(longlong *)(param_5 + 0x10);
  local_18 = *(undefined8 *)(param_6 + 0x38);
  uStack_10 = *(undefined8 *)(param_6 + 0x40);
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_5 + 0x50));
  local_48 = *(longlong *)(param_4 + 0x10);
  uStack_50 = uStack_60;
  local_28 = *(longlong *)(param_6 + 0x28);
  uStack_20 = *(undefined8 *)(param_6 + 0x30);
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_4 + 0x50));
  local_38 = *(longlong *)(param_3 + 0x10);
  uStack_40 = uStack_60;
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_68 = *(longlong *)(param_2 + 0x10);
  uStack_30 = uStack_60;
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_2 + 0x50));
  FUN_18000e4f0(*(int *)(param_6 + 0xc),*(int *)(param_6 + 0x10),&local_68,&local_38,&local_48,
                &local_28,&local_58);
  return;
}


```

## {'procedure': '0x180002e50'}
Evidence: `ev_4995d6886e1994bac47df223cac38803753e228fbcafe4a2007e423a28ef56e1`

```c
0x180002e50: MOV R11, RSP
0x180002e53: PUSH RDI
0x180002e54: PUSH R12
0x180002e56: PUSH R13
0x180002e58: PUSH R14
0x180002e5a: PUSH R15
0x180002e5c: SUB RSP, 0x70
0x180002e60: MOV qword ptr [R11 + -0x78], -0x2
0x180002e68: MOV qword ptr [R11 + 0x8], RBX
0x180002e6c: MOV qword ptr [R11 + 0x10], RBP
0x180002e70: MOV qword ptr [R11 + 0x20], RSI
0x180002e74: MOVAPS xmmword ptr [RSP + 0x60], XMM6
0x180002e79: MOV RAX, qword ptr [0x18008b080]
0x180002e80: XOR RAX, RSP
0x180002e83: MOV qword ptr [RSP + 0x50], RAX
0x180002e88: MOV RBX, R8
0x180002e8b: MOV R15, RDX
0x180002e8e: MOV R13, RCX
0x180002e91: MOVSXD RSI, dword ptr [R8]
0x180002e94: CMP byte ptr [RCX + 0x24], 0x0
0x180002e98: JZ 0x18000300d
0x180002e9e: XOR EAX, EAX
0x180002ea0: MOV qword ptr [R11 + -0x70], RAX
0x180002ea4: MOV qword ptr [R11 + -0x68], RAX
0x180002ea8: MOV qword ptr [R11 + -0x60], RAX
0x180002eac: MOV qword ptr [R11 + -0x58], RAX
0x180002eb0: MOV qword ptr [R11 + -0x50], RAX
0x180002eb4: MOV qword ptr [R11 + -0x70], RDX
0x180002eb8: MOV dword ptr [RSP + 0x30], 0x20
0x180002ec0: MOV RCX, qword ptr [RDX + 0x8]
0x180002ec4: MOV R14, qword ptr [RDX]
0x180002ec7: ADD R14, RCX
0x180002eca: MOV qword ptr [R11 + -0x58], R14
0x180002ece: LEA R12, [0x1f + RSI*0x8]
0x180002ed6: AND R12, -0x20
0x180002eda: MOV qword ptr [R11 + -0x60], R12
0x180002ede: LEA RAX, [RCX + R12*0x1]
0x180002ee2: MOV qword ptr [RDX + 0x8], RAX
0x180002ee6: CMP RAX, qword ptr [RDX + 0x10]
0x180002eea: JBE 0x180002ef9
0x180002eec: LEA RCX, [0x18007b2b0]
0x180002ef3: CALL 0x180001b90
0x180002ef9: XOR EDI, EDI
0x180002efb: MOV R8D, EDI
0x180002efe: MOV EDX, EDI
0x180002f00: CMP RSI, 0x4
0x180002f04: JL 0x180002f7b
0x180002f06: LEA R11, [RSI + -0x3]
0x180002f0a: LEA R10D, [RDI + 0x2]
0x180002f0e: LEA R9, [R14 + 0x8]
0x180002f12: NOP dword ptr [RAX]
0x180002f16: NOP dword ptr [RAX + RAX*0x1]
0x180002f20: MOV RAX, qword ptr [RBX + 0x8]
0x180002f24: MOV ECX, dword ptr [RAX + RDX*0x4]
0x180002f27: MOV dword ptr [R9 + -0x8], ECX
0x180002f2b: MOV dword ptr [R9 + -0x4], R8D
0x180002f2f: MOV RAX, qword ptr [RBX + 0x8]
0x180002f33: MOV ECX, dword ptr [RAX + RDX*0x4 + 0x4]
0x180002f37: MOV dword ptr [R9], ECX
0x180002f3a: LEA EAX, [R10 + -0x1]
0x180002f3e: MOV dword ptr [R9 + 0x4], EAX
0x180002f42: MOV RAX, qword ptr [RBX + 0x8]
0x180002f46: MOV ECX, dword ptr [RAX + RDX*0x4 + 0x8]
0x180002f4a: MOV dword ptr [R9 + 0x8], ECX
0x180002f4e: MOV dword ptr [R9 + 0xc], R10D
0x180002f52: MOV RAX, qword ptr [RBX + 0x8]
0x180002f56: MOV ECX, dword ptr [RAX + RDX*0x4 + 0xc]
0x180002f5a: MOV dword ptr [R9 + 0x10], ECX
0x180002f5e: LEA EAX, [R10 + 0x1]
0x180002f62: MOV dword ptr [R9 + 0x14], EAX
0x180002f66: ADD R8D, 0x4
0x180002f6a: ADD R10D, 0x4
0x180002f6e: ADD RDX, 0x4
0x180002f72: LEA R9, [R9 + 0x20]
0x180002f76: CMP RDX, R11
0x180002f79: JL 0x180002f20
0x180002f7b: CMP RDX, RSI
0x180002f7e: JGE 0x180002f9b
0x180002f80: MOV RAX, qword ptr [RBX + 0x8]
0x180002f84: MOV ECX, dword ptr [RAX + RDX*0x4]
0x180002f87: MOV dword ptr [R14 + RDX*0x8], ECX
0x180002f8b: MOV dword ptr [R14 + RDX*0x8 + 0x4], R8D
0x180002f90: INC R8D
0x180002f93: INC RDX
0x180002f96: CMP RDX, RSI
0x180002f99: JL 0x180002f80
0x180002f9b: LEA RDX, [R14 + RSI*0x8]
0x180002f9f: XOR EAX, EAX
0x180002fa1: MOV R8, RDX
0x180002fa4: SUB R8, R14
0x180002fa7: SAR R8, 0x3
0x180002fab: MOVZX R9D, AL
0x180002faf: MOV RCX, R14
0x180002fb2: CALL 0x1800114f0
0x180002fb7: TEST ESI, ESI
0x180002fb9: JLE 0x180003004
0x180002fbb: MOV R8, RDI
0x180002fbe: MOV R9, RDI
0x180002fc1: LEA RDX, [R14 + 0x4]
0x180002fc5: NOP word ptr [RAX + RAX*0x1]
0x180002fd0: MOV RCX, qword ptr [RBX + 0x10]
0x180002fd4: MOV EAX, dword ptr [RDX + -0x4]
0x180002fd7: MOV dword ptr [R9 + RCX*0x1], EAX
0x180002fdb: MOV RCX, qword ptr [RBX + 0x18]
0x180002fdf: MOVZX EAX, word ptr [RDX]
0x180002fe2: MOV word ptr [R8 + RCX*0x1], AX
0x180002fe7: MOVSXD RCX, dword ptr [RDX]
0x180002fea: MOV RAX, qword ptr [RBX + 0x20]
0x180002fee: MOV word ptr [RAX + RCX*0x2], DI
0x180002ff2: INC EDI
0x180002ff4: LEA R9, [R9 + 0x4]
0x180002ff8: LEA R8, [R8 + 0x2]
0x180002ffc: LEA RDX, [RDX + 0x8]
0x180003000: CMP EDI, ESI
0x180003002: JL 0x180002fd0
0x180003004: SUB qword ptr [R15 + 0x8], R12
0x180003008: JMP 0x1800030ce
0x18000300d: CMP ESI, 0x1
0x180003010: JLE 0x1800030ce
0x180003016: MOV RAX, qword ptr [R8 + 0x8]
0x18000301a: MOVSS XMM0, dword ptr [RAX]
0x18000301e: COMISS XMM0, dword ptr [RAX + RSI*0x4 + -0x4]
0x180003023: SETA AL
0x180003026: MOV byte ptr [R8 + 0x28], AL
0x18000302a: XOR EDI, EDI
0x18000302c: TEST AL, AL
0x18000302e: JZ 0x18000308c
0x180003030: TEST ESI, ESI
0x180003032: JLE 0x1800030ce
0x180003038: MOV R8D, EDI
0x18000303b: MOV R9D, EDI
0x18000303e: LEA EDX, [RSI + -0x1]
0x180003041: MOVSXD RAX, EDX
0x180003044: LEA R10, [RAX + RAX*0x1]
0x180003048: LEA R11, [RAX*0x4]
0x180003050: MOV RAX, qword ptr [RBX + 0x8]
0x180003054: MOV RCX, qword ptr [RBX + 0x10]
0x180003058: MOV EAX, dword ptr [R11 + RAX*0x1]
0x18000305c: MOV dword ptr [R9 + RCX*0x1], EAX
0x180003060: MOV RAX, qword ptr [RBX + 0x18]
0x180003064: MOV word ptr [R8 + RAX*0x1], DX
0x180003069: MOV RAX, qword ptr [RBX + 0x20]
0x18000306d: MOV word ptr [R10 + RAX*0x1], DI
0x180003072: INC EDI
0x180003074: DEC EDX
0x180003076: LEA R11, [R11 + -0x4]
0x18000307a: LEA R10, [R10 + -0x2]
0x18000307e: LEA R9, [R9 + 0x4]
0x180003082: LEA R8, [R8 + 0x2]
0x180003086: CMP EDI, ESI
0x180003088: JL 0x180003050
0x18000308a: JMP 0x1800030ce
0x18000308c: TEST ESI, ESI
0x18000308e: JLE 0x1800030ce
0x180003090: MOV RDX, RDI
0x180003093: MOV R8, RDI
0x180003096: NOP dword ptr [RAX + RAX*0x1]
0x1800030a0: MOV RAX, qword ptr [RBX + 0x8]
0x1800030a4: MOV RCX, qword ptr [RBX + 0x10]
0x1800030a8: MOV EAX, dword ptr [R8 + RAX*0x1]
0x1800030ac: MOV dword ptr [R8 + RCX*0x1], EAX
0x1800030b0: MOV RAX, qword ptr [RBX + 0x18]
0x1800030b4: MOV word ptr [RDX + RAX*0x1], DI
0x1800030b8: MOV RAX, qword ptr [RBX + 0x20]
0x1800030bc: MOV word ptr [RDX + RAX*0x1], DI
0x1800030c0: INC EDI
0x1800030c2: LEA R8, [R8 + 0x4]
0x1800030c6: LEA RDX, [RDX + 0x2]
0x1800030ca: CMP EDI, ESI
0x1800030cc: JL 0x1800030a0
0x1800030ce: MOV RAX, qword ptr [RBX + 0x10]
0x1800030d2: MOVSS XMM3, dword ptr [RAX]
0x1800030d6: MOV R9D, 0x1
0x1800030dc: MOV R8D, R9D
0x1800030df: CMP RSI, R9
0x1800030e2: JLE 0x180003348
0x1800030e8: LEA RAX, [RSI + -0x1]
0x1800030ec: MOVSD XMM5, qword ptr [0x18007c630]
0x1800030f4: MOVSS XMM6, dword ptr [0x18007c4f0]
0x1800030fc: CMP RAX, 0x4
0x180003100: JL 0x1800032d1
0x180003106: LEA R10, [RSI + -0x3]
0x18000310a: NOP word ptr [RAX + RAX*0x1]
0x180003110: MOV RAX, qword ptr [RBX + 0x10]
0x180003114: MOVSS XMM4, dword ptr [RAX + R8*0x4]
0x18000311a: MOVAPS XMM0, XMM4
0x18000311d: SUBSS XMM0, XMM3
0x180003121: CVTPS2PD XMM1, XMM0
0x180003124: ANDPS XMM1, XMM5
0x180003127: CVTPD2PS XMM2, XMM1
0x18000312b: COMISS XMM6, XMM2
0x18000312e: JC 0x180003147
0x180003130: LEA EDX, [R9 + -0x1]
0x180003134: MOV RAX, qword ptr [RBX + 0x18]
0x180003138: MOVZX ECX, word ptr [RAX + R8*0x2]
0x18000313d: MOV RAX, qword ptr [RBX + 0x20]
0x180003141: MOV word ptr [RAX + RCX*0x2], DX
0x180003145: JMP 0x18000317a
0x180003147: MOVSXD RDX, R9D
0x18000314a: MOVSS dword ptr [RAX + RDX*0x4], XMM4
0x18000314f: MOV RAX, qword ptr [RBX + 0x10]
0x180003153: MOVSS XMM3, dword ptr [RAX + RDX*0x4]
0x180003158: MOV RAX, qword ptr [RBX + 0x18]
0x18000315c: MOVZX ECX, word ptr [RAX + R8*0x2]
0x180003161: MOV RAX, qword ptr [RBX + 0x20]
0x180003165: MOV word ptr [RAX + RCX*0x2], R9W
0x18000316a: MOV RCX, qword ptr [RBX + 0x18]
0x18000316e: MOVZX EAX, word ptr [RCX + R8*0x2]
0x180003173: MOV word ptr [RCX + RDX*0x2], AX
0x180003177: INC R9D
0x18000317a: MOV RAX, qword ptr [RBX + 0x10]
0x18000317e: MOVSS XMM4, dword ptr [RAX + R8*0x4 + 0x4]
0x180003185: MOVAPS XMM0, XMM4
0x180003188: SUBSS XMM0, XMM3
0x18000318c: CVTPS2PD XMM1, XMM0
0x18000318f: ANDPS XMM1, XMM5
0x180003192: CVTPD2PS XMM2, XMM1
0x180003196: COMISS XMM6, XMM2
0x180003199: JC 0x1800031b3
0x18000319b: LEA EDX, [R9 + -0x1]
0x18000319f: MOV RAX, qword ptr [RBX + 0x18]
0x1800031a3: MOVZX ECX, word ptr [RAX + R8*0x2 + 0x2]
0x1800031a9: MOV RAX, qword ptr [RBX + 0x20]
0x1800031ad: MOV word ptr [RAX + RCX*0x2], DX
0x1800031b1: JMP 0x1800031e8
0x1800031b3: MOVSXD RDX, R9D
0x1800031b6: MOVSS dword ptr [RAX + RDX*0x4], XMM4
0x1800031bb: MOV RAX, qword ptr [RBX + 0x10]
0x1800031bf: MOVSS XMM3, dword ptr [RAX + RDX*0x4]
0x1800031c4: MOV RAX, qword ptr [RBX + 0x18]
0x1800031c8: MOVZX ECX, word ptr [RAX + R8*0x2 + 0x2]
0x1800031ce: MOV RAX, qword ptr [RBX + 0x20]
0x1800031d2: MOV word ptr [RAX + RCX*0x2], R9W
0x1800031d7: MOV RCX, qword ptr [RBX + 0x18]
0x1800031db: MOVZX EAX, word ptr [RCX + R8*0x2 + 0x2]
0x1800031e1: MOV word ptr [RCX + RDX*0x2], AX
0x1800031e5: INC R9D
0x1800031e8: MOV RAX, qword ptr [RBX + 0x10]
0x1800031ec: MOVSS XMM4, dword ptr [RAX + R8*0x4 + 0x8]
0x1800031f3: MOVAPS XMM0, XMM4
0x1800031f6: SUBSS XMM0, XMM3
0x1800031fa: CVTPS2PD XMM1, XMM0
0x1800031fd: ANDPS XMM1, XMM5
0x180003200: CVTPD2PS XMM2, XMM1
0x180003204: COMISS XMM6, XMM2
0x180003207: JC 0x180003221
0x180003209: LEA EDX, [R9 + -0x1]
0x18000320d: MOV RAX, qword ptr [RBX + 0x18]
0x180003211: MOVZX ECX, word ptr [RAX + R8*0x2 + 0x4]
0x180003217: MOV RAX, qword ptr [RBX + 0x20]
0x18000321b: MOV word ptr [RAX + RCX*0x2], DX
0x18000321f: JMP 0x180003256
0x180003221: MOVSXD RDX, R9D
0x180003224: MOVSS dword ptr [RAX + RDX*0x4], XMM4
0x180003229: MOV RAX, qword ptr [RBX + 0x10]
0x18000322d: MOVSS XMM3, dword ptr [RAX + RDX*0x4]
0x180003232: MOV RAX, qword ptr [RBX + 0x18]
0x180003236: MOVZX ECX, word ptr [RAX + R8*0x2 + 0x4]
0x18000323c: MOV RAX, qword ptr [RBX + 0x20]
0x180003240: MOV word ptr [RAX + RCX*0x2], R9W
0x180003245: MOV RCX, qword ptr [RBX + 0x18]
0x180003249: MOVZX EAX, word ptr [RCX + R8*0x2 + 0x4]
0x18000324f: MOV word ptr [RCX + RDX*0x2], AX
0x180003253: INC R9D
0x180003256: MOV RAX, qword ptr [RBX + 0x10]
0x18000325a: MOVSS XMM4, dword ptr [RAX + R8*0x4 + 0xc]
0x180003261: MOVAPS XMM0, XMM4
0x180003264: SUBSS XMM0, XMM3
0x180003268: CVTPS2PD XMM1, XMM0
0x18000326b: ANDPS XMM1, XMM5
0x18000326e: CVTPD2PS XMM2, XMM1
0x180003272: COMISS XMM6, XMM2
0x180003275: JC 0x18000328f
0x180003277: LEA EDX, [R9 + -0x1]
0x18000327b: MOV RAX, qword ptr [RBX + 0x18]
0x18000327f: MOVZX ECX, word ptr [RAX + R8*0x2 + 0x6]
0x180003285: MOV RAX, qword ptr [RBX + 0x20]
0x180003289: MOV word ptr [RAX + RCX*0x2], DX
0x18000328d: JMP 0x1800032c4
0x18000328f: MOVSXD RDX, R9D
0x180003292: MOVSS dword ptr [RAX + RDX*0x4], XMM4
0x180003297: MOV RAX, qword ptr [RBX + 0x10]
0x18000329b: MOVSS XMM3, dword ptr [RAX + RDX*0x4]
0x1800032a0: MOV RAX, qword ptr [RBX + 0x18]
0x1800032a4: MOVZX ECX, word ptr [RAX + R8*0x2 + 0x6]
0x1800032aa: MOV RAX, qword ptr [RBX + 0x20]
0x1800032ae: MOV word ptr [RAX + RCX*0x2], R9W
0x1800032b3: MOV RCX, qword ptr [RBX + 0x18]
0x1800032b7: MOVZX EAX, word ptr [RCX + R8*0x2 + 0x6]
0x1800032bd: MOV word ptr [RCX + RDX*0x2], AX
0x1800032c1: INC R9D
0x1800032c4: ADD R8, 0x4
0x1800032c8: CMP R8, R10
0x1800032cb: JL 0x180003110
0x1800032d1: CMP R8, RSI
0x1800032d4: JGE 0x180003348
0x1800032d6: MOV RAX, qword ptr [RBX + 0x10]
0x1800032da: MOVSS XMM4, dword ptr [RAX + R8*0x4]
0x1800032e0: MOVAPS XMM0, XMM4
0x1800032e3: SUBSS XMM0, XMM3
0x1800032e7: CVTPS2PD XMM1, XMM0
0x1800032ea: ANDPS XMM1, XMM5
0x1800032ed: CVTPD2PS XMM2, XMM1
0x1800032f1: COMISS XMM6, XMM2
0x1800032f4: JC 0x18000330d
0x1800032f6: LEA EDX, [R9 + -0x1]
0x1800032fa: MOV RAX, qword ptr [RBX + 0x18]
0x1800032fe: MOVZX ECX, word ptr [RAX + R8*0x2]
0x180003303: MOV RAX, qword ptr [RBX + 0x20]
0x180003307: MOV word ptr [RAX + RCX*0x2], DX
0x18000330b: JMP 0x180003340
0x18000330d: MOVSXD RDX, R9D
0x180003310: MOVSS dword ptr [RAX + RDX*0x4], XMM4
0x180003315: MOV RAX, qword ptr [RBX + 0x10]
0x180003319: MOVSS XMM3, dword ptr [RAX + RDX*0x4]
0x18000331e: MOV RAX, qword ptr [RBX + 0x18]
0x180003322: MOVZX ECX, word ptr [RAX + R8*0x2]
0x180003327: MOV RAX, qword ptr [RBX + 0x20]
0x18000332b: MOV word ptr [RAX + RCX*0x2], R9W
0x180003330: MOV RCX, qword ptr [RBX + 0x18]
0x180003334: MOVZX EAX, word ptr [RCX + R8*0x2]
0x180003339: MOV word ptr [RCX + RDX*0x2], AX
0x18000333d: INC R9D
0x180003340: INC R8
0x180003343: CMP R8, RSI
0x180003346: JL 0x1800032d6
0x180003348: MOV dword ptr [RBX + 0x4], R9D
0x18000334c: MOV RCX, qword ptr [RBX + 0x10]
0x180003350: MOVSXD RAX, R9D
0x180003353: MOVSS XMM1, dword ptr [RCX + RAX*0x4 + -0x4]
0x180003359: SUBSS XMM1, dword ptr [RCX]
0x18000335d: LEA EAX, [R9 + -0x1]
0x180003361: IMUL EAX, dword ptr [R13 + 0x7c]
0x180003366: MOVD XMM0, EAX
0x18000336a: CVTDQ2PS XMM0, XMM0
0x18000336d: DIVSS XMM1, XMM0
0x180003371: MOVSS dword ptr [RBX + 0x2c], XMM1
0x180003376: MOV RCX, qword ptr [RSP + 0x50]
0x18000337b: XOR RCX, RSP
0x18000337e: CALL 0x180072a30
0x180003383: LEA R11, [RSP + 0x70]
0x180003388: MOV RBX, qword ptr [R11 + 0x30]
0x18000338c: MOV RBP, qword ptr [R11 + 0x38]
0x180003390: MOV RSI, qword ptr [R11 + 0x48]
0x180003394: MOVAPS XMM6, xmmword ptr [RSP + 0x60]
0x180003399: MOV RSP, R11
0x18000339c: POP R15
0x18000339e: POP R14
0x1800033a0: POP R13
0x1800033a2: POP R12
0x1800033a4: POP RDI
0x1800033a5: RET
```

## {'procedure': 'EFI_GetMemorySize'}
Evidence: `ev_4fd99b6dd262829826d3cb4d8f0635bd9a76d63df10f5c3bfb10637a88c1d59d`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

ulonglong EFI_GetMemorySize(int *param_1,longlong *param_2,undefined8 param_3,char *param_4)

{
  int iVar1;
  ulonglong uVar2;
  int iVar3;
  undefined1 auStackY_b18 [32];
  undefined8 local_ae8;
  longlong local_ae0;
  longlong local_ad8 [2];
  ulonglong local_ac8;
  longlong local_ac0;
  longlong local_ab8;
  longlong local_ab0;
  int *local_aa8;
  longlong local_aa0;
  int local_a98;
  int local_a94;
  int local_a8c;
  int local_a88;
  int local_a84;
  int local_a80;
  ulonglong local_18;
  
                    /* 0x6270  5  EFI_GetMemorySize */
  local_18 = DAT_18008b080 ^ (ulonglong)auStackY_b18;
  if ((param_1 == (int *)0x0) || (param_2 == (longlong *)0x0)) {
    uVar2 = 0x11;
  }
  else {
    uVar2 = FUN_180002a40(param_1);
    if ((int)uVar2 == 0) {
      iVar1 = param_1[0x46];
      if (iVar1 == 0) {
        uVar2 = 0;
        iVar1 = param_1[3];
        if (iVar1 < 1) {
          iVar1 = param_1[5] * *param_1 * ((param_1[6] != 8) + 1);
        }
        iVar3 = param_1[4];
        if (iVar3 < 1) {
          iVar3 = *param_1 << 2;
        }
        param_2[2] = (longlong)(param_1[1] * iVar1);
        param_2[3] = (longlong)(param_1[1] * iVar3);
        *param_2 = 0;
        param_2[1] = 0;
      }
      else if (iVar1 == 1) {
        iVar1 = omp_get_max_threads();
        uVar2 = FUN_18001de80(param_1,param_2,local_ad8,iVar1);
        uVar2 = uVar2 & 0xffffffff;
      }
      else {
        uVar2 = uVar2 & 0xffffffff;
        if (iVar1 == 2) {
          FUN_18002f840(&local_a98,(undefined8 *)param_1,param_3,param_4);
          uVar2 = 0;
          local_ac8 = 0;
          local_ac0 = 0;
          local_ae8 = 0;
          local_ae0 = 0;
          local_aa8 = (int *)0x0;
          local_aa0 = 0;
          local_ab8 = 0;
          local_ab0 = 0;
          FUN_180031ac0(&local_a98,&local_ac8,(longlong)&local_ae8,&local_aa8,&local_ab8);
          if (local_a8c < 1) {
            local_a8c = local_a84 * local_a98 * ((local_a80 != 8) + 1);
          }
          if (local_a88 < 1) {
            local_a88 = local_a98 * 4;
          }
          param_2[2] = (longlong)(local_a94 * local_a8c);
          param_2[3] = (longlong)(local_a94 * local_a88);
          *param_2 = local_ac0 + local_ae0;
          param_2[1] = local_aa0 + local_ab0;
        }
      }
    }
  }
  return uVar2;
}


```

## {'procedure': 'EFI_CreateContext'}
Evidence: `ev_58b2b8783f86af1edb89a397457a4aec7b5d7d1fc5a163167b0cf15b529caff3`

```c

ulonglong EFI_CreateContext(int *param_1,longlong *param_2,ulonglong param_3,longlong *param_4,
                           ulonglong param_5,undefined8 *param_6)

{
  int iVar1;
  uint uVar2;
  ulonglong uVar3;
  undefined8 uVar4;
  longlong *plVar5;
  undefined8 in_stack_ffffffffffffffd8;
  
                    /* 0x6500  3  EFI_CreateContext */
  if (((param_1 == (int *)0x0) || (param_6 == (undefined8 *)0x0)) || (param_2 == (longlong *)0x0)) {
    uVar3 = 0x11;
  }
  else {
    plVar5 = param_4;
    uVar3 = FUN_180002a40(param_1);
    if ((int)uVar3 == 0) {
      iVar1 = param_1[0x46];
      if (iVar1 == 0) {
        uVar4 = FUN_180057f70(param_1);
        uVar2 = (uint)uVar4;
      }
      else if (iVar1 == 1) {
        uVar4 = FUN_180023a90((undefined8 *)param_1,param_2,param_3,plVar5,in_stack_ffffffffffffffd8
                              ,param_6);
        uVar2 = (uint)uVar4;
      }
      else {
        if (iVar1 != 2) {
          return uVar3 & 0xffffffff;
        }
        uVar2 = FUN_180038210((undefined8 *)param_1,param_2,param_3,param_4,param_5,param_6);
      }
      uVar3 = (ulonglong)uVar2;
    }
  }
  return uVar3;
}


```

## {'procedure': 'EFI_EndConstruct'}
Evidence: `ev_674d4ce16e371b6e87d3701fd59c19ec81f7c79add7825d7018d2dbc35b3872d`

```c

undefined8 EFI_EndConstruct(longlong *param_1,longlong param_2,longlong param_3)

{
  undefined8 uVar1;
  
                    /* 0x66a0  4  EFI_EndConstruct */
  if (((param_1 == (longlong *)0x0) || (param_2 == 0)) || (param_3 == 0)) {
    uVar1 = 0x11;
  }
  else {
    uVar1 = (**(code **)(*param_1 + 0x18))();
  }
  return uVar1;
}


```

## {'procedure': '0x180020450'}
Evidence: `ev_6ce11dafef4c2ab6f50541af640036033db7e1743cb3fa845910f88401dd2e93`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180020450(longlong param_1,Mat *param_2,Mat *param_3,Mat *param_4,Mat *param_5,Mat *param_6
                  )

{
  int *piVar1;
  int *piVar2;
  int iVar3;
  int iVar4;
  _InputOutputArray *p_Var5;
  undefined8 *puVar6;
  uint uVar7;
  uint uVar8;
  int iVar9;
  ulonglong uVar10;
  Mat *pMVar11;
  longlong lVar12;
  int *piVar13;
  int iVar14;
  longlong lVar15;
  longlong lVar16;
  ulonglong uVar17;
  ulonglong uVar18;
  double dVar19;
  float fVar20;
  undefined1 auStackY_da8 [32];
  longlong local_d30;
  int local_d28;
  undefined4 local_d18;
  int local_d14;
  undefined4 local_d10;
  undefined4 local_d0c;
  undefined8 local_d08;
  undefined8 local_d00;
  undefined8 local_cf8;
  undefined8 local_cf0;
  undefined8 local_ce8;
  longlong local_ce0;
  undefined4 *local_cd8;
  undefined8 *local_cd0;
  undefined8 local_cc8;
  undefined8 local_cc0;
  undefined4 local_cb8;
  undefined4 local_ca8;
  int local_ca4;
  undefined4 local_ca0;
  undefined4 local_c9c;
  undefined8 local_c98;
  undefined8 local_c90;
  undefined8 local_c88;
  undefined8 local_c80;
  undefined8 local_c78;
  longlong local_c70;
  undefined4 *local_c68;
  undefined8 *local_c60;
  undefined8 local_c58;
  undefined8 local_c50;
  undefined4 local_c48;
  undefined8 local_c38;
  undefined8 uStack_c30;
  undefined8 local_c28;
  undefined8 uStack_c20;
  undefined8 local_c18;
  undefined8 uStack_c10;
  undefined8 local_c08;
  longlong lStack_c00;
  undefined8 *local_bf8;
  undefined8 *local_bf0;
  undefined8 local_be8;
  undefined8 local_be0;
  undefined8 local_bd8;
  undefined8 uStack_bd0;
  undefined8 local_bc8;
  undefined8 uStack_bc0;
  undefined8 local_bb8;
  undefined8 uStack_bb0;
  undefined8 local_ba8;
  longlong lStack_ba0;
  undefined8 *local_b98;
  undefined8 *local_b90;
  undefined8 local_b88;
  undefined8 local_b80;
  longlong local_b78;
  Mat *local_b70;
  longlong local_b68;
  undefined4 local_b58;
  int local_b54;
  undefined4 local_b50;
  undefined4 local_b4c;
  undefined8 local_b48;
  undefined8 local_b40;
  undefined8 local_b38;
  undefined8 local_b30;
  undefined8 local_b28;
  longlong local_b20;
  undefined4 *local_b18;
  undefined8 *local_b10;
  undefined8 local_b08;
  undefined8 local_b00;
  undefined4 local_af8;
  int local_af4;
  undefined4 local_af0;
  undefined4 local_aec;
  undefined8 local_ae8;
  undefined8 local_ae0;
  undefined8 local_ad8;
  undefined8 local_ad0;
  undefined8 local_ac8;
  longlong local_ac0;
  undefined4 *local_ab8;
  undefined8 *local_ab0;
  undefined8 local_aa8;
  undefined8 local_aa0;
  undefined4 local_a98;
  int local_a94;
  undefined4 local_a90;
  undefined4 local_a8c;
  undefined8 local_a88;
  undefined8 local_a80;
  undefined8 local_a78;
  undefined8 local_a70;
  undefined8 local_a68;
  longlong local_a60;
  undefined4 *local_a58;
  undefined8 *local_a50;
  undefined8 local_a48;
  undefined8 local_a40;
  undefined4 local_a38;
  int local_a34;
  undefined4 local_a30;
  undefined4 local_a2c;
  undefined8 local_a28;
  undefined8 local_a20;
  undefined8 local_a18;
  undefined8 local_a10;
  undefined8 local_a08;
  longlong local_a00;
  undefined4 *local_9f8;
  undefined8 *local_9f0;
  undefined8 local_9e8;
  undefined8 local_9e0;
  undefined4 local_9d8;
  int local_9d4;
  undefined4 local_9d0;
  undefined4 local_9cc;
  undefined8 local_9c8;
  undefined8 local_9c0;
  undefined8 local_9b8;
  undefined8 local_9b0;
  undefined8 local_9a8;
  longlong local_9a0;
  undefined4 *local_998;
  undefined8 *local_990;
  undefined8 local_988;
  undefined8 local_980;
  undefined4 local_978;
  int local_974;
  undefined4 local_970;
  undefined4 local_96c;
  undefined8 local_968;
  undefined8 local_960;
  undefined8 local_958;
  undefined8 local_950;
  undefined8 local_948;
  longlong local_940;
  undefined4 *local_938;
  undefined8 *local_930;
  undefined8 local_928;
  undefined8 local_920;
  undefined4 local_918;
  int local_914;
  undefined4 local_910;
  undefined4 local_90c;
  undefined8 local_908;
  undefined8 local_900;
  undefined8 local_8f8;
  undefined8 local_8f0;
  undefined8 local_8e8;
  longlong local_8e0;
  undefined4 *local_8d8;
  undefined8 *local_8d0;
  undefined8 local_8c8;
  undefined8 local_8c0;
  undefined4 local_8b8;
  int local_8b4;
  undefined4 local_8b0;
  undefined4 local_8ac;
  undefined8 local_8a8;
  undefined8 local_8a0;
  undefined8 local_898;
  undefined8 local_890;
  undefined8 local_888;
  longlong local_880;
  undefined4 *local_878;
  undefined8 *local_870;
  undefined8 local_868;
  undefined8 local_860;
  undefined4 local_858;
  int local_854;
  undefined4 local_850;
  undefined4 local_84c;
  undefined8 local_848;
  undefined8 local_840;
  undefined8 local_838;
  undefined8 local_830;
  undefined8 local_828;
  longlong local_820;
  undefined4 *local_818;
  undefined8 *local_810;
  undefined8 local_808;
  undefined8 local_800;
  longlong local_7f8;
  undefined4 uStack_7f0;
  undefined4 uStack_7ec;
  longlong local_7e8;
  undefined4 local_7e0 [2];
  undefined8 *local_7d8;
  undefined4 local_7d0;
  undefined4 local_7cc;
  undefined4 local_7c8 [2];
  undefined8 *local_7c0;
  undefined4 local_7b8;
  undefined4 local_7b4;
  undefined4 local_7b0 [2];
  undefined8 *local_7a8;
  undefined4 local_7a0;
  undefined4 local_79c;
  undefined4 local_798 [2];
  longlong local_790;
  undefined8 local_788;
  undefined4 local_780 [2];
  undefined4 *local_778;
  undefined8 local_770;
  undefined4 local_768 [2];
  Mat *local_760;
  undefined8 local_758;
  undefined4 local_750 [2];
  longlong local_748;
  undefined8 local_740;
  undefined4 local_738 [2];
  undefined4 *local_730;
  undefined8 local_728;
  undefined4 local_720 [2];
  Mat *local_718;
  undefined8 local_710;
  undefined4 local_708 [2];
  longlong local_700;
  undefined8 local_6f8;
  undefined4 local_6f0 [2];
  undefined4 *local_6e8;
  undefined8 local_6e0;
  undefined4 local_6d8 [2];
  Mat *local_6d0;
  undefined8 local_6c8;
  undefined4 local_6c0 [2];
  longlong local_6b8;
  undefined8 local_6b0;
  undefined4 local_6a8 [2];
  longlong local_6a0;
  undefined8 local_698;
  undefined4 local_690 [2];
  Mat *local_688;
  undefined8 local_680;
  undefined4 local_678 [2];
  longlong local_670;
  undefined8 local_668;
  undefined4 local_660 [2];
  undefined4 *local_658;
  undefined8 local_650;
  undefined4 local_648 [2];
  undefined8 *local_640;
  undefined4 local_638;
  undefined4 local_634;
  undefined4 local_630 [2];
  Mat *local_628;
  undefined8 local_620;
  undefined4 local_618 [2];
  undefined8 *local_610;
  undefined4 local_608;
  undefined4 local_604;
  undefined4 local_600 [2];
  Mat *local_5f8;
  undefined8 local_5f0;
  undefined4 local_5e8 [2];
  undefined8 *local_5e0;
  undefined4 local_5d8;
  undefined4 local_5d4;
  undefined4 local_5d0 [2];
  Mat *local_5c8;
  undefined8 local_5c0;
  undefined4 local_5b8 [2];
  Mat *local_5b0;
  undefined8 local_5a8;
  longlong local_598;
  undefined4 uStack_590;
  undefined4 uStack_58c;
  undefined8 local_588;
  undefined8 local_580;
  undefined8 local_578;
  undefined8 uStack_570;
  undefined8 local_568;
  undefined8 local_560;
  undefined8 local_558;
  undefined8 uStack_550;
  longlong local_548 [2];
  undefined8 local_538;
  longlong local_528 [2];
  longlong local_518;
  undefined8 uStack_510;
  longlong local_508;
  undefined8 uStack_500;
  Mat local_4e8 [4];
  int local_4e4;
  undefined8 local_4d8;
  undefined8 local_4d0;
  undefined8 local_4c8;
  undefined8 local_4c0;
  longlong local_4b0;
  longlong local_4a8;
  undefined1 *local_4a0;
  undefined1 local_498 [16];
  Mat local_488 [4];
  int local_484;
  undefined8 local_478;
  undefined8 local_470;
  undefined8 local_468;
  undefined8 local_460;
  longlong local_450;
  longlong local_448;
  undefined1 *local_440;
  undefined1 local_438 [16];
  Mat local_428 [4];
  int local_424;
  undefined8 local_418;
  undefined8 local_410;
  undefined8 local_408;
  undefined8 local_400;
  longlong local_3f0;
  longlong local_3e8;
  undefined1 *local_3e0;
  undefined1 local_3d8 [64];
  Mat local_398 [16];
  longlong local_388;
  int *local_358;
  undefined4 *local_350;
  undefined4 local_348 [4];
  undefined4 local_338;
  undefined8 local_330;
  undefined8 uStack_328;
  undefined8 local_320;
  undefined8 uStack_318;
  undefined8 local_310;
  undefined8 uStack_308;
  undefined8 local_300;
  undefined8 uStack_2f8;
  undefined8 local_2f0;
  undefined8 uStack_2e8;
  undefined8 local_2e0;
  undefined8 uStack_2d8;
  undefined8 local_2d0;
  undefined8 uStack_2c8;
  undefined8 local_2c0;
  undefined8 uStack_2b8;
  undefined8 local_2b0;
  undefined8 uStack_2a8;
  undefined8 local_2a0;
  undefined8 uStack_298;
  undefined8 local_290;
  undefined8 uStack_288;
  undefined8 local_280;
  undefined8 uStack_278;
  undefined4 local_268;
  int local_264;
  undefined4 local_260;
  undefined4 local_25c;
  undefined8 local_258;
  undefined8 local_250;
  undefined8 local_248;
  undefined8 local_240;
  undefined8 local_238;
  longlong local_230;
  undefined4 *local_228;
  undefined8 *local_220;
  undefined8 local_218;
  undefined8 local_210;
  undefined4 local_208;
  undefined8 local_1f8;
  undefined8 uStack_1f0;
  undefined8 local_1e8;
  undefined8 uStack_1e0;
  undefined8 local_1d8;
  undefined8 uStack_1d0;
  undefined8 local_1c8;
  longlong lStack_1c0;
  undefined8 *local_1b8;
  undefined8 *local_1b0;
  undefined8 local_1a8;
  undefined8 local_1a0;
  undefined4 local_198;
  undefined8 local_188;
  undefined8 uStack_180;
  undefined8 local_178;
  undefined8 uStack_170;
  undefined8 local_168;
  undefined8 uStack_160;
  undefined8 local_158;
  longlong lStack_150;
  undefined8 *local_148;
  undefined8 *local_140;
  undefined8 local_138;
  undefined8 local_130;
  undefined4 local_128;
  undefined4 local_118;
  int local_114;
  undefined4 local_110;
  undefined4 local_10c;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  undefined8 local_f0;
  undefined8 local_e8;
  longlong local_e0;
  undefined4 *local_d8;
  undefined8 *local_d0;
  undefined8 local_c8;
  undefined8 local_c0;
  undefined4 local_b8;
  longlong *local_a8;
  undefined8 local_a0;
  longlong local_98;
  longlong local_90;
  undefined8 local_88;
  undefined8 local_78;
  undefined4 uStack_70;
  undefined8 local_68;
  ulonglong local_60;
  
  local_538 = 0xfffffffffffffffe;
  local_60 = DAT_18008b080 ^ (ulonglong)auStackY_da8;
  local_d28 = *(int *)(param_1 + 0x844);
  iVar3 = *(int *)(param_1 + 0x80);
  iVar14 = *(int *)(param_1 + 0x1c);
  iVar4 = *(int *)(param_1 + 0x4e4);
  uVar7 = *(uint *)(param_1 + 0x4e8);
  local_7e8 = *(longlong *)(param_1 + 0x848);
  local_68 = *(longlong *)(param_1 + 0xb00);
  local_7f8 = *(longlong *)(param_1 + 0xb08);
  lVar16 = *(longlong *)(param_1 + 0xae8);
  local_b68 = *(longlong *)(param_1 + 0xaf0);
  lVar12 = *(longlong *)(param_1 + 0xaf8);
  local_88 = 0;
  local_a8 = (longlong *)(param_1 + 0x808);
  local_a0 = 0x20;
  local_98 = 0;
  local_90 = *(longlong *)(param_1 + 0x810) + *local_a8;
  local_b70 = param_4;
  FUN_18002adc0((longlong *)&local_a8,param_1 + 8,&local_d30,&local_b78);
  local_290 = 0;
  uStack_288 = 0;
  local_280 = 0;
  uStack_278 = 0;
  local_618[0] = 0xc1020006;
  local_610 = &local_290;
  local_608 = 1;
  local_604 = 4;
  p_Var5 = cv::noArray();
  cv::Mat::setTo(param_5,(_InputArray *)local_618,(_InputArray *)p_Var5);
  local_2b0 = 0;
  uStack_2a8 = 0;
  local_2a0 = 0;
  uStack_298 = 0;
  local_5e8[0] = 0xc1020006;
  local_5e0 = &local_2b0;
  local_5d8 = 1;
  local_5d4 = 4;
  p_Var5 = cv::noArray();
  cv::Mat::setTo(param_4,(_InputArray *)local_5e8,(_InputArray *)p_Var5);
  FUN_1800013f0(local_398,param_6);
  local_338 = *(undefined4 *)(param_6 + 0x60);
  fVar20 = 1.0 / (float)*(int *)(param_1 + 0x844);
  local_78 = local_388;
  uStack_70 = local_348[0];
  if (iVar14 == 1) {
    local_548[0] = local_388;
    FUN_180013c90(local_358[1],*local_358,local_548,fVar20);
  }
  else {
    local_528[0] = local_388;
    FUN_180013d90(local_358[1],*local_358,local_528,fVar20);
  }
  FUN_180001900(local_398);
  if (local_350 != local_348) {
    cv::fastFree(local_350);
  }
  piVar13 = (int *)(param_1 + 0x238);
  local_78 = 0;
  uVar18 = (ulonglong)((longlong)(piVar13 + *(int *)(param_1 + 0x248)) + (3 - (longlong)piVar13)) >>
           2;
  if (piVar13 + *(int *)(param_1 + 0x248) < piVar13) {
    uVar18 = 0;
  }
  if (uVar18 != 0) {
    do {
      local_5a8 = 0;
      local_5b8[0] = 0x2010000;
      local_5b0 = param_5;
      local_5f0 = 0;
      local_600[0] = 0x1010000;
      local_5f8 = param_5;
      local_668 = 0;
      local_678[0] = 0x1010000;
      lVar15 = (longlong)*piVar13;
      local_670 = local_68 + lVar15 * 0x68;
      cv::scaleAdd((_InputArray *)local_678,*(double *)(param_1 + 0x40 + lVar15 * 8),
                   (_InputArray *)local_600,(_OutputArray *)local_5b8);
      local_710 = 0;
      local_720[0] = 0x2010000;
      local_680 = 0;
      local_690[0] = 0x1010000;
      local_788 = 0;
      local_798[0] = 0x1010000;
      local_790 = local_7f8 + lVar15 * 0x68;
      local_718 = param_4;
      local_688 = param_4;
      cv::scaleAdd((_InputArray *)local_798,*(double *)(param_1 + 0x40 + lVar15 * 8),
                   (_InputArray *)local_690,(_OutputArray *)local_720);
      piVar13 = piVar13 + 1;
      local_78 = local_78 + 1;
    } while (local_78 != uVar18);
  }
  uVar18 = 0;
  if (*(char *)(param_1 + 0x124) == '\x01') {
    local_8b8 = *(undefined4 *)param_4;
    local_8b4 = *(int *)(param_4 + 4);
    local_8b0 = *(undefined4 *)(param_4 + 8);
    local_8ac = *(undefined4 *)(param_4 + 0xc);
    local_8a8 = *(undefined8 *)(param_4 + 0x10);
    local_8a0 = *(undefined8 *)(param_4 + 0x18);
    local_898 = *(undefined8 *)(param_4 + 0x20);
    local_890 = *(undefined8 *)(param_4 + 0x28);
    local_888 = *(undefined8 *)(param_4 + 0x30);
    local_880 = *(longlong *)(param_4 + 0x38);
    local_878 = &local_8b0;
    local_870 = &local_868;
    local_860 = 0;
    local_868 = 0;
    if (local_880 != 0) {
      LOCK();
      *(int *)(local_880 + 0x14) = *(int *)(local_880 + 0x14) + 1;
      UNLOCK();
    }
    if (*(int *)(param_4 + 4) < 3) {
      local_868 = **(undefined8 **)(param_4 + 0x48);
      local_860 = *(undefined8 *)(*(longlong *)(param_4 + 0x48) + 8);
    }
    else {
      local_8b4 = 0;
      cv::Mat::copySize((Mat *)&local_8b8,param_4);
    }
    FUN_180005740(param_1 + 8,(Mat *)&local_8b8,1);
    if (local_880 != 0) {
      LOCK();
      piVar13 = (int *)(local_880 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_8b8);
      }
    }
    local_880 = 0;
    local_8a8 = 0;
    local_890 = 0;
    local_898 = 0;
    local_8a0 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_8b4) {
      do {
        *(undefined4 *)((longlong)local_878 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_8b4);
    }
    if (local_870 != &local_868) {
      cv::fastFree(local_870);
    }
    local_a38 = *(undefined4 *)param_5;
    local_a34 = *(int *)(param_5 + 4);
    local_a30 = *(undefined4 *)(param_5 + 8);
    local_a2c = *(undefined4 *)(param_5 + 0xc);
    local_a28 = *(undefined8 *)(param_5 + 0x10);
    local_a20 = *(undefined8 *)(param_5 + 0x18);
    local_a18 = *(undefined8 *)(param_5 + 0x20);
    local_a10 = *(undefined8 *)(param_5 + 0x28);
    local_a08 = *(undefined8 *)(param_5 + 0x30);
    local_a00 = *(longlong *)(param_5 + 0x38);
    local_9f8 = &local_a30;
    local_9f0 = &local_9e8;
    local_9e0 = 0;
    local_9e8 = 0;
    if (local_a00 != 0) {
      LOCK();
      *(int *)(local_a00 + 0x14) = *(int *)(local_a00 + 0x14) + 1;
      UNLOCK();
    }
    if (*(int *)(param_5 + 4) < 3) {
      local_9e8 = **(undefined8 **)(param_5 + 0x48);
      local_9e0 = *(undefined8 *)(*(longlong *)(param_5 + 0x48) + 8);
    }
    else {
      local_a34 = 0;
      cv::Mat::copySize((Mat *)&local_a38,param_5);
    }
    FUN_180004b70(param_1 + 8,param_1 + 0x840,(Mat *)&local_a38,9);
    if (local_a00 != 0) {
      LOCK();
      piVar13 = (int *)(local_a00 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_a38);
      }
    }
    local_a00 = 0;
    local_a28 = 0;
    local_a10 = 0;
    local_a18 = 0;
    local_a20 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_a34) {
      do {
        *(undefined4 *)((longlong)local_9f8 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_a34);
    }
    if (local_9f0 != &local_9e8) {
      cv::fastFree(local_9f0);
    }
    local_b58 = *(undefined4 *)param_6;
    local_b54 = *(int *)(param_6 + 4);
    local_b50 = *(undefined4 *)(param_6 + 8);
    local_b4c = *(undefined4 *)(param_6 + 0xc);
    local_b48 = *(undefined8 *)(param_6 + 0x10);
    local_b40 = *(undefined8 *)(param_6 + 0x18);
    local_b38 = *(undefined8 *)(param_6 + 0x20);
    local_b30 = *(undefined8 *)(param_6 + 0x28);
    local_b28 = *(undefined8 *)(param_6 + 0x30);
    local_b20 = *(longlong *)(param_6 + 0x38);
    local_b18 = &local_b50;
    local_b10 = &local_b08;
    local_b00 = 0;
    local_b08 = 0;
    if (local_b20 != 0) {
      LOCK();
      *(int *)(local_b20 + 0x14) = *(int *)(local_b20 + 0x14) + 1;
      UNLOCK();
    }
    if (*(int *)(param_6 + 4) < 3) {
      local_b08 = **(undefined8 **)(param_6 + 0x48);
      local_b00 = *(undefined8 *)(*(longlong *)(param_6 + 0x48) + 8);
    }
    else {
      local_b54 = 0;
      cv::Mat::copySize((Mat *)&local_b58,param_6);
    }
    FUN_180005740(param_1 + 8,(Mat *)&local_b58,0);
    if (local_b20 != 0) {
      LOCK();
      piVar13 = (int *)(local_b20 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_b58);
      }
    }
    local_b20 = 0;
    local_b48 = 0;
    local_b30 = 0;
    local_b38 = 0;
    local_b40 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_b54) {
      do {
        *(undefined4 *)((longlong)local_b18 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_b54);
    }
    if (local_b10 != &local_b08) {
      cv::fastFree(local_b10);
    }
  }
  if (*(char *)(param_1 + 0x124) == '\x01') {
    memset(&local_268,0,0x68);
    local_588 = 0;
    local_578 = 0;
    uStack_570 = 0;
    local_580 = 0;
    local_bd8 = 0x42ff0000;
    uStack_bd0 = 0;
    local_bc8 = 0;
    uStack_bc0 = 0;
    local_bb8 = 0;
    uStack_bb0 = 0;
    local_ba8 = 0;
    lStack_ba0 = 0;
    local_b98 = &uStack_bd0;
    local_b90 = &local_b88;
    local_b80 = 0;
    local_b88 = 0;
    local_78 = CONCAT44(iVar4,uVar7);
    cv::Mat::create((Mat *)&local_bd8,2,(int *)&local_78,5);
    cv::Mat::operator=((Mat *)&local_bd8,(Scalar_<double> *)&local_588);
    local_ca8 = (undefined4)local_bd8;
    local_ca4 = local_bd8._4_4_;
    local_ca0 = (undefined4)uStack_bd0;
    local_c9c = uStack_bd0._4_4_;
    local_c98 = local_bc8;
    local_c90 = uStack_bc0;
    local_c88 = local_bb8;
    local_c80 = uStack_bb0;
    local_c78 = local_ba8;
    local_c70 = lStack_ba0;
    local_c68 = &local_ca0;
    local_c60 = &local_c58;
    local_c50 = 0;
    local_c58 = 0;
    if (lStack_ba0 != 0) {
      LOCK();
      *(int *)(lStack_ba0 + 0x14) = *(int *)(lStack_ba0 + 0x14) + 1;
      UNLOCK();
    }
    if (local_bd8._4_4_ < 3) {
      local_c58 = *local_b90;
      local_c50 = local_b90[1];
    }
    else {
      local_ca4 = 0;
      cv::Mat::copySize((Mat *)&local_ca8,(Mat *)&local_bd8);
    }
    local_c48 = 0;
    if (lStack_ba0 != 0) {
      LOCK();
      piVar13 = (int *)(lStack_ba0 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_bd8);
      }
    }
    lStack_ba0 = 0;
    local_bc8 = 0;
    uStack_bb0 = 0;
    local_bb8 = 0;
    uStack_bc0 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_bd8._4_4_) {
      do {
        *(undefined4 *)((longlong)local_b98 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_bd8._4_4_);
    }
    if (local_b90 != &local_b88) {
      cv::fastFree(local_b90);
    }
    local_268 = local_ca8;
    local_264 = local_ca4;
    local_260 = local_ca0;
    local_25c = local_c9c;
    local_258 = local_c98;
    local_250 = local_c90;
    local_248 = local_c88;
    local_240 = local_c80;
    local_238 = local_c78;
    local_230 = local_c70;
    local_228 = &local_260;
    local_220 = &local_218;
    local_210 = 0;
    local_218 = 0;
    if (local_c70 != 0) {
      LOCK();
      *(int *)(local_c70 + 0x14) = *(int *)(local_c70 + 0x14) + 1;
      UNLOCK();
    }
    if (local_ca4 < 3) {
      local_218 = *local_c60;
      local_210 = local_c60[1];
    }
    else {
      local_264 = 0;
      cv::Mat::copySize((Mat *)&local_268,(Mat *)&local_ca8);
    }
    local_208 = local_c48;
    if (local_c70 != 0) {
      LOCK();
      piVar13 = (int *)(local_c70 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_ca8);
      }
    }
    local_c70 = 0;
    local_c98 = 0;
    local_c80 = 0;
    local_c88 = 0;
    local_c90 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_ca4) {
      do {
        *(undefined4 *)((longlong)local_c68 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_ca4);
    }
    if (local_c60 != &local_c58) {
      cv::fastFree(local_c60);
    }
    memset(&local_118,0,0x68);
    local_568 = 0;
    local_558 = 0;
    uStack_550 = 0;
    local_560 = 0;
    local_c38 = 0x42ff0000;
    uStack_c30 = 0;
    local_c28 = 0;
    uStack_c20 = 0;
    local_c18 = 0;
    uStack_c10 = 0;
    local_c08 = 0;
    lStack_c00 = 0;
    local_bf8 = &uStack_c30;
    local_bf0 = &local_be8;
    local_be0 = 0;
    local_be8 = 0;
    local_68 = CONCAT44(iVar4,uVar7);
    cv::Mat::create((Mat *)&local_c38,2,(int *)&local_68,5);
    cv::Mat::operator=((Mat *)&local_c38,(Scalar_<double> *)&local_568);
    local_d18 = (undefined4)local_c38;
    local_d14 = local_c38._4_4_;
    local_d10 = (undefined4)uStack_c30;
    local_d0c = uStack_c30._4_4_;
    local_d08 = local_c28;
    local_d00 = uStack_c20;
    local_cf8 = local_c18;
    local_cf0 = uStack_c10;
    local_ce8 = local_c08;
    local_ce0 = lStack_c00;
    local_cd8 = &local_d10;
    local_cd0 = &local_cc8;
    local_cc0 = 0;
    local_cc8 = 0;
    if (lStack_c00 != 0) {
      LOCK();
      *(int *)(lStack_c00 + 0x14) = *(int *)(lStack_c00 + 0x14) + 1;
      UNLOCK();
    }
    if (local_c38._4_4_ < 3) {
      local_cc8 = *local_bf0;
      local_cc0 = local_bf0[1];
    }
    else {
      local_d14 = 0;
      cv::Mat::copySize((Mat *)&local_d18,(Mat *)&local_c38);
    }
    local_cb8 = 0;
    if (lStack_c00 != 0) {
      LOCK();
      piVar13 = (int *)(lStack_c00 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_c38);
      }
    }
    lStack_c00 = 0;
    local_c28 = 0;
    uStack_c10 = 0;
    local_c18 = 0;
    uStack_c20 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_c38._4_4_) {
      do {
        *(undefined4 *)((longlong)local_bf8 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_c38._4_4_);
    }
    if (local_bf0 != &local_be8) {
      cv::fastFree(local_bf0);
    }
    local_118 = local_d18;
    local_114 = local_d14;
    local_110 = local_d10;
    local_10c = local_d0c;
    local_108 = local_d08;
    local_100 = local_d00;
    local_f8 = local_cf8;
    local_f0 = local_cf0;
    local_e8 = local_ce8;
    local_e0 = local_ce0;
    local_d8 = &local_110;
    local_d0 = &local_c8;
    local_c0 = 0;
    local_c8 = 0;
    if (local_ce0 != 0) {
      LOCK();
      *(int *)(local_ce0 + 0x14) = *(int *)(local_ce0 + 0x14) + 1;
      UNLOCK();
    }
    if (local_d14 < 3) {
      local_c8 = *local_cd0;
      local_c0 = local_cd0[1];
    }
    else {
      local_114 = 0;
      cv::Mat::copySize((Mat *)&local_118,(Mat *)&local_d18);
    }
    local_b8 = local_cb8;
    if (local_ce0 != 0) {
      LOCK();
      piVar13 = (int *)(local_ce0 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_d18);
      }
    }
    local_ce0 = 0;
    local_d08 = 0;
    local_cf0 = 0;
    local_cf8 = 0;
    local_d00 = 0;
    uVar17 = uVar18;
    if (0 < local_d14) {
      do {
        *(undefined4 *)((longlong)local_cd8 + uVar18) = 0;
        uVar8 = (int)uVar17 + 1;
        uVar18 = uVar18 + 4;
        uVar17 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_d14);
    }
    if (local_cd0 != &local_cc8) {
      cv::fastFree(local_cd0);
    }
    piVar13 = (int *)(param_1 + 0x24c);
    uVar18 = (ulonglong)((longlong)(piVar13 + *(int *)(param_1 + 0x25c)) + (3 - (longlong)piVar13))
             >> 2;
    if (piVar13 + *(int *)(param_1 + 0x25c) < piVar13) {
      uVar18 = 0;
    }
    if (uVar18 != 0) {
      uVar17 = 0;
      do {
        local_770 = 0;
        local_780[0] = 0x2010000;
        local_778 = &local_268;
        local_6e0 = 0;
        local_6f0[0] = 0x1010000;
        local_6e8 = &local_268;
        local_6b0 = 0;
        local_6c0[0] = 0x1010000;
        lVar15 = (longlong)*piVar13;
        local_6b8 = lVar15 * 0x68 + lVar16;
        cv::scaleAdd((_InputArray *)local_6c0,*(double *)(param_1 + 0x60 + lVar15 * 8),
                     (_InputArray *)local_6f0,(_OutputArray *)local_780);
        local_728 = 0;
        local_738[0] = 0x2010000;
        local_730 = &local_118;
        local_650 = 0;
        local_660[0] = 0x1010000;
        local_658 = &local_118;
        local_6f8 = 0;
        local_708[0] = 0x1010000;
        local_700 = local_b68 + lVar15 * 0x68;
        cv::scaleAdd((_InputArray *)local_708,*(double *)(param_1 + 0x60 + lVar15 * 8),
                     (_InputArray *)local_660,(_OutputArray *)local_738);
        piVar13 = piVar13 + 1;
        uVar17 = uVar17 + 1;
        param_4 = local_b70;
      } while (uVar17 != uVar18);
    }
    memset(&local_1f8,0,0x68);
    local_1f8 = 0x42ff0000;
    uStack_1f0 = 0;
    local_1e8 = 0;
    uStack_1e0 = 0;
    local_1d8 = 0;
    uStack_1d0 = 0;
    local_1c8 = 0;
    lStack_1c0 = 0;
    local_1b8 = &uStack_1f0;
    local_1b0 = &local_1a8;
    local_1a0 = 0;
    local_1a8 = 0;
    local_198 = 0;
    lVar15 = param_1 + 8;
    FUN_180009ec0(lVar15,lVar16,iVar4,uVar7,(Mat *)&local_1f8);
    local_978 = (undefined4)local_1f8;
    local_974 = local_1f8._4_4_;
    local_970 = (undefined4)uStack_1f0;
    local_96c = uStack_1f0._4_4_;
    local_968 = local_1e8;
    local_960 = uStack_1e0;
    local_958 = local_1d8;
    local_950 = uStack_1d0;
    local_948 = local_1c8;
    local_940 = lStack_1c0;
    local_938 = &local_970;
    local_930 = &local_928;
    iVar14 = 0;
    uVar18 = 0;
    local_920 = 0;
    local_928 = 0;
    if (lStack_1c0 != 0) {
      LOCK();
      *(int *)(lStack_1c0 + 0x14) = *(int *)(lStack_1c0 + 0x14) + 1;
      UNLOCK();
    }
    if (local_1f8._4_4_ < 3) {
      local_928 = *local_1b0;
      local_920 = local_1b0[1];
    }
    else {
      local_974 = 0;
      cv::Mat::copySize((Mat *)&local_978,(Mat *)&local_1f8);
    }
    FUN_180004b70(lVar15,param_1 + 0x840,(Mat *)&local_978,0);
    if (local_940 != 0) {
      LOCK();
      piVar13 = (int *)(local_940 + 0x14);
      iVar9 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar9 == 1) {
        cv::Mat::deallocate((Mat *)&local_978);
      }
    }
    local_940 = 0;
    local_968 = 0;
    local_950 = 0;
    local_958 = 0;
    local_960 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_974) {
      do {
        *(undefined4 *)((longlong)local_938 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_974);
    }
    if (local_930 != &local_928) {
      cv::fastFree(local_930);
    }
    local_af8 = local_268;
    local_af4 = local_264;
    local_af0 = local_260;
    local_aec = local_25c;
    local_ae8 = local_258;
    local_ae0 = local_250;
    local_ad8 = local_248;
    local_ad0 = local_240;
    local_ac8 = local_238;
    local_ac0 = local_230;
    local_ab8 = &local_af0;
    local_ab0 = &local_aa8;
    local_aa0 = 0;
    local_aa8 = 0;
    if (local_230 != 0) {
      LOCK();
      *(int *)(local_230 + 0x14) = *(int *)(local_230 + 0x14) + 1;
      UNLOCK();
    }
    if (local_264 < 3) {
      local_aa8 = *local_220;
      local_aa0 = local_220[1];
    }
    else {
      local_af4 = 0;
      cv::Mat::copySize((Mat *)&local_af8,(Mat *)&local_268);
    }
    FUN_180004b70(lVar15,param_1 + 0x840,(Mat *)&local_af8,2);
    if (local_ac0 != 0) {
      LOCK();
      piVar13 = (int *)(local_ac0 + 0x14);
      iVar9 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar9 == 1) {
        cv::Mat::deallocate((Mat *)&local_af8);
      }
    }
    local_ac0 = 0;
    local_ae8 = 0;
    local_ad0 = 0;
    local_ad8 = 0;
    local_ae0 = 0;
    uVar17 = uVar18;
    uVar10 = uVar18;
    if (0 < local_af4) {
      do {
        *(undefined4 *)((longlong)local_ab8 + uVar17) = 0;
        uVar8 = (int)uVar10 + 1;
        uVar17 = uVar17 + 4;
        uVar10 = (ulonglong)uVar8;
      } while ((int)uVar8 < local_af4);
    }
    if (local_ab0 != &local_aa8) {
      cv::fastFree(local_ab0);
    }
    if (lStack_1c0 != 0) {
      LOCK();
      piVar13 = (int *)(lStack_1c0 + 0x14);
      iVar9 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar9 == 1) {
        cv::Mat::deallocate((Mat *)&local_1f8);
      }
    }
    lStack_1c0 = 0;
    local_1e8 = 0;
    uStack_1d0 = 0;
    local_1d8 = 0;
    uStack_1e0 = 0;
    uVar17 = uVar18;
    iVar9 = iVar14;
    if (0 < local_1f8._4_4_) {
      do {
        *(undefined4 *)((longlong)local_1b8 + uVar17) = 0;
        iVar9 = iVar9 + 1;
        uVar17 = uVar17 + 4;
      } while (iVar9 < local_1f8._4_4_);
    }
    if (local_1b0 != &local_1a8) {
      cv::fastFree(local_1b0);
    }
    if (local_e0 != 0) {
      LOCK();
      piVar13 = (int *)(local_e0 + 0x14);
      iVar9 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar9 == 1) {
        cv::Mat::deallocate((Mat *)&local_118);
      }
    }
    local_e0 = 0;
    local_108 = 0;
    local_f0 = 0;
    local_f8 = 0;
    local_100 = 0;
    uVar17 = uVar18;
    if (0 < local_114) {
      do {
        *(undefined4 *)((longlong)local_d8 + uVar17) = 0;
        iVar14 = iVar14 + 1;
        uVar17 = uVar17 + 4;
      } while (iVar14 < local_114);
    }
    if (local_d0 != &local_c8) {
      cv::fastFree(local_d0);
    }
    if (local_230 != 0) {
      LOCK();
      piVar13 = (int *)(local_230 + 0x14);
      iVar14 = *piVar13;
      *piVar13 = *piVar13 + -1;
      UNLOCK();
      if (iVar14 == 1) {
        cv::Mat::deallocate((Mat *)&local_268);
      }
    }
    local_230 = 0;
    local_258 = 0;
    local_240 = 0;
    local_248 = 0;
    local_250 = 0;
    iVar14 = 0;
    if (0 < local_264) {
      do {
        *(undefined4 *)((longlong)local_228 + uVar18) = 0;
        iVar14 = iVar14 + 1;
        uVar18 = uVar18 + 4;
      } while (iVar14 < local_264);
    }
    if (local_220 != &local_218) {
      cv::fastFree(local_220);
    }
  }
  if ((local_d28 < iVar3) || (iVar3 < 2)) {
    FUN_180001d80(&DAT_18007bf60,0,0,0,0);
    local_2f0 = 0;
    uStack_2e8 = 0;
    local_2e0 = 0;
    uStack_2d8 = 0;
    local_7b0[0] = 0xc1020006;
    local_7a8 = &local_2f0;
    local_7a0 = 1;
    local_79c = 4;
    p_Var5 = cv::noArray();
    cv::Mat::setTo(param_2,(_InputArray *)local_7b0,(_InputArray *)p_Var5);
    local_310 = 0;
    uStack_308 = 0;
    local_300 = 0;
    uStack_2f8 = 0;
    local_7c8[0] = 0xc1020006;
    local_7c0 = &local_310;
    local_7b8 = 1;
    local_7b4 = 4;
    p_Var5 = cv::noArray();
    cv::Mat::setTo(param_3,(_InputArray *)local_7c8,(_InputArray *)p_Var5);
    piVar13 = (int *)(param_1 + 0x24c);
    piVar2 = piVar13 + *(int *)(param_1 + 0x25c);
    for (; piVar13 != piVar2; piVar13 = piVar13 + 1) {
      lVar12 = (longlong)*piVar13 * 0x68;
      local_7f8 = *(longlong *)(lVar12 + 0x10 + lVar16);
      uStack_7f0 = (undefined4)*(undefined8 *)(lVar12 + 0x50 + lVar16);
      uStack_500 = CONCAT44(uStack_7ec,uStack_7f0);
      local_598 = *(longlong *)(param_2 + 0x10);
      uStack_590 = (undefined4)*(undefined8 *)(param_2 + 0x50);
      uStack_510 = CONCAT44(uStack_58c,uStack_590);
      pMVar11 = (Mat *)(ulonglong)uVar7;
      local_518 = local_598;
      local_508 = local_7f8;
      dVar19 = (double)FUN_180014f30(iVar4,uVar7,
                                     (float)*(double *)(param_1 + 0x60 + (longlong)*piVar13 * 8),
                                     local_7e8,&local_518,&local_508);
      puVar6 = (undefined8 *)cv::operator*(dVar19,pMVar11);
      (**(code **)(*(longlong *)*puVar6 + 0x28))((longlong *)*puVar6,puVar6,param_3);
      if (local_3f0 != 0) {
        LOCK();
        piVar1 = (int *)(local_3f0 + 0x14);
        iVar3 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate(local_428);
        }
      }
      uVar18 = 0;
      local_3f0 = 0;
      local_418 = 0;
      local_400 = 0;
      local_408 = 0;
      local_410 = 0;
      uVar17 = uVar18;
      uVar10 = uVar18;
      if (0 < local_424) {
        do {
          *(undefined4 *)(local_3e8 + uVar17) = 0;
          uVar8 = (int)uVar10 + 1;
          uVar17 = uVar17 + 4;
          uVar10 = (ulonglong)uVar8;
        } while ((int)uVar8 < local_424);
      }
      if (local_3e0 != local_3d8) {
        cv::fastFree(local_3e0);
      }
      if (local_450 != 0) {
        LOCK();
        piVar1 = (int *)(local_450 + 0x14);
        iVar3 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate(local_488);
        }
      }
      local_450 = 0;
      local_478 = 0;
      local_460 = 0;
      local_468 = 0;
      local_470 = 0;
      uVar17 = uVar18;
      uVar10 = uVar18;
      if (0 < local_484) {
        do {
          *(undefined4 *)(local_448 + uVar17) = 0;
          uVar8 = (int)uVar10 + 1;
          uVar17 = uVar17 + 4;
          uVar10 = (ulonglong)uVar8;
        } while ((int)uVar8 < local_484);
      }
      if (local_440 != local_438) {
        cv::fastFree(local_440);
      }
      if (local_4b0 != 0) {
        LOCK();
        piVar1 = (int *)(local_4b0 + 0x14);
        iVar3 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate(local_4e8);
        }
      }
      local_4b0 = 0;
      local_4d8 = 0;
      local_4c0 = 0;
      local_4c8 = 0;
      local_4d0 = 0;
      uVar17 = uVar18;
      if (0 < local_4e4) {
        do {
          *(undefined4 *)(local_4a8 + uVar18) = 0;
          uVar8 = (int)uVar17 + 1;
          uVar18 = uVar18 + 4;
          uVar17 = (ulonglong)uVar8;
        } while ((int)uVar8 < local_4e4);
      }
      if (local_4a0 != local_498) {
        cv::fastFree(local_4a0);
      }
    }
    if (*(char *)(param_1 + 0x124) == '\x01') {
      FUN_180004b70(param_1 + 8,param_1 + 0x840,param_3,4);
    }
  }
  else {
    FUN_180017560(param_1 + 8,*(longlong *)(param_1 + 0x870),*(longlong *)(param_1 + 0xaf8),
                  *(longlong *)(param_1 + 0xb10),*(longlong *)(param_1 + 0xb18));
    lVar15 = local_d30;
    FUN_180014890(param_1 + 8,(longlong *)(param_1 + 0x808),(int *)(param_1 + 0x840),lVar16,lVar12,
                  local_d30,local_b78);
    local_330 = 0;
    uStack_328 = 0;
    local_320 = 0;
    uStack_318 = 0;
    local_7e0[0] = 0xc1020006;
    local_7d8 = &local_330;
    local_7d0 = 1;
    local_7cc = 4;
    p_Var5 = cv::noArray();
    cv::Mat::setTo(param_2,(_InputArray *)local_7e0,(_InputArray *)p_Var5);
    local_2d0 = 0;
    uStack_2c8 = 0;
    local_2c0 = 0;
    uStack_2b8 = 0;
    local_648[0] = 0xc1020006;
    local_640 = &local_2d0;
    local_638 = 1;
    local_634 = 4;
    p_Var5 = cv::noArray();
    cv::Mat::setTo(param_3,(_InputArray *)local_648,(_InputArray *)p_Var5);
    piVar13 = (int *)(param_1 + 0x24c);
    uVar17 = 0;
    uVar18 = (ulonglong)((longlong)(piVar13 + *(int *)(param_1 + 0x25c)) + (3 - (longlong)piVar13))
             >> 2;
    if (piVar13 + *(int *)(param_1 + 0x25c) < piVar13) {
      uVar18 = uVar17;
    }
    lVar16 = lVar15;
    if (uVar18 != 0) {
      do {
        local_6c8 = 0;
        local_6d8[0] = 0x2010000;
        local_620 = 0;
        local_630[0] = 0x1010000;
        lVar16 = (longlong)*piVar13;
        local_6a0 = lVar16 * 0x68 + lVar15;
        local_698 = 0;
        local_6a8[0] = 0x1010000;
        local_6d0 = param_2;
        local_628 = param_2;
        cv::scaleAdd((_InputArray *)local_6a8,*(double *)(param_1 + 0x60 + lVar16 * 8),
                     (_InputArray *)local_630,(_OutputArray *)local_6d8);
        local_5c0 = 0;
        local_5d0[0] = 0x2010000;
        local_758 = 0;
        local_768[0] = 0x1010000;
        local_748 = local_b78 + lVar16 * 0x68;
        local_740 = 0;
        local_750[0] = 0x1010000;
        local_760 = param_3;
        local_5c8 = param_3;
        cv::scaleAdd((_InputArray *)local_750,*(double *)(param_1 + 0x60 + lVar16 * 8),
                     (_InputArray *)local_768,(_OutputArray *)local_5d0);
        piVar13 = piVar13 + 1;
        uVar17 = uVar17 + 1;
        lVar16 = local_d30;
        param_4 = local_b70;
      } while (uVar17 != uVar18);
    }
    lVar12 = param_1 + 0x840;
    if (*(char *)(param_1 + 0x124) == '\x01') {
      memset(&local_188,0,0x68);
      local_188 = 0x42ff0000;
      uStack_180 = 0;
      local_178 = 0;
      uStack_170 = 0;
      local_168 = 0;
      uStack_160 = 0;
      local_158 = 0;
      lStack_150 = 0;
      local_148 = &uStack_180;
      local_140 = &local_138;
      local_130 = 0;
      local_138 = 0;
      local_128 = 0;
      lVar15 = param_1 + 8;
      FUN_180009ec0(lVar15,lVar16,iVar4,uVar7,(Mat *)&local_188);
      local_a98 = (undefined4)local_188;
      local_a94 = local_188._4_4_;
      local_a90 = (undefined4)uStack_180;
      local_a8c = uStack_180._4_4_;
      local_a88 = local_178;
      local_a80 = uStack_170;
      local_a78 = local_168;
      local_a70 = uStack_160;
      local_a68 = local_158;
      local_a60 = lStack_150;
      local_a58 = &local_a90;
      local_a50 = &local_a48;
      uVar18 = 0;
      local_a40 = 0;
      local_a48 = 0;
      if (lStack_150 != 0) {
        LOCK();
        *(int *)(lStack_150 + 0x14) = *(int *)(lStack_150 + 0x14) + 1;
        UNLOCK();
      }
      if (local_188._4_4_ < 3) {
        local_a48 = *local_140;
        local_a40 = local_140[1];
      }
      else {
        local_a94 = 0;
        cv::Mat::copySize((Mat *)&local_a98,(Mat *)&local_188);
      }
      FUN_180004b70(lVar15,lVar12,(Mat *)&local_a98,1);
      if (local_a60 != 0) {
        LOCK();
        piVar13 = (int *)(local_a60 + 0x14);
        iVar3 = *piVar13;
        *piVar13 = *piVar13 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate((Mat *)&local_a98);
        }
      }
      local_a60 = 0;
      local_a88 = 0;
      local_a70 = 0;
      local_a78 = 0;
      local_a80 = 0;
      uVar17 = uVar18;
      uVar10 = uVar18;
      if (0 < local_a94) {
        do {
          *(undefined4 *)(uVar17 + (longlong)local_a58) = 0;
          uVar8 = (int)uVar10 + 1;
          uVar17 = uVar17 + 4;
          uVar10 = (ulonglong)uVar8;
        } while ((int)uVar8 < local_a94);
      }
      if (local_a50 != &local_a48) {
        cv::fastFree(local_a50);
      }
      local_9d8 = *(undefined4 *)param_2;
      local_9d4 = *(int *)(param_2 + 4);
      local_9d0 = *(undefined4 *)(param_2 + 8);
      local_9cc = *(undefined4 *)(param_2 + 0xc);
      local_9c8 = *(undefined8 *)(param_2 + 0x10);
      local_9c0 = *(undefined8 *)(param_2 + 0x18);
      local_9b8 = *(undefined8 *)(param_2 + 0x20);
      local_9b0 = *(undefined8 *)(param_2 + 0x28);
      local_9a8 = *(undefined8 *)(param_2 + 0x30);
      local_9a0 = *(longlong *)(param_2 + 0x38);
      local_998 = &local_9d0;
      local_990 = &local_988;
      local_980 = 0;
      local_988 = 0;
      if (local_9a0 != 0) {
        LOCK();
        *(int *)(local_9a0 + 0x14) = *(int *)(local_9a0 + 0x14) + 1;
        UNLOCK();
      }
      if (*(int *)(param_2 + 4) < 3) {
        local_988 = **(undefined8 **)(param_2 + 0x48);
        local_980 = *(undefined8 *)(*(longlong *)(param_2 + 0x48) + 8);
      }
      else {
        local_9d4 = 0;
        cv::Mat::copySize((Mat *)&local_9d8,param_2);
      }
      FUN_180004b70(lVar15,lVar12,(Mat *)&local_9d8,3);
      if (local_9a0 != 0) {
        LOCK();
        piVar13 = (int *)(local_9a0 + 0x14);
        iVar3 = *piVar13;
        *piVar13 = *piVar13 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate((Mat *)&local_9d8);
        }
      }
      local_9a0 = 0;
      local_9c8 = 0;
      local_9b0 = 0;
      local_9b8 = 0;
      local_9c0 = 0;
      uVar17 = uVar18;
      uVar10 = uVar18;
      if (0 < local_9d4) {
        do {
          *(undefined4 *)(uVar17 + (longlong)local_998) = 0;
          uVar8 = (int)uVar10 + 1;
          uVar17 = uVar17 + 4;
          uVar10 = (ulonglong)uVar8;
        } while ((int)uVar8 < local_9d4);
      }
      if (local_990 != &local_988) {
        cv::fastFree(local_990);
      }
      local_918 = *(undefined4 *)param_3;
      local_914 = *(int *)(param_3 + 4);
      local_910 = *(undefined4 *)(param_3 + 8);
      local_90c = *(undefined4 *)(param_3 + 0xc);
      local_908 = *(undefined8 *)(param_3 + 0x10);
      local_900 = *(undefined8 *)(param_3 + 0x18);
      local_8f8 = *(undefined8 *)(param_3 + 0x20);
      local_8f0 = *(undefined8 *)(param_3 + 0x28);
      local_8e8 = *(undefined8 *)(param_3 + 0x30);
      local_8e0 = *(longlong *)(param_3 + 0x38);
      local_8d8 = &local_910;
      local_8d0 = &local_8c8;
      local_8c0 = 0;
      local_8c8 = 0;
      if (local_8e0 != 0) {
        LOCK();
        *(int *)(local_8e0 + 0x14) = *(int *)(local_8e0 + 0x14) + 1;
        UNLOCK();
      }
      if (*(int *)(param_3 + 4) < 3) {
        local_8c8 = **(undefined8 **)(param_3 + 0x48);
        local_8c0 = *(undefined8 *)(*(longlong *)(param_3 + 0x48) + 8);
      }
      else {
        local_914 = 0;
        cv::Mat::copySize((Mat *)&local_918,param_3);
      }
      FUN_180004b70(lVar15,lVar12,(Mat *)&local_918,4);
      if (local_8e0 != 0) {
        LOCK();
        piVar13 = (int *)(local_8e0 + 0x14);
        iVar3 = *piVar13;
        *piVar13 = *piVar13 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate((Mat *)&local_918);
        }
      }
      local_8e0 = 0;
      local_908 = 0;
      local_8f0 = 0;
      local_8f8 = 0;
      local_900 = 0;
      uVar17 = uVar18;
      uVar10 = uVar18;
      if (0 < local_914) {
        do {
          *(undefined4 *)(uVar17 + (longlong)local_8d8) = 0;
          uVar8 = (int)uVar10 + 1;
          uVar17 = uVar17 + 4;
          uVar10 = (ulonglong)uVar8;
        } while ((int)uVar8 < local_914);
      }
      if (local_8d0 != &local_8c8) {
        cv::fastFree(local_8d0);
      }
      local_858 = *(undefined4 *)param_4;
      local_854 = *(int *)(param_4 + 4);
      local_850 = *(undefined4 *)(param_4 + 8);
      local_84c = *(undefined4 *)(param_4 + 0xc);
      local_848 = *(undefined8 *)(param_4 + 0x10);
      local_840 = *(undefined8 *)(param_4 + 0x18);
      local_838 = *(undefined8 *)(param_4 + 0x20);
      local_830 = *(undefined8 *)(param_4 + 0x28);
      local_828 = *(undefined8 *)(param_4 + 0x30);
      local_820 = *(longlong *)(param_4 + 0x38);
      local_818 = &local_850;
      local_810 = &local_808;
      local_800 = 0;
      local_808 = 0;
      if (local_820 != 0) {
        LOCK();
        *(int *)(local_820 + 0x14) = *(int *)(local_820 + 0x14) + 1;
        UNLOCK();
      }
      if (*(int *)(param_4 + 4) < 3) {
        local_808 = **(undefined8 **)(param_4 + 0x48);
        local_800 = *(undefined8 *)(*(longlong *)(param_4 + 0x48) + 8);
      }
      else {
        local_854 = 0;
        cv::Mat::copySize((Mat *)&local_858,param_4);
      }
      FUN_180005ed0(lVar15,&local_858,iVar4,uVar7);
      if (local_820 != 0) {
        LOCK();
        piVar13 = (int *)(local_820 + 0x14);
        iVar3 = *piVar13;
        *piVar13 = *piVar13 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate((Mat *)&local_858);
        }
      }
      local_820 = 0;
      local_848 = 0;
      local_830 = 0;
      local_838 = 0;
      local_840 = 0;
      uVar17 = uVar18;
      uVar10 = uVar18;
      if (0 < local_854) {
        do {
          *(undefined4 *)(uVar10 + (longlong)local_818) = 0;
          uVar7 = (int)uVar17 + 1;
          uVar17 = (ulonglong)uVar7;
          uVar10 = uVar10 + 4;
        } while ((int)uVar7 < local_854);
      }
      if (local_810 != &local_808) {
        cv::fastFree(local_810);
      }
      if (lStack_150 != 0) {
        LOCK();
        piVar13 = (int *)(lStack_150 + 0x14);
        iVar3 = *piVar13;
        *piVar13 = *piVar13 + -1;
        UNLOCK();
        if (iVar3 == 1) {
          cv::Mat::deallocate((Mat *)&local_188);
        }
      }
      lStack_150 = 0;
      local_178 = 0;
      uStack_160 = 0;
      local_168 = 0;
      uStack_170 = 0;
      uVar17 = uVar18;
      if (0 < local_188._4_4_) {
        do {
          *(undefined4 *)(uVar17 + (longlong)local_148) = 0;
          uVar7 = (int)uVar18 + 1;
          uVar18 = (ulonglong)uVar7;
          uVar17 = uVar17 + 4;
        } while ((int)uVar7 < local_188._4_4_);
      }
      if (local_140 != &local_138) {
        cv::fastFree(local_140);
      }
    }
  }
  local_a8[1] = local_a8[1] - local_98;
  return;
}


```

## {'procedure': '0x18001aa20'}
Evidence: `ev_7536072e8ebcfd1d1440749f9dd10fc377a473f350d94267af031ec3fb9e0ffc`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_18001aa20(longlong param_1,longlong *param_2,longlong param_3,undefined8 *param_4,Mat *param_5,
             longlong param_6,int param_7)

{
  float fVar1;
  float fVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  int iVar6;
  int iVar7;
  char *pcVar8;
  _InputOutputArray *p_Var9;
  longlong lVar10;
  uint uVar11;
  int *piVar12;
  float *pfVar13;
  int iVar15;
  ulonglong uVar16;
  int iVar17;
  longlong lVar18;
  int iVar19;
  float *pfVar20;
  longlong lVar21;
  float *pfVar22;
  int iVar23;
  ulonglong uVar24;
  uint uVar25;
  float fVar26;
  undefined4 uVar27;
  undefined4 uVar28;
  undefined1 auStackY_f88 [32];
  undefined8 uVar29;
  longlong *plVar30;
  int local_f2c;
  int local_ef8;
  uint local_ef4;
  longlong local_ef0;
  longlong local_ee8;
  undefined8 local_ee0;
  undefined8 local_ed8;
  longlong local_ed0;
  longlong *local_ec8;
  longlong local_ec0;
  longlong local_eb8;
  undefined8 *local_eb0;
  undefined4 local_ea8 [2];
  undefined8 *local_ea0;
  undefined8 local_e98;
  undefined4 local_e90 [2];
  undefined8 *local_e88;
  undefined4 local_e80;
  undefined4 local_e7c;
  undefined4 local_e78 [2];
  undefined8 *local_e70;
  undefined8 local_e68;
  undefined4 local_e60 [2];
  undefined8 *local_e58;
  undefined4 local_e50;
  undefined4 local_e4c;
  undefined4 local_e48 [2];
  longlong local_e40;
  undefined8 local_e38;
  undefined4 local_e30 [2];
  Mat *local_e28;
  undefined8 local_e20;
  longlong local_e18;
  int iStack_e10;
  longlong local_e08;
  undefined4 uStack_e00;
  undefined4 uStack_dfc;
  longlong local_df8;
  undefined4 uStack_df0;
  undefined4 uStack_dec;
  longlong local_de8;
  undefined4 uStack_de0;
  undefined4 uStack_ddc;
  longlong local_dd8;
  undefined4 uStack_dd0;
  undefined4 uStack_dcc;
  longlong local_dc8;
  undefined4 uStack_dc0;
  undefined4 uStack_dbc;
  Mat local_db8 [4];
  int local_db4;
  undefined8 local_da8;
  undefined8 local_da0;
  undefined8 local_d98;
  undefined8 local_d90;
  longlong local_d80;
  longlong local_d78;
  undefined1 *local_d70;
  undefined1 local_d68 [16];
  undefined4 local_d58;
  Mat local_d48 [4];
  int local_d44;
  undefined8 local_d38;
  undefined8 local_d30;
  undefined8 local_d28;
  undefined8 local_d20;
  longlong local_d10;
  longlong local_d08;
  undefined1 *local_d00;
  undefined1 local_cf8 [16];
  undefined4 local_ce8;
  longlong local_cd8;
  undefined8 uStack_cd0;
  longlong local_cc8;
  undefined8 uStack_cc0;
  longlong local_cb8;
  int local_cb0;
  longlong local_ca8;
  undefined8 uStack_ca0;
  undefined8 local_c98;
  longlong local_c88 [2];
  longlong local_c78;
  int local_c70;
  longlong local_c68;
  int local_c60;
  longlong local_c58;
  undefined8 uStack_c50;
  longlong local_c48;
  undefined8 uStack_c40;
  undefined8 local_c38;
  undefined8 uStack_c30;
  undefined8 local_c28;
  undefined8 uStack_c20;
  longlong local_c18;
  undefined8 uStack_c10;
  undefined4 local_c08;
  undefined4 uStack_c04;
  undefined4 uStack_c00;
  undefined4 uStack_bfc;
  longlong local_bf8;
  undefined8 uStack_bf0;
  undefined8 local_be8;
  undefined8 uStack_be0;
  undefined8 local_bd8;
  undefined8 uStack_bd0;
  undefined8 local_bc8;
  undefined8 uStack_bc0;
  longlong local_bb8;
  undefined8 uStack_bb0;
  undefined8 local_ba8;
  undefined8 uStack_ba0;
  float *local_b98;
  char local_b88 [16];
  undefined8 local_b78;
  undefined8 uStack_b70;
  longlong local_b68;
  undefined8 uStack_b60;
  undefined8 local_b58;
  undefined8 uStack_b50;
  undefined8 local_b48;
  longlong lStack_b40;
  undefined8 *local_b38;
  undefined8 *local_b30;
  undefined8 local_b28;
  undefined8 local_b20;
  undefined4 local_b18;
  longlong *local_b08;
  undefined8 local_b00;
  longlong local_af8;
  longlong local_af0;
  undefined8 local_ae8;
  undefined8 local_ad8;
  undefined8 uStack_ad0;
  longlong local_ac8;
  undefined8 uStack_ac0;
  undefined8 local_ab8;
  undefined8 uStack_ab0;
  undefined8 local_aa8;
  longlong lStack_aa0;
  undefined8 *local_a98;
  undefined8 *local_a90;
  undefined8 local_a88;
  undefined8 local_a80;
  undefined4 local_a78;
  undefined8 local_a68;
  undefined8 uStack_a60;
  longlong local_a58;
  undefined8 uStack_a50;
  undefined8 local_a48;
  undefined8 uStack_a40;
  undefined8 local_a38;
  longlong lStack_a30;
  undefined8 *local_a28;
  ulonglong *local_a20;
  ulonglong local_a18 [2];
  undefined4 local_a08;
  Mat local_9f8 [312];
  undefined1 local_8c0 [312];
  undefined1 local_788 [312];
  undefined1 local_650 [632];
  Mat local_3d8 [8];
  int local_3d0;
  longlong local_3c8;
  int local_388;
  int local_378;
  Mat local_238 [8];
  int local_230;
  longlong local_228;
  int local_1e8;
  int local_1d8;
  ulonglong local_98;
  ulonglong uVar14;
  
  local_c98 = 0xfffffffffffffffe;
  local_98 = DAT_18008b080 ^ (ulonglong)auStackY_f88;
  iVar17 = *(int *)(param_1 + 0x6f4);
  lVar18 = (longlong)iVar17;
  uVar11 = *(uint *)(param_1 + 0x6f8);
  iVar3 = *(int *)(param_1 + 0x6fc);
  iVar4 = *(int *)(param_1 + 0x700);
  piVar12 = (int *)(param_1 + 0x14);
  if (*(int *)(param_1 + 0xa8) < *(int *)(param_1 + 0x14)) {
    piVar12 = (int *)(param_1 + 0xa8);
  }
  iVar5 = *piVar12;
  local_ef0 = param_1;
  local_ec8 = param_2;
  local_eb0 = param_4;
  if (param_7 == 0) {
    uVar25 = *(uint *)(param_1 + 0xb8);
    uVar27 = (undefined4)*(undefined8 *)(param_1 + 0xc0);
    uVar28 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0xc0) >> 0x20);
    iVar6 = *(int *)(param_1 + 200);
  }
  else {
    if (param_7 != 1) {
      local_ee0 = 0;
      local_ed8 = 0;
      pcVar8 = cv::String::allocate((String *)&local_ee0,0x31);
      builtin_strncpy(pcVar8,"nDepthType == DEPTH_3D || nDepthType == DEPTH_EFI",0x31);
      cv::error(-0xd7,(String *)&local_ee0,"sg::cpu::WeightedMedianFilterCPU","SweepGuideCpu.cpp",
                0xbd3);
      cv::String::deallocate((String *)&local_ee0);
    }
    uVar25 = *(uint *)(param_1 + 0xd8);
    uVar27 = (undefined4)*(undefined8 *)(param_1 + 0xe0);
    uVar28 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0xe0) >> 0x20);
    iVar6 = *(int *)(param_1 + 0xe8);
  }
  iVar7 = (*(int *)(param_3 + 4) + -1) * *(int *)(param_1 + 0x7c) + 1;
  FUN_18001b920((longlong)local_9f8);
  memset(&local_b78,0,0x68);
  local_b78 = 0x42ff0000;
  uStack_b70 = 0;
  local_b68 = 0;
  uStack_b60 = 0;
  local_b58 = 0;
  uStack_b50 = 0;
  local_b48 = 0;
  lStack_b40 = 0;
  local_b38 = &uStack_b70;
  local_b30 = &local_b28;
  local_b20 = 0;
  local_b28 = 0;
  local_b18 = 0;
  FUN_1800726b8((longlong)local_238,0x68,4,&LAB_180006700);
  FUN_1800726b8((longlong)local_3d8,0x68,4,&LAB_180006700);
  memset(&local_a68,0,0x68);
  local_a68 = 0x42ff0000;
  uStack_a60 = 0;
  local_a58 = 0;
  uStack_a50 = 0;
  local_a48 = 0;
  uStack_a40 = 0;
  local_a38 = 0;
  lStack_a30 = 0;
  local_a28 = &uStack_a60;
  local_a20 = local_a18;
  local_a18[1] = 0;
  local_a18[0] = 0;
  local_a08 = 0;
  memset(&local_ad8,0,0x68);
  local_ad8 = 0x42ff0000;
  uStack_ad0 = 0;
  local_ac8 = 0;
  uStack_ac0 = 0;
  local_ab8 = 0;
  uStack_ab0 = 0;
  local_aa8 = 0;
  lStack_aa0 = 0;
  local_a98 = &uStack_ad0;
  local_a90 = &local_a88;
  local_a80 = 0;
  local_a88 = 0;
  local_a78 = 0;
  local_ae8 = 0;
  local_b00 = 0x20;
  local_af8 = 0;
  local_af0 = param_2[1] + *param_2;
  local_b08 = param_2;
  FUN_180026940((longlong *)&local_b08,param_1,uVar25,iVar6,local_9f8,&local_b78,local_238,local_3d8
                ,(Mat *)&local_a68,(Mat *)&local_ad8);
  if (0 < (int)uVar25) {
    if (iVar5 == 3) {
      local_e98 = 0;
      local_ea8[0] = 0x2010000;
      local_ea0 = &local_b78;
      local_e38 = 0;
      local_e48[0] = 0x1010000;
      local_e40 = param_6;
      uVar29 = 0;
      local_ef8 = iVar17;
      local_ef4 = uVar11;
      cv::resize(local_e48,local_ea8,&local_ef8);
      FUN_180019a20(param_1,param_2,(Mat *)&local_b78,local_9f8,uVar29,uVar25,
                    (double)CONCAT44(uVar28,uVar27));
    }
    else {
      if (*(int *)(param_1 + 0x14) == 3) {
        local_df8 = local_b68;
        uStack_df0 = (undefined4)local_b28;
        uStack_cd0 = CONCAT44(uStack_dec,(undefined4)local_b28);
        local_cd8 = local_b68;
        local_de8 = *(longlong *)(param_6 + 0x10);
        uStack_de0 = (undefined4)*(undefined8 *)(param_6 + 0x50);
        uStack_c50 = CONCAT44(uStack_ddc,uStack_de0);
        plVar30 = &local_bf8;
        local_c58 = local_de8;
        local_bf8 = *(longlong *)(param_1 + 0x4f8);
        uStack_bf0 = *(undefined8 *)(param_1 + 0x500);
        local_be8 = *(undefined8 *)(param_1 + 0x508);
        uStack_be0 = *(undefined8 *)(param_1 + 0x510);
        FUN_180026e20(iVar17,uVar11,&local_c58,&local_cd8,plVar30);
      }
      else {
        local_dd8 = local_b68;
        uStack_dd0 = (undefined4)local_b28;
        uStack_cc0 = CONCAT44(uStack_dcc,(undefined4)local_b28);
        local_cc8 = local_b68;
        local_e08 = *(longlong *)(param_6 + 0x10);
        uStack_e00 = (undefined4)*(undefined8 *)(param_6 + 0x50);
        uStack_ca0 = CONCAT44(uStack_dfc,uStack_e00);
        plVar30 = &local_bb8;
        local_ca8 = local_e08;
        local_bb8 = *(longlong *)(param_1 + 0x4f8);
        uStack_bb0 = *(undefined8 *)(param_1 + 0x500);
        local_ba8 = *(undefined8 *)(param_1 + 0x508);
        uStack_ba0 = *(undefined8 *)(param_1 + 0x510);
        FUN_1800270c0(iVar17,uVar11,&local_ca8,&local_cc8,plVar30);
      }
      FUN_180019240(param_1,param_2,(Mat *)&local_b78,(longlong)local_9f8,plVar30,uVar25,
                    (double)CONCAT44(uVar28,uVar27));
    }
    pfVar20 = *(float **)(param_5 + 0x10);
    fVar1 = *(float *)(param_3 + 0x2c);
    fVar2 = **(float **)(param_3 + 0x10);
    local_ee8 = lVar18;
    local_b98 = pfVar20;
    if (0 < (int)uVar11) {
      iVar17 = *(int *)(param_5 + 0x50);
      uVar24 = (ulonglong)uVar11;
      do {
        pfVar22 = pfVar20;
        lVar21 = lVar18;
        if (0 < lVar18) {
          do {
            fVar26 = (float)rintf((*pfVar22 - fVar2) * (1.0 / fVar1));
            *pfVar22 = fVar26;
            lVar21 = lVar21 + -1;
            pfVar22 = pfVar22 + 1;
          } while (lVar21 != 0);
        }
        pfVar20 = (float *)((longlong)pfVar20 + (longlong)iVar17);
        uVar24 = uVar24 - 1;
      } while (uVar24 != 0);
    }
    iVar19 = (local_230 + local_1d8 * 2) * local_1e8;
    iVar17 = (local_3d0 + local_378 * 2) * local_388;
    local_c38 = 0;
    uStack_c30 = 0;
    local_c28 = 0;
    uStack_c20 = 0;
    local_e90[0] = 0xc1020006;
    local_e88 = &local_c38;
    local_e80 = 1;
    local_e7c = 4;
    p_Var9 = cv::noArray();
    cv::Mat::setTo((Mat *)&local_a68,(_InputArray *)local_e90,(_InputArray *)p_Var9);
    local_bd8 = 0;
    uStack_bd0 = 0;
    local_bc8 = 0;
    uStack_bc0 = 0;
    local_e60[0] = 0xc1020006;
    local_e58 = &local_bd8;
    local_e50 = 1;
    local_e4c = 4;
    p_Var9 = cv::noArray();
    cv::Mat::setTo((Mat *)&local_ad8,(_InputArray *)local_e60,(_InputArray *)p_Var9);
    local_f2c = 0;
    if (0 < iVar7) {
      local_ed0 = (longlong)(int)(iVar17 + 0x1fU & 0xffffffe0);
      do {
        lVar18 = 0;
        if (((code *)*local_eb0 != (code *)0x0) &&
           (iVar17 = (*(code *)*local_eb0)
                               (local_eb0[1],
                                ((double)local_f2c / (double)iVar7) * 40.0 +
                                (double)(byte)((-(param_7 != 0) & 0x28U) + 5)), iVar17 != 0)) {
          uVar29 = 1;
          goto LAB_18001b67c;
        }
        local_c18 = *(longlong *)(param_1 + 0x718);
        uStack_c10 = *(undefined8 *)(param_1 + 0x720);
        local_c08 = *(undefined4 *)(param_1 + 0x728);
        uStack_c04 = *(undefined4 *)(param_1 + 0x72c);
        uStack_c00 = *(undefined4 *)(param_1 + 0x730);
        uStack_bfc = *(undefined4 *)(param_1 + 0x734);
        local_e18 = local_228;
        iStack_e10 = local_1e8;
        local_c88[0] = local_228;
        local_dc8 = *(longlong *)(param_5 + 0x10);
        uStack_dc0 = (undefined4)*(undefined8 *)(param_5 + 0x50);
        uStack_c40 = CONCAT44(uStack_dbc,uStack_dc0);
        local_c48 = local_dc8;
        FUN_180027570(iVar3,iVar4,p_Var9,&local_c48,local_c88,local_b88,iVar19 + 0x1fU & 0xffffffe0,
                      (float)local_f2c,&local_c18);
        plVar30 = local_ec8;
        pcVar8 = local_b88;
        lVar21 = 4;
        do {
          if (*pcVar8 != '\0') {
            FUN_18001a680(param_1,plVar30,local_238 + lVar18,(longlong)local_9f8,
                          (longlong)(local_3d8 + lVar18),uVar25,iVar5);
          }
          lVar18 = lVar18 + 0x68;
          pcVar8 = pcVar8 + 1;
          lVar21 = lVar21 + -1;
        } while (lVar21 != 0);
        local_cb8 = local_ac8;
        local_cb0 = (int)local_a88;
        local_ec0 = local_a58;
        local_c68 = local_a58;
        local_c60 = (int)local_a18[0];
        local_c78 = local_3c8;
        local_c70 = local_388;
        fVar1 = *(float *)(param_3 + 0x2c);
        fVar2 = **(float **)(param_3 + 0x10);
        iVar17 = 0;
        lVar18 = 0;
        local_eb8 = (longlong)(int)uVar11;
        if (0 < local_eb8) {
          do {
            iVar23 = 0;
            if (0 < local_ee8) {
              pfVar22 = (float *)((int)local_a18[0] * lVar18 + local_a58);
              pfVar20 = pfVar22;
              lVar21 = local_ee8;
              do {
                fVar26 = *pfVar20;
                if (fVar26 <= 0.5) {
                  pfVar13 = (float *)(local_3c8 +
                                     (longlong)
                                     ((int)((ulonglong)(longlong)local_388 >> 2) * iVar17 + iVar23)
                                     * 4);
                  iVar15 = 0;
                  lVar10 = 0;
                  do {
                    if ((local_b88[lVar10] != '\0') && (fVar26 = fVar26 + *pfVar13, 0.5 < fVar26)) {
                      if (-1 < iVar15) {
                        *(float *)((((int)local_a88 * lVar18 + local_ac8) - (longlong)pfVar22) +
                                  (longlong)pfVar20) =
                             ((float)iVar15 + (float)local_f2c) * fVar1 + fVar2;
                      }
                      break;
                    }
                    iVar15 = iVar15 + 1;
                    lVar10 = lVar10 + 1;
                    pfVar13 = (float *)((longlong)pfVar13 + local_ed0);
                  } while (lVar10 < 4);
                  *pfVar20 = fVar26;
                }
                iVar23 = iVar23 + 1;
                pfVar20 = pfVar20 + 1;
                lVar21 = lVar21 + -1;
              } while (lVar21 != 0);
            }
            iVar17 = iVar17 + 1;
            lVar18 = lVar18 + 1;
          } while (lVar18 < local_eb8);
        }
        p_Var9 = (_InputOutputArray *)(local_a18[0] & 0xffffffff);
        local_f2c = local_f2c + 4;
        param_1 = local_ef0;
      } while (local_f2c < iVar7);
    }
    FUN_180001d80("& Weighted Median",0,0,0,0);
    local_e20 = 0;
    local_e30[0] = 0x2010000;
    local_e28 = param_5;
    local_e68 = 0;
    local_e78[0] = 0x1010000;
    local_e70 = &local_ad8;
    cv::medianBlur((_InputArray *)local_e78,(_OutputArray *)local_e30,iVar6);
    FUN_180001d80("& Median Filter Finished",0,0,0,0);
  }
  if (*(char *)(param_1 + 0x11c) == '\x01') {
    if (param_7 == 0) {
      FUN_1800013f0(local_d48,param_5);
      local_ce8 = *(undefined4 *)(param_5 + 0x60);
      FUN_180004b70(param_1,param_3,local_d48,6);
      if (local_d10 != 0) {
        LOCK();
        piVar12 = (int *)(local_d10 + 0x14);
        iVar17 = *piVar12;
        *piVar12 = *piVar12 + -1;
        UNLOCK();
        if (iVar17 == 1) {
          cv::Mat::deallocate(local_d48);
        }
      }
      uVar14 = 0;
      local_d10 = 0;
      local_d38 = 0;
      local_d20 = 0;
      local_d28 = 0;
      local_d30 = 0;
      uVar24 = uVar14;
      if (0 < local_d44) {
        do {
          *(undefined4 *)(local_d08 + uVar24) = 0;
          uVar11 = (int)uVar14 + 1;
          uVar14 = (ulonglong)uVar11;
          uVar24 = uVar24 + 4;
        } while ((int)uVar11 < local_d44);
      }
      if (local_d00 != local_cf8) {
        cv::fastFree(local_d00);
      }
    }
    else if (param_7 == 1) {
      FUN_1800013f0(local_db8,param_5);
      local_d58 = *(undefined4 *)(param_5 + 0x60);
      FUN_180004b70(param_1,param_3,local_db8,0xb);
      if (local_d80 != 0) {
        LOCK();
        piVar12 = (int *)(local_d80 + 0x14);
        iVar17 = *piVar12;
        *piVar12 = *piVar12 + -1;
        UNLOCK();
        if (iVar17 == 1) {
          cv::Mat::deallocate(local_db8);
        }
      }
      uVar14 = 0;
      local_d80 = 0;
      local_da8 = 0;
      local_d90 = 0;
      local_d98 = 0;
      local_da0 = 0;
      uVar24 = uVar14;
      if (0 < local_db4) {
        do {
          *(undefined4 *)(local_d78 + uVar24) = 0;
          uVar11 = (int)uVar14 + 1;
          uVar14 = (ulonglong)uVar11;
          uVar24 = uVar24 + 4;
        } while ((int)uVar11 < local_db4);
      }
      if (local_d70 != local_d68) {
        cv::fastFree(local_d70);
      }
    }
  }
  uVar29 = 0;
LAB_18001b67c:
  uVar24 = 0;
  local_b08[1] = local_b08[1] - local_af8;
  if (lStack_aa0 != 0) {
    LOCK();
    piVar12 = (int *)(lStack_aa0 + 0x14);
    iVar17 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar17 == 1) {
      cv::Mat::deallocate((Mat *)&local_ad8);
    }
  }
  lStack_aa0 = 0;
  local_ac8 = 0;
  uStack_ab0 = 0;
  local_ab8 = 0;
  uStack_ac0 = 0;
  uVar14 = uVar24;
  uVar16 = uVar24;
  if (0 < local_ad8._4_4_) {
    do {
      *(undefined4 *)((longlong)local_a98 + uVar14) = 0;
      uVar11 = (int)uVar16 + 1;
      uVar14 = uVar14 + 4;
      uVar16 = (ulonglong)uVar11;
    } while ((int)uVar11 < local_ad8._4_4_);
  }
  if (local_a90 != &local_a88) {
    cv::fastFree(local_a90);
  }
  if (lStack_a30 != 0) {
    LOCK();
    piVar12 = (int *)(lStack_a30 + 0x14);
    iVar17 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar17 == 1) {
      cv::Mat::deallocate((Mat *)&local_a68);
    }
  }
  lStack_a30 = 0;
  local_a58 = 0;
  uStack_a40 = 0;
  local_a48 = 0;
  uStack_a50 = 0;
  uVar14 = uVar24;
  uVar16 = uVar24;
  if (0 < local_a68._4_4_) {
    do {
      *(undefined4 *)((longlong)local_a28 + uVar14) = 0;
      uVar11 = (int)uVar16 + 1;
      uVar14 = uVar14 + 4;
      uVar16 = (ulonglong)uVar11;
    } while ((int)uVar11 < local_a68._4_4_);
  }
  if (local_a20 != local_a18) {
    cv::fastFree(local_a20);
  }
  _eh_vector_destructor_iterator_(local_3d8,0x68,4,FUN_180001750);
  _eh_vector_destructor_iterator_(local_238,0x68,4,FUN_180001750);
  if (lStack_b40 != 0) {
    LOCK();
    piVar12 = (int *)(lStack_b40 + 0x14);
    iVar17 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar17 == 1) {
      cv::Mat::deallocate((Mat *)&local_b78);
    }
  }
  lStack_b40 = 0;
  local_b68 = 0;
  uStack_b50 = 0;
  local_b58 = 0;
  uStack_b60 = 0;
  uVar14 = uVar24;
  if (0 < local_b78._4_4_) {
    do {
      *(undefined4 *)((longlong)local_b38 + uVar14) = 0;
      uVar11 = (int)uVar24 + 1;
      uVar24 = (ulonglong)uVar11;
      uVar14 = uVar14 + 4;
    } while ((int)uVar11 < local_b78._4_4_);
  }
  if (local_b30 != &local_b28) {
    cv::fastFree(local_b30);
  }
  _eh_vector_destructor_iterator_(local_650,0x68,6,FUN_180001750);
  _eh_vector_destructor_iterator_(local_788,0x68,3,FUN_180001750);
  _eh_vector_destructor_iterator_(local_8c0,0x68,3,FUN_180001750);
  _eh_vector_destructor_iterator_(local_9f8,0x68,3,FUN_180001750);
  return uVar29;
}


```

## {'procedure': '0x18001fc80'}
Evidence: `ev_7b67ede848601e9752368f7180cc7132fc949ffa62475b28c06928c1edb1f548`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

int FUN_18001fc80(longlong param_1,undefined8 param_2,undefined8 param_3)

{
  int *piVar1;
  uint uVar2;
  uint uVar3;
  int iVar4;
  uint uVar5;
  ulonglong uVar6;
  ulonglong uVar7;
  longlong lVar8;
  longlong lVar9;
  ulonglong uVar10;
  int iVar11;
  uint uVar12;
  undefined1 auStackY_408 [32];
  undefined4 local_3b8 [2];
  Mat *local_3b0;
  undefined8 local_3a8;
  undefined4 local_3a0 [2];
  Mat *local_398;
  undefined8 local_390;
  longlong local_388;
  undefined4 uStack_380;
  longlong local_378 [2];
  undefined8 local_368;
  longlong local_358 [2];
  longlong local_348;
  undefined8 uStack_340;
  undefined8 local_338;
  undefined8 uStack_330;
  Mat local_328 [4];
  int local_324;
  undefined8 local_318;
  undefined8 local_310;
  undefined8 local_308;
  undefined8 local_300;
  longlong local_2f0;
  longlong local_2e8;
  undefined1 *local_2e0;
  undefined1 local_2d8 [16];
  undefined4 local_2c8;
  Mat local_2b8 [4];
  int local_2b4;
  longlong local_2a8;
  undefined8 local_2a0;
  undefined8 local_298;
  undefined8 local_290;
  longlong local_280;
  longlong local_278;
  undefined1 *local_270;
  undefined1 local_268 [16];
  undefined4 local_258;
  Mat local_248 [4];
  int local_244;
  longlong local_238;
  undefined8 local_230;
  undefined8 local_228;
  undefined8 local_220;
  longlong local_210;
  longlong local_208;
  undefined4 *local_200;
  undefined4 local_1f8 [4];
  undefined4 local_1e8;
  Mat local_1d8 [4];
  int local_1d4;
  undefined8 local_1c8;
  undefined8 local_1c0;
  undefined8 local_1b8;
  undefined8 local_1b0;
  longlong local_1a0;
  longlong local_198;
  undefined1 *local_190;
  undefined1 local_188 [16];
  Mat local_178 [4];
  int local_174;
  undefined8 local_168;
  undefined8 local_160;
  undefined8 local_158;
  undefined8 local_150;
  longlong local_140;
  longlong local_138;
  undefined1 *local_130;
  undefined1 local_128 [16];
  Mat local_118 [4];
  int local_114;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  undefined8 local_f0;
  longlong local_e0;
  longlong local_d8;
  undefined1 *local_d0;
  undefined1 local_c8 [16];
  undefined4 local_b8;
  Mat local_a8 [4];
  int local_a4;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_80;
  longlong local_70;
  longlong local_68;
  undefined1 *local_60;
  undefined1 local_58 [16];
  undefined4 local_48;
  ulonglong local_38;
  
  local_368 = 0xfffffffffffffffe;
  local_38 = DAT_18008b080 ^ (ulonglong)auStackY_408;
  iVar11 = *(int *)(param_1 + 0xb24);
  if (iVar11 == 0) {
    iVar11 = 0x16;
  }
  else {
    *(undefined8 *)(param_1 + 0x810) = *(undefined8 *)(param_1 + 0xb28);
    memset(local_248,0,0x68);
    FUN_1800013f0(local_248,(Mat *)(param_1 + 0xb38));
    local_1e8 = *(undefined4 *)(param_1 + 0xb98);
    memset(local_2b8,0,0x68);
    FUN_1800013f0(local_2b8,(Mat *)(param_1 + 0xba0));
    local_258 = *(undefined4 *)(param_1 + 0xc00);
    memset(local_328,0,0x68);
    FUN_1800013f0(local_328,(Mat *)(param_1 + 0xcd8));
    local_2c8 = *(undefined4 *)(param_1 + 0xd38);
    if (*(int *)(param_1 + 0xb20) < iVar11) {
      iVar11 = 1;
    }
    else {
      if (*(char *)(param_1 + 0x124) == '\x01') {
        FUN_180005740(param_1 + 8,local_328,2);
      }
      uVar5 = *(uint *)(param_1 + 8);
      uVar2 = *(uint *)(param_1 + 0xc);
      memset(local_1d8,0,0x60);
      FUN_1800014c0((uint *)local_1d8,uVar2,uVar5,5,param_3,(longlong)*(int *)(param_1 + 0x18));
      memset(local_178,0,0x60);
      if (*(int *)(param_1 + 0x1c) == 1) {
        uVar12 = 2;
        uVar3 = 0;
      }
      else {
        uVar3 = 0x10;
        uVar12 = 0x12;
      }
      if (*(int *)(param_1 + 0x20) == 8) {
        uVar12 = uVar3;
      }
      FUN_1800014c0((uint *)local_178,uVar2,uVar5,uVar12,param_2,(longlong)*(int *)(param_1 + 0x14))
      ;
      memset(local_a8,0,0x68);
      FUN_1800013f0(local_a8,(Mat *)(param_1 + 0x9b0));
      local_48 = *(undefined4 *)(param_1 + 0xa10);
      memset(local_118,0,0x68);
      FUN_1800013f0(local_118,(Mat *)(param_1 + 0xa18));
      local_b8 = *(undefined4 *)(param_1 + 0xa78);
      FUN_180003550(param_1 + 8,(longlong *)(param_1 + 0x808),local_a8,local_328,(longlong)local_118
                    ,local_328,4);
      iVar11 = 0;
      lVar9 = 0;
      FUN_180001d80("& BlendEFI",0,0,0,0);
      local_390 = 0;
      local_3a0[0] = 0x2010000;
      local_398 = local_178;
      iVar4 = 2;
      if (*(int *)(param_1 + 0x20) == 8) {
        iVar4 = iVar11;
      }
      cv::Mat::convertTo(local_328,(_OutputArray *)local_3a0,iVar4,1.0,0.0);
      FUN_180001d80("# convertTo(matOutEFI)",0,0,0,0);
      local_378[0] = local_2a8;
      local_388 = local_238;
      uStack_380 = local_1f8[0];
      local_358[0] = local_238;
      local_348 = *(longlong *)(param_1 + 0x588);
      uStack_340 = *(undefined8 *)(param_1 + 0x590);
      local_338 = *(undefined8 *)(param_1 + 0x598);
      uStack_330 = *(undefined8 *)(param_1 + 0x5a0);
      FUN_18001ba70(*(int *)(param_1 + 8),*(int *)(param_1 + 0xc),&local_348,local_358,local_378);
      local_3a8 = 0;
      local_3b8[0] = 0x2010000;
      local_3b0 = local_1d8;
      cv::Mat::copyTo(local_2b8,(_OutputArray *)local_3b8);
      FUN_180001d80("# download(matOutDepth3D)",0,0,0,0);
      *(undefined4 *)(param_1 + 0xb24) = 0;
      *(undefined8 *)(param_1 + 0x840) = 0;
      *(undefined1 *)(param_1 + 0x868) = 0;
      FUN_180001d80("& Construct Finished",0,0,0,0);
      if (local_e0 != 0) {
        LOCK();
        piVar1 = (int *)(local_e0 + 0x14);
        iVar4 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar4 == 1) {
          cv::Mat::deallocate(local_118);
        }
      }
      local_e0 = 0;
      local_108 = 0;
      local_f0 = 0;
      local_f8 = 0;
      local_100 = 0;
      lVar8 = lVar9;
      iVar4 = iVar11;
      if (0 < local_114) {
        do {
          *(undefined4 *)(local_d8 + lVar8) = 0;
          iVar4 = iVar4 + 1;
          lVar8 = lVar8 + 4;
        } while (iVar4 < local_114);
      }
      if (local_d0 != local_c8) {
        cv::fastFree(local_d0);
      }
      if (local_70 != 0) {
        LOCK();
        piVar1 = (int *)(local_70 + 0x14);
        iVar4 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar4 == 1) {
          cv::Mat::deallocate(local_a8);
        }
      }
      local_70 = 0;
      local_98 = 0;
      local_80 = 0;
      local_88 = 0;
      local_90 = 0;
      lVar8 = lVar9;
      iVar4 = iVar11;
      if (0 < local_a4) {
        do {
          *(undefined4 *)(local_68 + lVar8) = 0;
          iVar4 = iVar4 + 1;
          lVar8 = lVar8 + 4;
        } while (iVar4 < local_a4);
      }
      if (local_60 != local_58) {
        cv::fastFree(local_60);
      }
      if (local_140 != 0) {
        LOCK();
        piVar1 = (int *)(local_140 + 0x14);
        iVar4 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar4 == 1) {
          cv::Mat::deallocate(local_178);
        }
      }
      local_140 = 0;
      local_168 = 0;
      local_150 = 0;
      local_158 = 0;
      local_160 = 0;
      lVar8 = lVar9;
      iVar4 = iVar11;
      if (0 < local_174) {
        do {
          *(undefined4 *)(local_138 + lVar8) = 0;
          iVar4 = iVar4 + 1;
          lVar8 = lVar8 + 4;
        } while (iVar4 < local_174);
      }
      if (local_130 != local_128) {
        cv::fastFree(local_130);
      }
      if (local_1a0 != 0) {
        LOCK();
        piVar1 = (int *)(local_1a0 + 0x14);
        iVar4 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar4 == 1) {
          cv::Mat::deallocate(local_1d8);
        }
      }
      local_1a0 = 0;
      local_1c8 = 0;
      local_1b0 = 0;
      local_1b8 = 0;
      local_1c0 = 0;
      iVar4 = iVar11;
      if (0 < local_1d4) {
        do {
          *(undefined4 *)(local_198 + lVar9) = 0;
          iVar4 = iVar4 + 1;
          lVar9 = lVar9 + 4;
        } while (iVar4 < local_1d4);
      }
      if (local_190 != local_188) {
        cv::fastFree(local_190);
      }
    }
    uVar7 = 0;
    if (local_2f0 != 0) {
      LOCK();
      piVar1 = (int *)(local_2f0 + 0x14);
      iVar4 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar4 == 1) {
        cv::Mat::deallocate(local_328);
      }
    }
    local_2f0 = 0;
    local_318 = 0;
    local_300 = 0;
    local_308 = 0;
    local_310 = 0;
    uVar6 = uVar7;
    uVar10 = uVar7;
    if (0 < local_324) {
      do {
        *(undefined4 *)(local_2e8 + uVar10) = 0;
        uVar5 = (int)uVar6 + 1;
        uVar6 = (ulonglong)uVar5;
        uVar10 = uVar10 + 4;
      } while ((int)uVar5 < local_324);
    }
    if (local_2e0 != local_2d8) {
      cv::fastFree(local_2e0);
    }
    if (local_280 != 0) {
      LOCK();
      piVar1 = (int *)(local_280 + 0x14);
      iVar4 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar4 == 1) {
        cv::Mat::deallocate(local_2b8);
      }
    }
    local_280 = 0;
    local_2a8 = 0;
    local_290 = 0;
    local_298 = 0;
    local_2a0 = 0;
    uVar6 = uVar7;
    uVar10 = uVar7;
    if (0 < local_2b4) {
      do {
        *(undefined4 *)(local_278 + uVar10) = 0;
        uVar5 = (int)uVar6 + 1;
        uVar6 = (ulonglong)uVar5;
        uVar10 = uVar10 + 4;
      } while ((int)uVar5 < local_2b4);
    }
    if (local_270 != local_268) {
      cv::fastFree(local_270);
    }
    if (local_210 != 0) {
      LOCK();
      piVar1 = (int *)(local_210 + 0x14);
      iVar4 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar4 == 1) {
        cv::Mat::deallocate(local_248);
      }
    }
    local_210 = 0;
    local_238 = 0;
    local_220 = 0;
    local_228 = 0;
    local_230 = 0;
    uVar6 = uVar7;
    if (0 < local_244) {
      do {
        *(undefined4 *)(local_208 + uVar6) = 0;
        uVar5 = (int)uVar7 + 1;
        uVar7 = (ulonglong)uVar5;
        uVar6 = uVar6 + 4;
      } while ((int)uVar5 < local_244);
    }
    if (local_200 != local_1f8) {
      cv::fastFree(local_200);
    }
  }
  return iVar11;
}


```

## {'procedure': '0x1800152a0'}
Evidence: `ev_7c1f4c5c271c5d17c0ccaf6e544993594262407348aeb757a2097872d4d60f35`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_1800152a0(longlong param_1,longlong param_2,longlong param_3)

{
  ulonglong uVar1;
  float *pfVar2;
  float *pfVar3;
  float *pfVar4;
  float *pfVar5;
  float *pfVar6;
  uint uVar7;
  float fVar8;
  float fVar9;
  float fVar10;
  float fVar11;
  float fVar12;
  float fVar13;
  float fVar14;
  float fVar15;
  float fVar16;
  float fVar17;
  float fVar18;
  float fVar19;
  float fVar20;
  float fVar21;
  float fVar22;
  int iVar23;
  uint uVar24;
  longlong lVar25;
  longlong lVar26;
  float *pfVar27;
  longlong lVar28;
  ulonglong uVar29;
  int iVar30;
  int iVar32;
  ulonglong uVar33;
  ulonglong uVar34;
  longlong lVar35;
  longlong *plVar36;
  float fVar37;
  undefined1 auStack_138 [32];
  ulonglong local_118;
  int local_110;
  ulonglong local_108;
  longlong local_100;
  longlong local_f8;
  longlong local_f0;
  longlong local_e8;
  longlong local_e0;
  longlong local_d8;
  undefined8 local_d0;
  longlong *local_c8;
  undefined8 local_c0;
  longlong local_b8;
  longlong local_b0;
  undefined8 local_a8;
  ulonglong local_a0 [5];
  ulonglong local_78;
  int iVar31;
  
  local_d0 = 0xfffffffffffffffe;
  local_78 = DAT_18008b080 ^ (ulonglong)auStack_138;
  uVar24 = **(uint **)(param_3 + 0x40);
  uVar29 = (ulonglong)uVar24;
  uVar7 = (*(uint **)(param_3 + 0x40))[1];
  local_c8 = (longlong *)0x0;
  local_c0 = 0;
  local_b8 = 0;
  local_b0 = 0;
  local_a8 = 0;
  local_e8 = param_3;
  iVar23 = omp_get_thread_num();
  plVar36 = (longlong *)(*(longlong *)(param_1 + 0x30) + (longlong)iVar23 * 0x18);
  local_c0 = CONCAT44(local_c0._4_4_,0x20);
  local_f8 = 0;
  local_b0 = *plVar36 + plVar36[1];
  lVar28 = 0;
  uVar33 = (longlong)(int)uVar7 * 4 + 0x1fU & 0xffffffffffffffe0;
  do {
    local_f8 = local_f8 + uVar33;
    uVar34 = *plVar36 + plVar36[1];
    uVar1 = plVar36[1] + uVar33;
    plVar36[1] = uVar1;
    local_c8 = plVar36;
    local_b8 = local_f8;
    if ((ulonglong)plVar36[2] < uVar1) {
                    /* WARNING: Subroutine does not return */
      FUN_180001b90("Allocating memory exceeding the limit ...",lVar28,uVar33,uVar34);
    }
    local_a0[lVar28] = uVar34;
    lVar28 = lVar28 + 1;
  } while (lVar28 < 5);
  iVar23 = 0;
  do {
    FUN_180015100((longlong)((iVar23 + -2) * *(int *)(param_2 + 0x50)) +
                  *(longlong *)(param_2 + 0x10),(longlong *)local_a0,uVar7);
    iVar23 = iVar23 + 1;
  } while (iVar23 < 3);
  if (0 < (int)uVar24) {
    local_100 = 0;
    local_110 = 2;
    fVar37 = 4.0;
    do {
      lVar28 = local_100;
      iVar23 = local_110;
      local_118 = uVar29;
      FUN_180015100((longlong)((local_110 + -1) * *(int *)(param_2 + 0x50)) +
                    *(longlong *)(param_2 + 0x10),(longlong *)local_a0,uVar7);
      FUN_180015100((longlong)((int)*(undefined8 *)(param_2 + 0x50) * iVar23) +
                    *(longlong *)(param_2 + 0x10),(longlong *)local_a0,uVar7);
      uVar33 = **(longlong **)(param_3 + 0x48) * lVar28 + *(longlong *)(param_3 + 0x10);
      iVar31 = 0;
      iVar30 = 0;
      lVar35 = 0;
      if (0 < (int)uVar7) {
        lVar25 = lVar35;
        iVar32 = iVar31;
        if (3 < uVar7) {
          lVar26 = (longlong)(int)(uVar7 - 1);
          uVar1 = uVar33 + lVar26 * 4;
          if ((((((local_a0[2] + lVar26 * 4 < uVar33) || (uVar29 = local_118, uVar1 < local_a0[2]))
                && ((iVar32 = iVar30, local_a0[0] + lVar26 * 4 < uVar33 ||
                    (uVar29 = local_118, uVar1 < local_a0[0])))) &&
               ((local_a0[4] + lVar26 * 4 < uVar33 || (uVar29 = local_118, uVar1 < local_a0[4]))))
              && ((lVar25 = 0, local_a0[1] + lVar26 * 4 < uVar33 ||
                  (uVar29 = local_118, uVar1 < local_a0[1])))) &&
             ((local_a0[3] + lVar26 * 4 < uVar33 || (uVar29 = local_118, uVar1 < local_a0[3])))) {
            uVar24 = uVar7 & 0x80000003;
            if ((int)uVar24 < 0) {
              uVar24 = (uVar24 - 1 | 0xfffffffc) + 1;
            }
            iVar32 = iVar31;
            do {
              lVar25 = (longlong)iVar32;
              pfVar27 = (float *)(local_a0[1] + lVar25 * 4);
              fVar8 = pfVar27[1];
              fVar9 = pfVar27[2];
              fVar10 = pfVar27[3];
              pfVar2 = (float *)(local_a0[3] + lVar25 * 4);
              fVar11 = pfVar2[1];
              fVar12 = pfVar2[2];
              fVar13 = pfVar2[3];
              pfVar3 = (float *)(local_a0[0] + lVar25 * 4);
              fVar14 = pfVar3[1];
              fVar15 = pfVar3[2];
              fVar16 = pfVar3[3];
              pfVar4 = (float *)(local_a0[4] + lVar25 * 4);
              fVar17 = pfVar4[1];
              fVar18 = pfVar4[2];
              fVar19 = pfVar4[3];
              pfVar5 = (float *)(local_a0[2] + lVar25 * 4);
              fVar20 = pfVar5[1];
              fVar21 = pfVar5[2];
              fVar22 = pfVar5[3];
              pfVar6 = (float *)(uVar33 + lVar25 * 4);
              *pfVar6 = ((*pfVar27 + *pfVar2) * 4.0 + *pfVar3 + *pfVar4 + *pfVar5 * 6.0) *
                        0.00390625;
              pfVar6[1] = ((fVar8 + fVar11) * 4.0 + fVar14 + fVar17 + fVar20 * 6.0) * 0.00390625;
              pfVar6[2] = ((fVar9 + fVar12) * 4.0 + fVar15 + fVar18 + fVar21 * 6.0) * 0.00390625;
              pfVar6[3] = ((fVar10 + fVar13) * 4.0 + fVar16 + fVar19 + fVar22 * 6.0) * 0.00390625;
              iVar32 = iVar32 + 4;
              lVar35 = lVar35 + 4;
              uVar29 = local_118;
              lVar25 = lVar35;
            } while (iVar32 < (int)(uVar7 - uVar24));
          }
        }
        if (iVar32 < (int)uVar7) {
          if (3 < (int)(uVar7 - iVar32)) {
            pfVar27 = (float *)(local_a0[1] + 4 + lVar25 * 4);
            lVar28 = local_a0[3] - local_a0[1];
            lVar35 = local_a0[4] - local_a0[1];
            local_f0 = local_a0[0] - local_a0[1];
            local_d8 = local_a0[2] - local_a0[1];
            local_e0 = uVar33 - local_a0[1];
            uVar24 = ((uVar7 - iVar32) - 4 >> 2) + 1;
            local_108 = (ulonglong)uVar24;
            iVar32 = iVar32 + uVar24 * 4;
            lVar25 = lVar25 + local_108 * 4;
            do {
              *(float *)(local_e0 + -4 + (longlong)pfVar27) =
                   (*(float *)(local_f0 + -4 + (longlong)pfVar27) +
                    *(float *)(lVar35 + -4 + (longlong)pfVar27) +
                    (*(float *)((longlong)pfVar27 + lVar28 + -4) + pfVar27[-1]) * fVar37 +
                   *(float *)(local_d8 + -4 + (longlong)pfVar27) * 6.0) * 0.00390625;
              *(float *)(local_e0 + (longlong)pfVar27) =
                   (*(float *)(local_f0 + (longlong)pfVar27) +
                    *(float *)(lVar35 + (longlong)pfVar27) +
                    (*(float *)((longlong)pfVar27 + lVar28) + *pfVar27) * fVar37 +
                   *(float *)(local_d8 + (longlong)pfVar27) * 6.0) * 0.00390625;
              *(float *)(local_e0 + 4 + (longlong)pfVar27) =
                   (*(float *)(local_f0 + 4 + (longlong)pfVar27) +
                    *(float *)(lVar35 + 4 + (longlong)pfVar27) +
                    (*(float *)((longlong)pfVar27 + lVar28 + 4) + pfVar27[1]) * fVar37 +
                   *(float *)(local_d8 + 4 + (longlong)pfVar27) * 6.0) * 0.00390625;
              *(float *)(local_e0 + 8 + (longlong)pfVar27) =
                   ((*(float *)((longlong)pfVar27 + lVar28 + 8) + pfVar27[2]) * fVar37 +
                    *(float *)(local_f0 + 8 + (longlong)pfVar27) +
                    *(float *)(lVar35 + 8 + (longlong)pfVar27) +
                   *(float *)(local_d8 + 8 + (longlong)pfVar27) * 6.0) * 0.00390625;
              pfVar27 = pfVar27 + 4;
              local_108 = local_108 - 1;
            } while (local_108 != 0);
            local_108 = 0;
            lVar28 = local_100;
            iVar23 = local_110;
          }
          uVar29 = local_118;
          if (iVar32 < (int)uVar7) {
            pfVar27 = (float *)(local_a0[1] + lVar25 * 4);
            lVar35 = (longlong)(int)(uVar7 - iVar32);
            do {
              *(float *)((longlong)pfVar27 + (uVar33 - local_a0[1])) =
                   (*(float *)((longlong)pfVar27 + (local_a0[0] - local_a0[1])) +
                    *(float *)((longlong)pfVar27 + (local_a0[4] - local_a0[1])) +
                    (*(float *)((longlong)pfVar27 + (local_a0[3] - local_a0[1])) + *pfVar27) *
                    fVar37 + *(float *)((longlong)pfVar27 + (local_a0[2] - local_a0[1])) * 6.0) *
                   0.00390625;
              pfVar27 = pfVar27 + 1;
              lVar35 = lVar35 + -1;
            } while (lVar35 != 0);
          }
        }
      }
      local_110 = iVar23 + 2;
      local_100 = lVar28 + 1;
      uVar29 = uVar29 - 1;
      param_3 = local_e8;
      plVar36 = local_c8;
    } while (uVar29 != 0);
  }
  plVar36[1] = plVar36[1] - local_f8;
  return;
}


```

## {'procedure': '0x180016000'}
Evidence: `ev_86838a8dce3e48684acb1ece763e25c502e9b9062ef5d7fbf05cc7b117806a97`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180016000(longlong param_1,longlong param_2,Mat *param_3,longlong param_4)

{
  Mat *this;
  int *piVar1;
  char *pcVar2;
  uint uVar3;
  longlong lVar4;
  longlong lVar5;
  ulonglong uVar6;
  uint uVar7;
  int iVar8;
  ulonglong uVar9;
  longlong lVar10;
  Mat *pMVar11;
  int *piVar12;
  int iVar13;
  undefined1 auStackY_1e8 [32];
  undefined8 local_1b8;
  undefined8 local_1b0;
  undefined8 local_1a8;
  undefined8 local_1a0;
  undefined8 local_198;
  undefined8 local_190;
  longlong local_188;
  longlong local_180;
  undefined4 local_178 [2];
  longlong local_170;
  undefined8 local_168;
  undefined1 *local_160 [4];
  uint local_140;
  undefined8 local_138;
  undefined4 local_128;
  int local_124;
  int local_120;
  undefined4 local_11c;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  longlong local_f0;
  uint *local_e8;
  undefined8 *local_e0;
  undefined8 local_d8 [2];
  undefined4 local_c8;
  undefined4 local_b8;
  int local_b4;
  int local_b0;
  undefined4 local_ac;
  undefined8 local_a8;
  undefined8 local_a0;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  longlong local_80;
  int *local_78;
  undefined8 *local_70;
  undefined8 local_68;
  undefined8 local_60;
  undefined4 local_58;
  ulonglong local_48;
  
  local_138 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_1e8;
  iVar8 = *(int *)(param_1 + 0x28);
  lVar10 = (longlong)iVar8;
  local_188 = param_2;
  memset(&local_128,0,0x68);
  FUN_1800013f0((Mat *)&local_128,param_3);
  local_c8 = *(undefined4 *)(param_3 + 0x60);
  iVar13 = 0;
  if (0 < iVar8) {
    piVar12 = (int *)(param_1 + 0x264);
    pMVar11 = param_3 + 0x74;
    local_180 = (param_4 - (longlong)param_3) + -0x34;
    do {
      memset(&local_b8,0,0x68);
      local_b8 = *(undefined4 *)(pMVar11 + -0xc);
      local_b4 = *(int *)(pMVar11 + -8);
      local_b0 = *(int *)(pMVar11 + -4);
      local_ac = *(undefined4 *)pMVar11;
      local_a8 = *(undefined8 *)(pMVar11 + 4);
      local_a0 = *(undefined8 *)(pMVar11 + 0xc);
      local_98 = *(undefined8 *)(pMVar11 + 0x14);
      local_90 = *(undefined8 *)(pMVar11 + 0x1c);
      local_88 = *(undefined8 *)(pMVar11 + 0x24);
      local_80 = *(longlong *)(pMVar11 + 0x2c);
      local_78 = &local_b0;
      local_70 = &local_68;
      local_60 = 0;
      local_68 = 0;
      if (local_80 != 0) {
        LOCK();
        *(int *)(local_80 + 0x14) = *(int *)(local_80 + 0x14) + 1;
        UNLOCK();
      }
      if (*(int *)(pMVar11 + -8) < 3) {
        local_68 = **(undefined8 **)(pMVar11 + 0x3c);
        local_60 = *(undefined8 *)(*(longlong *)(pMVar11 + 0x3c) + 8);
      }
      else {
        local_b4 = 0;
        cv::Mat::copySize((Mat *)&local_b8,pMVar11 + -0xc);
      }
      local_58 = *(undefined4 *)(pMVar11 + 0x54);
      if ((piVar12[-3] != local_e8[1]) || (piVar12[-2] != *local_e8)) {
        local_198 = 0;
        local_190 = 0;
        pcVar2 = cv::String::allocate((String *)&local_198,0x23);
        builtin_strncpy(pcVar2,"tab->sizPyrImg[i] == matBuf1.size()",0x23);
        cv::error(-0xd7,(String *)&local_198,"sg::cpu::ConstructLapPyrCPU","SweepGuideCpu.cpp",0x595
                 );
        cv::String::deallocate((String *)&local_198);
      }
      if ((piVar12[-1] != local_78[1]) || (*piVar12 != *local_78)) {
        local_1b8 = 0;
        local_1b0 = 0;
        pcVar2 = cv::String::allocate((String *)&local_1b8,0x27);
        builtin_strncpy(pcVar2,"tab->sizPyrImg[i + 1] == matBuf2.size()",0x27);
        cv::error(-0xd7,(String *)&local_1b8,"sg::cpu::ConstructLapPyrCPU","SweepGuideCpu.cpp",0x596
                 );
        cv::String::deallocate((String *)&local_1b8);
      }
      FUN_1800256b0((Mat *)&local_128,2);
      lVar5 = local_188;
      FUN_1800152a0(local_188,(longlong)&local_128,(longlong)&local_b8);
      local_140 = local_e8[1];
      local_160[0] = &LAB_180025a40;
      local_160[1] = &LAB_180025a50;
      local_160[2] = &LAB_180025a60;
      local_160[3] = &LAB_180025a70;
      uVar3 = *local_e8 & 0x80000001;
      if ((int)uVar3 < 0) {
        uVar3 = (uVar3 - 1 | 0xfffffffe) + 1;
      }
      uVar7 = local_e8[1] & 0x80000001;
      if ((int)uVar7 < 0) {
        uVar7 = (uVar7 - 1 | 0xfffffffe) + 1;
      }
      (*(code *)local_160[(longlong)(int)uVar7 + (longlong)(int)uVar3 * 2])(&local_b8,2);
      FUN_180015960(lVar5,(longlong)&local_b8,(longlong)&local_128);
      this = pMVar11 + -0x74;
      if (this != (Mat *)&local_128) {
        if (local_f0 != 0) {
          LOCK();
          *(int *)(local_f0 + 0x14) = *(int *)(local_f0 + 0x14) + 1;
          UNLOCK();
        }
        if (*(longlong *)(pMVar11 + -0x3c) != 0) {
          LOCK();
          piVar1 = (int *)(*(longlong *)(pMVar11 + -0x3c) + 0x14);
          iVar8 = *piVar1;
          *piVar1 = *piVar1 + -1;
          UNLOCK();
          if (iVar8 == 1) {
            cv::Mat::deallocate(this);
          }
        }
        uVar6 = 0;
        *(undefined8 *)(pMVar11 + -0x3c) = 0;
        *(undefined8 *)(pMVar11 + -100) = 0;
        *(undefined8 *)(pMVar11 + -0x4c) = 0;
        *(undefined8 *)(pMVar11 + -0x54) = 0;
        *(undefined8 *)(pMVar11 + -0x5c) = 0;
        uVar9 = uVar6;
        if (0 < *(int *)(pMVar11 + -0x70)) {
          do {
            *(undefined4 *)(uVar6 + *(longlong *)(pMVar11 + -0x34)) = 0;
            uVar3 = (int)uVar9 + 1;
            uVar6 = uVar6 + 4;
            uVar9 = (ulonglong)uVar3;
          } while ((int)uVar3 < *(int *)(pMVar11 + -0x70));
        }
        *(undefined4 *)this = local_128;
        if ((*(int *)(pMVar11 + -0x70) < 3) && (local_124 < 3)) {
          *(int *)(pMVar11 + -0x70) = local_124;
          *(int *)(pMVar11 + -0x6c) = local_120;
          *(undefined4 *)(pMVar11 + -0x68) = local_11c;
          **(undefined8 **)(pMVar11 + -0x2c) = *local_e0;
          *(undefined8 *)(*(longlong *)(pMVar11 + -0x2c) + 8) = local_e0[1];
        }
        else {
          cv::Mat::copySize(this,(Mat *)&local_128);
        }
        *(undefined8 *)(pMVar11 + -100) = local_118;
        *(undefined8 *)(pMVar11 + -0x5c) = local_110;
        *(undefined8 *)(pMVar11 + -0x54) = local_108;
        *(undefined8 *)(pMVar11 + -0x4c) = local_100;
        *(undefined8 *)(pMVar11 + -0x44) = local_f8;
        *(longlong *)(pMVar11 + -0x3c) = local_f0;
      }
      *(undefined4 *)(pMVar11 + -0x14) = local_c8;
      if (param_4 != 0) {
        if ((local_78[1] != (*(int **)(pMVar11 + local_180))[1]) ||
           (*local_78 != **(int **)(pMVar11 + local_180))) {
          local_1a8 = 0;
          local_1a0 = 0;
          pcVar2 = cv::String::allocate((String *)&local_1a8,0x27);
          builtin_strncpy(pcVar2,"matBuf2.size() == matGaussPyr[i].size()",0x27);
          cv::error(-0xd7,(String *)&local_1a8,"sg::cpu::ConstructLapPyrCPU","SweepGuideCpu.cpp",
                    0x5a7);
          cv::String::deallocate((String *)&local_1a8);
        }
        local_168 = 0;
        local_178[0] = 0x2010000;
        local_170 = (longlong)iVar13 * 0x68 + param_4;
        cv::Mat::copyTo((Mat *)&local_b8,(_OutputArray *)local_178);
      }
      lVar5 = 0;
      if (local_80 != 0) {
        LOCK();
        *(int *)(local_80 + 0x14) = *(int *)(local_80 + 0x14) + 1;
        UNLOCK();
      }
      if (local_f0 != 0) {
        LOCK();
        piVar1 = (int *)(local_f0 + 0x14);
        iVar8 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar8 == 1) {
          cv::Mat::deallocate((Mat *)&local_128);
        }
      }
      local_f0 = 0;
      local_118 = 0;
      local_100 = 0;
      local_108 = 0;
      local_110 = 0;
      iVar8 = 0;
      lVar4 = lVar5;
      if (0 < local_124) {
        do {
          *(undefined4 *)(lVar4 + (longlong)local_e8) = 0;
          iVar8 = iVar8 + 1;
          lVar4 = lVar4 + 4;
        } while (iVar8 < local_124);
      }
      local_128 = local_b8;
      if ((local_124 < 3) && (local_b4 < 3)) {
        local_124 = local_b4;
        local_120 = local_b0;
        local_11c = local_ac;
        *local_e0 = *local_70;
        local_e0[1] = local_70[1];
      }
      else {
        cv::Mat::copySize((Mat *)&local_128,(Mat *)&local_b8);
      }
      local_118 = local_a8;
      local_110 = local_a0;
      local_108 = local_98;
      local_100 = local_90;
      local_f8 = local_88;
      local_f0 = local_80;
      local_c8 = local_58;
      if (local_80 != 0) {
        LOCK();
        piVar1 = (int *)(local_80 + 0x14);
        iVar8 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar8 == 1) {
          cv::Mat::deallocate((Mat *)&local_b8);
        }
      }
      local_80 = 0;
      local_a8 = 0;
      local_90 = 0;
      local_98 = 0;
      local_a0 = 0;
      iVar8 = 0;
      if (0 < local_b4) {
        do {
          *(undefined4 *)(lVar5 + (longlong)local_78) = 0;
          iVar8 = iVar8 + 1;
          lVar5 = lVar5 + 4;
        } while (iVar8 < local_b4);
      }
      if (local_70 != &local_68) {
        cv::fastFree(local_70);
      }
      iVar13 = iVar13 + 1;
      piVar12 = piVar12 + 2;
      pMVar11 = pMVar11 + 0x68;
      lVar10 = lVar10 + -1;
    } while (lVar10 != 0);
  }
  uVar6 = 0;
  if (local_f0 != 0) {
    LOCK();
    piVar12 = (int *)(local_f0 + 0x14);
    iVar8 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar8 == 1) {
      cv::Mat::deallocate((Mat *)&local_128);
    }
  }
  local_f0 = 0;
  local_118 = 0;
  local_100 = 0;
  local_108 = 0;
  local_110 = 0;
  uVar9 = uVar6;
  if (0 < local_124) {
    do {
      *(undefined4 *)(uVar9 + (longlong)local_e8) = 0;
      uVar3 = (int)uVar6 + 1;
      uVar6 = (ulonglong)uVar3;
      uVar9 = uVar9 + 4;
    } while ((int)uVar3 < local_124);
  }
  if (local_e0 != local_d8) {
    cv::fastFree(local_e0);
  }
  return;
}


```

## {'procedure': '0x180016840'}
Evidence: `ev_8a0c56d03763cf3b6aa001c84437b22100a4a38f2f799be9540674d4e545b9f4`

```c

void FUN_180016840(int param_1,int param_2,int param_3,longlong *param_4,longlong *param_5,
                  longlong *param_6,longlong *param_7)

{
  float fVar1;
  bool bVar2;
  longlong lVar3;
  int iVar4;
  int iVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  
  if (0 < param_2) {
    iVar5 = 0;
    lVar8 = (longlong)param_1;
    lVar6 = 0;
    do {
      iVar4 = 0;
      lVar3 = 0;
      if (3 < lVar8) {
        do {
          lVar7 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar5 + iVar4);
          fVar1 = *(float *)(*param_4 + lVar7 * 4);
          bVar2 = *(float *)(*param_5 + lVar7 * 4) <= fVar1;
          if (bVar2) {
            *(float *)(*param_5 + lVar7 * 4) = fVar1;
            *(float *)(*param_6 + lVar7 * 4) = (float)param_3;
          }
          *(bool *)((int)param_7[1] * lVar6 + *param_7 + lVar3) = bVar2;
          lVar7 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar5 + 1 + iVar4);
          fVar1 = *(float *)(*param_4 + lVar7 * 4);
          bVar2 = *(float *)(*param_5 + lVar7 * 4) <= fVar1;
          if (bVar2) {
            *(float *)(*param_5 + lVar7 * 4) = fVar1;
            *(float *)(*param_6 + lVar7 * 4) = (float)param_3;
          }
          *(bool *)((int)param_7[1] * lVar6 + *param_7 + 1 + lVar3) = bVar2;
          lVar7 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar5 + 2 + iVar4);
          fVar1 = *(float *)(*param_4 + lVar7 * 4);
          bVar2 = *(float *)(*param_5 + lVar7 * 4) <= fVar1;
          if (bVar2) {
            *(float *)(*param_5 + lVar7 * 4) = fVar1;
            *(float *)(*param_6 + lVar7 * 4) = (float)param_3;
          }
          *(bool *)((int)param_7[1] * lVar6 + *param_7 + 2 + lVar3) = bVar2;
          lVar7 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar5 + 3 + iVar4);
          fVar1 = *(float *)(*param_4 + lVar7 * 4);
          bVar2 = *(float *)(*param_5 + lVar7 * 4) <= fVar1;
          if (bVar2) {
            *(float *)(*param_5 + lVar7 * 4) = fVar1;
            *(float *)(*param_6 + lVar7 * 4) = (float)param_3;
          }
          lVar3 = lVar3 + 4;
          iVar4 = iVar4 + 4;
          *(bool *)((int)param_7[1] * lVar6 + *param_7 + -1 + lVar3) = bVar2;
        } while (lVar3 < lVar8 + -3);
      }
      while (lVar3 < lVar8) {
        lVar7 = (longlong)((int)((ulonglong)(longlong)(int)param_4[1] >> 2) * iVar5 + iVar4);
        fVar1 = *(float *)(*param_4 + lVar7 * 4);
        bVar2 = *(float *)(*param_5 + lVar7 * 4) <= fVar1;
        if (bVar2) {
          *(float *)(*param_5 + lVar7 * 4) = fVar1;
          *(float *)(*param_6 + lVar7 * 4) = (float)param_3;
        }
        lVar3 = lVar3 + 1;
        iVar4 = iVar4 + 1;
        *(bool *)((int)param_7[1] * lVar6 + *param_7 + -1 + lVar3) = bVar2;
      }
      iVar5 = iVar5 + 1;
      lVar6 = lVar6 + 1;
    } while (iVar5 < param_2);
  }
  return;
}


```

## {'procedure': '0x1800121b0'}
Evidence: `ev_8f3644f8f11c96afc308a2d923058b00f290f0a2dd7d56956b77dc82b46e918d`

```c

float FUN_1800121b0(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5,
                   longlong *param_6)

{
  float fVar1;
  float fVar2;
  longlong lVar3;
  longlong lVar4;
  int iVar5;
  longlong lVar6;
  longlong lVar7;
  int iVar8;
  float fVar9;
  float fVar10;
  
  iVar5 = 0;
  lVar6 = (longlong)param_1;
  fVar10 = 0.0;
  if (0 < param_2) {
    lVar7 = 0;
    do {
      iVar8 = 0;
      lVar4 = 0;
      if (3 < lVar6) {
        do {
          lVar3 = (longlong)((int)((ulonglong)(longlong)(int)param_5[1] >> 2) * iVar5 + iVar8);
          fVar1 = *(float *)(*param_5 + lVar3 * 4);
          fVar2 = *(float *)(*param_3 + lVar3 * 4);
          fVar9 = fVar2;
          if (*(char *)((int)param_6[1] * lVar7 + *param_6 + lVar4) == '\0') {
            if (fVar1 == 0.0) {
              fVar9 = 0.0;
            }
            else {
              fVar9 = *(float *)(*param_4 + lVar3 * 4) / fVar1;
            }
            fVar10 = fVar10 + ABS(fVar2 - fVar9);
          }
          *(float *)(*param_4 + lVar3 * 4) = fVar9;
          if (0.0 < fVar1) {
            *(undefined4 *)(*param_5 + lVar3 * 4) = 0x3f800000;
          }
          lVar3 = (longlong)((int)((ulonglong)(longlong)(int)param_5[1] >> 2) * iVar5 + 1 + iVar8);
          fVar1 = *(float *)(*param_5 + lVar3 * 4);
          fVar2 = *(float *)(*param_3 + lVar3 * 4);
          fVar9 = fVar2;
          if (*(char *)((int)param_6[1] * lVar7 + *param_6 + 1 + lVar4) == '\0') {
            if (fVar1 == 0.0) {
              fVar9 = 0.0;
            }
            else {
              fVar9 = *(float *)(*param_4 + lVar3 * 4) / fVar1;
            }
            fVar10 = fVar10 + ABS(fVar2 - fVar9);
          }
          *(float *)(*param_4 + lVar3 * 4) = fVar9;
          if (0.0 < fVar1) {
            *(undefined4 *)(*param_5 + lVar3 * 4) = 0x3f800000;
          }
          lVar3 = (longlong)((int)((ulonglong)(longlong)(int)param_5[1] >> 2) * iVar5 + 2 + iVar8);
          fVar1 = *(float *)(*param_5 + lVar3 * 4);
          fVar2 = *(float *)(*param_3 + lVar3 * 4);
          fVar9 = fVar2;
          if (*(char *)((int)param_6[1] * lVar7 + *param_6 + 2 + lVar4) == '\0') {
            if (fVar1 == 0.0) {
              fVar9 = 0.0;
            }
            else {
              fVar9 = *(float *)(*param_4 + lVar3 * 4) / fVar1;
            }
            fVar10 = fVar10 + ABS(fVar2 - fVar9);
          }
          *(float *)(*param_4 + lVar3 * 4) = fVar9;
          if (0.0 < fVar1) {
            *(undefined4 *)(*param_5 + lVar3 * 4) = 0x3f800000;
          }
          lVar3 = (longlong)((int)((ulonglong)(longlong)(int)param_5[1] >> 2) * iVar5 + 3 + iVar8);
          fVar1 = *(float *)(*param_5 + lVar3 * 4);
          fVar2 = *(float *)(*param_3 + lVar3 * 4);
          fVar9 = fVar2;
          if (*(char *)((int)param_6[1] * lVar7 + *param_6 + 3 + lVar4) == '\0') {
            if (fVar1 == 0.0) {
              fVar9 = 0.0;
            }
            else {
              fVar9 = *(float *)(*param_4 + lVar3 * 4) / fVar1;
            }
            fVar10 = fVar10 + ABS(fVar2 - fVar9);
          }
          *(float *)(*param_4 + lVar3 * 4) = fVar9;
          if (0.0 < fVar1) {
            *(undefined4 *)(*param_5 + lVar3 * 4) = 0x3f800000;
          }
          lVar4 = lVar4 + 4;
          iVar8 = iVar8 + 4;
        } while (lVar4 < lVar6 + -3);
      }
      for (; lVar4 < lVar6; lVar4 = lVar4 + 1) {
        lVar3 = (longlong)((int)((ulonglong)(longlong)(int)param_5[1] >> 2) * iVar5 + iVar8);
        fVar1 = *(float *)(*param_5 + lVar3 * 4);
        fVar2 = *(float *)(*param_3 + lVar3 * 4);
        fVar9 = fVar2;
        if (*(char *)((int)param_6[1] * lVar7 + *param_6 + lVar4) == '\0') {
          if (fVar1 == 0.0) {
            fVar9 = 0.0;
          }
          else {
            fVar9 = *(float *)(*param_4 + lVar3 * 4) / fVar1;
          }
          fVar10 = fVar10 + ABS(fVar2 - fVar9);
        }
        *(float *)(*param_4 + lVar3 * 4) = fVar9;
        if (0.0 < fVar1) {
          *(undefined4 *)(*param_5 + lVar3 * 4) = 0x3f800000;
        }
        iVar8 = iVar8 + 1;
      }
      iVar5 = iVar5 + 1;
      lVar7 = lVar7 + 1;
    } while (iVar5 < param_2);
  }
  return fVar10;
}


```

## {'procedure': '0x180002e50'}
Evidence: `ev_938acaf413863a99a130e35c0289295d469236a97a0821d91032d86cbbf527f2`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180002e50(longlong param_1,longlong *param_2,int *param_3,undefined8 param_4)

{
  float fVar1;
  float fVar3;
  int iVar4;
  ulonglong uVar5;
  longlong lVar6;
  longlong lVar7;
  uint uVar8;
  ulonglong uVar9;
  ulonglong uVar10;
  longlong lVar11;
  int iVar12;
  float *pfVar13;
  float fVar14;
  ulonglong uVar15;
  float *pfVar16;
  undefined1 auStack_98 [32];
  undefined8 local_78;
  longlong *local_70;
  undefined8 local_68;
  ulonglong local_60;
  float *local_58;
  undefined8 local_50;
  ulonglong local_48;
  float fVar2;
  
  local_78 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStack_98;
  iVar12 = *param_3;
  lVar7 = (longlong)iVar12;
  if (*(char *)(param_1 + 0x24) == '\0') {
    if (1 < iVar12) {
      fVar14 = **(float **)(param_3 + 2);
      pfVar16 = *(float **)(param_3 + 2) + lVar7 + -1;
      fVar3 = *pfVar16;
      fVar1 = *pfVar16;
      fVar2 = *pfVar16;
      *(bool *)(param_3 + 10) = fVar2 <= fVar14 && fVar14 != fVar1;
      uVar5 = 0;
      if ((NAN(fVar14) || NAN(fVar3)) || (fVar2 > fVar14 || fVar14 == fVar1)) {
        uVar15 = uVar5;
        uVar10 = uVar5;
        if (0 < iVar12) {
          do {
            *(undefined4 *)(uVar10 + *(longlong *)(param_3 + 4)) =
                 *(undefined4 *)(uVar10 + *(longlong *)(param_3 + 2));
            *(short *)(uVar5 + *(longlong *)(param_3 + 6)) = (short)uVar15;
            *(short *)(uVar5 + *(longlong *)(param_3 + 8)) = (short)uVar15;
            uVar8 = (int)uVar15 + 1;
            uVar5 = uVar5 + 2;
            uVar15 = (ulonglong)uVar8;
            uVar10 = uVar10 + 4;
          } while ((int)uVar8 < iVar12);
        }
      }
      else if (0 < iVar12) {
        iVar4 = iVar12 + -1;
        lVar6 = (longlong)iVar4 * 2;
        lVar11 = (longlong)iVar4 * 4;
        uVar10 = uVar5;
        uVar15 = uVar5;
        do {
          *(undefined4 *)(uVar15 + *(longlong *)(param_3 + 4)) =
               *(undefined4 *)(lVar11 + *(longlong *)(param_3 + 2));
          *(short *)(uVar10 + *(longlong *)(param_3 + 6)) = (short)iVar4;
          *(short *)(lVar6 + *(longlong *)(param_3 + 8)) = (short)uVar5;
          uVar8 = (int)uVar5 + 1;
          uVar5 = (ulonglong)uVar8;
          iVar4 = iVar4 + -1;
          lVar11 = lVar11 + -4;
          lVar6 = lVar6 + -2;
          uVar15 = uVar15 + 4;
          uVar10 = uVar10 + 2;
        } while ((int)uVar8 < iVar12);
      }
    }
  }
  else {
    local_50 = 0;
    local_68 = 0x20;
    pfVar16 = (float *)(*param_2 + param_2[1]);
    uVar15 = lVar7 * 8 + 0x1fU & 0xffffffffffffffe0;
    uVar5 = param_2[1] + uVar15;
    param_2[1] = uVar5;
    local_70 = param_2;
    local_60 = uVar15;
    local_58 = pfVar16;
    if ((ulonglong)param_2[2] < uVar5) {
                    /* WARNING: Subroutine does not return */
      FUN_180001b90("Allocating memory exceeding the limit ...",param_2,param_3,param_4);
    }
    uVar9 = 0;
    uVar5 = uVar9;
    uVar10 = uVar9;
    if (3 < lVar7) {
      fVar14 = 2.8026e-45;
      pfVar13 = pfVar16 + 2;
      do {
        pfVar13[-2] = *(float *)(*(longlong *)(param_3 + 2) + uVar5 * 4);
        pfVar13[-1] = (float)uVar10;
        *pfVar13 = *(float *)(*(longlong *)(param_3 + 2) + 4 + uVar5 * 4);
        pfVar13[1] = (float)((int)fVar14 + -1);
        pfVar13[2] = *(float *)(*(longlong *)(param_3 + 2) + 8 + uVar5 * 4);
        pfVar13[3] = fVar14;
        pfVar13[4] = *(float *)(*(longlong *)(param_3 + 2) + 0xc + uVar5 * 4);
        pfVar13[5] = (float)((int)fVar14 + 1);
        uVar10 = (ulonglong)((int)(float)uVar10 + 4);
        fVar14 = (float)((int)fVar14 + 4);
        uVar5 = uVar5 + 4;
        pfVar13 = pfVar13 + 8;
      } while ((longlong)uVar5 < lVar7 + -3);
    }
    for (; (longlong)uVar5 < lVar7; uVar5 = uVar5 + 1) {
      pfVar16[uVar5 * 2] = *(float *)(*(longlong *)(param_3 + 2) + uVar5 * 4);
      pfVar16[uVar5 * 2 + 1] = (float)uVar10;
      uVar10 = (ulonglong)((int)(float)uVar10 + 1);
    }
    FUN_1800114f0(pfVar16,pfVar16 + lVar7 * 2,
                  (longlong)(pfVar16 + lVar7 * 2) - (longlong)pfVar16 >> 3,0);
    if (0 < iVar12) {
      pfVar16 = pfVar16 + 1;
      uVar10 = uVar9;
      uVar5 = uVar9;
      do {
        *(float *)(uVar5 + *(longlong *)(param_3 + 4)) = pfVar16[-1];
        *(undefined2 *)(uVar10 + *(longlong *)(param_3 + 6)) = *(undefined2 *)pfVar16;
        *(short *)(*(longlong *)(param_3 + 8) + (longlong)(int)*pfVar16 * 2) = (short)uVar9;
        uVar8 = (int)uVar9 + 1;
        uVar9 = (ulonglong)uVar8;
        uVar5 = uVar5 + 4;
        uVar10 = uVar10 + 2;
        pfVar16 = pfVar16 + 2;
      } while ((int)uVar8 < iVar12);
    }
    param_2[1] = param_2[1] - uVar15;
  }
  fVar14 = **(float **)(param_3 + 4);
  iVar12 = 1;
  lVar11 = 1;
  if (1 < lVar7) {
    if (3 < lVar7 + -1) {
      do {
        fVar3 = *(float *)(*(longlong *)(param_3 + 4) + lVar11 * 4);
        if (4.9303807e-32 < ABS(fVar3 - fVar14)) {
          lVar6 = (longlong)iVar12;
          *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4) = fVar3;
          fVar14 = *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4);
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + lVar11 * 2) * 2) =
               (short)iVar12;
          *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar6 * 2) =
               *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar11 * 2);
          iVar12 = iVar12 + 1;
        }
        else {
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + lVar11 * 2) * 2) =
               (short)iVar12 + -1;
        }
        fVar3 = *(float *)(*(longlong *)(param_3 + 4) + 4 + lVar11 * 4);
        if (4.9303807e-32 < ABS(fVar3 - fVar14)) {
          lVar6 = (longlong)iVar12;
          *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4) = fVar3;
          fVar14 = *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4);
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + 2 + lVar11 * 2) * 2) =
               (short)iVar12;
          *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar6 * 2) =
               *(undefined2 *)(*(longlong *)(param_3 + 6) + 2 + lVar11 * 2);
          iVar12 = iVar12 + 1;
        }
        else {
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + 2 + lVar11 * 2) * 2) =
               (short)iVar12 + -1;
        }
        fVar3 = *(float *)(*(longlong *)(param_3 + 4) + 8 + lVar11 * 4);
        if (4.9303807e-32 < ABS(fVar3 - fVar14)) {
          lVar6 = (longlong)iVar12;
          *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4) = fVar3;
          fVar14 = *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4);
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + 4 + lVar11 * 2) * 2) =
               (short)iVar12;
          *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar6 * 2) =
               *(undefined2 *)(*(longlong *)(param_3 + 6) + 4 + lVar11 * 2);
          iVar12 = iVar12 + 1;
        }
        else {
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + 4 + lVar11 * 2) * 2) =
               (short)iVar12 + -1;
        }
        fVar3 = *(float *)(*(longlong *)(param_3 + 4) + 0xc + lVar11 * 4);
        if (4.9303807e-32 < ABS(fVar3 - fVar14)) {
          lVar6 = (longlong)iVar12;
          *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4) = fVar3;
          fVar14 = *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4);
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + 6 + lVar11 * 2) * 2) =
               (short)iVar12;
          *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar6 * 2) =
               *(undefined2 *)(*(longlong *)(param_3 + 6) + 6 + lVar11 * 2);
          iVar12 = iVar12 + 1;
        }
        else {
          *(short *)(*(longlong *)(param_3 + 8) +
                    (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + 6 + lVar11 * 2) * 2) =
               (short)iVar12 + -1;
        }
        lVar11 = lVar11 + 4;
      } while (lVar11 < lVar7 + -3);
    }
    for (; lVar11 < lVar7; lVar11 = lVar11 + 1) {
      fVar3 = *(float *)(*(longlong *)(param_3 + 4) + lVar11 * 4);
      if (4.9303807e-32 < ABS(fVar3 - fVar14)) {
        lVar6 = (longlong)iVar12;
        *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4) = fVar3;
        fVar14 = *(float *)(*(longlong *)(param_3 + 4) + lVar6 * 4);
        *(short *)(*(longlong *)(param_3 + 8) +
                  (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + lVar11 * 2) * 2) =
             (short)iVar12;
        *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar6 * 2) =
             *(undefined2 *)(*(longlong *)(param_3 + 6) + lVar11 * 2);
        iVar12 = iVar12 + 1;
      }
      else {
        *(short *)(*(longlong *)(param_3 + 8) +
                  (ulonglong)*(ushort *)(*(longlong *)(param_3 + 6) + lVar11 * 2) * 2) =
             (short)iVar12 + -1;
      }
    }
  }
  param_3[1] = iVar12;
  param_3[0xb] = (int)(((*(float **)(param_3 + 4))[(longlong)iVar12 + -1] -
                       **(float **)(param_3 + 4)) /
                      (float)((iVar12 + -1) * *(int *)(param_1 + 0x7c)));
  return;
}


```

## {'procedure': '0x180003550'}
Evidence: `ev_93900dcdfbf4362d4415fde79e60a122b72bcee3e3b0e1ffdbc175568f9702b7`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180003550(longlong param_1,longlong *param_2,undefined8 param_3,undefined8 param_4,
                  longlong param_5,Mat *param_6,int param_7)

{
  int *piVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  char *pcVar6;
  Mat *pMVar7;
  ulonglong uVar8;
  ulonglong uVar9;
  uint uVar10;
  code *pcVar11;
  float fVar12;
  undefined4 uVar13;
  undefined4 uVar14;
  undefined4 uVar15;
  undefined4 uVar16;
  undefined4 uVar17;
  undefined1 auStackY_228 [32];
  uint local_1e8;
  undefined4 local_1e4;
  undefined8 local_1e0;
  undefined8 local_1d8;
  longlong local_1c8;
  undefined4 uStack_1c0;
  undefined4 uStack_1bc;
  longlong local_1b8;
  undefined4 uStack_1b0;
  undefined4 uStack_1ac;
  longlong local_1a8;
  undefined8 uStack_1a0;
  undefined8 local_198;
  Mat local_188 [4];
  int local_184;
  undefined8 local_178;
  undefined8 local_170;
  undefined8 local_168;
  undefined8 local_160;
  longlong local_150;
  longlong local_148;
  undefined1 *local_140;
  undefined1 local_138 [32];
  longlong local_118;
  undefined8 uStack_110;
  longlong *local_108;
  undefined8 local_100;
  longlong local_f8;
  longlong local_f0;
  undefined8 local_e8;
  undefined8 local_d8;
  undefined8 uStack_d0;
  longlong local_c8;
  undefined8 uStack_c0;
  undefined8 local_b8;
  undefined8 uStack_b0;
  undefined8 local_a8;
  longlong lStack_a0;
  undefined8 *local_98;
  undefined8 *local_90;
  undefined8 local_88;
  undefined8 local_80;
  undefined4 local_78;
  ulonglong local_68;
  
  local_198 = 0xfffffffffffffffe;
  local_68 = DAT_18008b080 ^ (ulonglong)auStackY_228;
  iVar3 = *(int *)(param_1 + 0x14);
  iVar4 = **(int **)(param_5 + 0x40);
  iVar5 = (*(int **)(param_5 + 0x40))[1];
  if (param_7 == 3) {
    uVar16 = (undefined4)*(undefined8 *)(param_1 + 0xf8);
    uVar17 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0xf8) >> 0x20);
    uVar14 = (undefined4)*(undefined8 *)(param_1 + 0x100);
    uVar15 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0x100) >> 0x20);
  }
  else {
    if (param_7 != 4) {
      local_1e0 = 0;
      local_1d8 = 0;
      pcVar6 = cv::String::allocate((String *)&local_1e0,0x37);
      builtin_strncpy(pcVar6,"nEFIType == EFI_FAST_SWEEP || nEFIType == EFI_FAST_FINE",0x37);
      cv::error(-0xd7,(String *)&local_1e0,"sg::cpu::BlendEFICPU","SweepGuideCommon.cpp",0x49e);
      cv::String::deallocate((String *)&local_1e0);
    }
    uVar16 = (undefined4)*(undefined8 *)(param_1 + 0x108);
    uVar17 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0x108) >> 0x20);
    uVar14 = (undefined4)*(undefined8 *)(param_1 + 0x110);
    uVar15 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0x110) >> 0x20);
  }
  memset(&local_d8,0,0x68);
  local_d8 = 0x42ff0000;
  uStack_d0 = 0;
  local_c8 = 0;
  uStack_c0 = 0;
  local_b8 = 0;
  uStack_b0 = 0;
  local_a8 = 0;
  lStack_a0 = 0;
  local_98 = &uStack_d0;
  local_90 = &local_88;
  local_80 = 0;
  local_88 = 0;
  local_78 = 0;
  local_e8 = 0;
  local_100 = 0x20;
  local_f8 = 0;
  local_f0 = param_2[1] + *param_2;
  local_1e8 = *(uint *)(param_1 + 0x4dc);
  local_1e4 = *(undefined4 *)(param_1 + 0x4e0);
  local_108 = param_2;
  pMVar7 = (Mat *)FUN_18000e170((longlong *)&local_108,(uint *)local_188,&local_1e8,0,1);
  if ((Mat *)&local_d8 != pMVar7) {
    FUN_180001780((Mat *)&local_d8,pMVar7);
  }
  local_78 = *(undefined4 *)(pMVar7 + 0x60);
  if (local_150 != 0) {
    LOCK();
    piVar1 = (int *)(local_150 + 0x14);
    iVar2 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar2 == 1) {
      cv::Mat::deallocate(local_188);
    }
  }
  uVar8 = 0;
  local_150 = 0;
  local_178 = 0;
  local_160 = 0;
  local_168 = 0;
  local_170 = 0;
  uVar9 = uVar8;
  if (0 < local_184) {
    do {
      *(undefined4 *)(uVar8 + local_148) = 0;
      uVar10 = (int)uVar9 + 1;
      uVar8 = uVar8 + 4;
      uVar9 = (ulonglong)uVar10;
    } while ((int)uVar10 < local_184);
  }
  if (local_140 != local_138) {
    cv::fastFree(local_140);
  }
  local_1c8 = local_c8;
  uStack_1c0 = (undefined4)local_88;
  uStack_110 = CONCAT44(uStack_1bc,(undefined4)local_88);
  local_118 = local_c8;
  local_1b8 = *(longlong *)(param_5 + 0x10);
  uStack_1b0 = (undefined4)*(undefined8 *)(param_5 + 0x50);
  uStack_1a0 = CONCAT44(uStack_1ac,uStack_1b0);
  uVar13 = 0;
  fVar12 = (float)(double)CONCAT44(uVar15,uVar14);
  local_1a8 = local_1b8;
  FUN_1800033d0(iVar5,iVar4,&local_1a8,
                (float)((double)CONCAT44(uVar15,uVar14) / (double)CONCAT44(uVar17,uVar16)),fVar12,
                &local_118);
  pcVar11 = FUN_180009040;
  if (iVar3 == 3) {
    pcVar11 = FUN_180008f80;
  }
  (*pcVar11)(param_1,param_3,param_4,&local_d8);
  if (*(char *)(param_1 + 0x11c) == '\x01') {
    FUN_180005960(param_1,(double)CONCAT44(uVar13,fVar12),param_7);
    FUN_180005740(param_1,param_6,param_7);
  }
  local_108[1] = local_108[1] - local_f8;
  if (lStack_a0 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_a0 + 0x14);
    iVar3 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar3 == 1) {
      cv::Mat::deallocate((Mat *)&local_d8);
    }
  }
  uVar9 = 0;
  lStack_a0 = 0;
  local_c8 = 0;
  uStack_b0 = 0;
  local_b8 = 0;
  uStack_c0 = 0;
  uVar8 = uVar9;
  if (0 < local_d8._4_4_) {
    do {
      *(undefined4 *)(uVar8 + (longlong)local_98) = 0;
      uVar10 = (int)uVar9 + 1;
      uVar9 = (ulonglong)uVar10;
      uVar8 = uVar8 + 4;
    } while ((int)uVar10 < local_d8._4_4_);
  }
  if (local_90 != &local_88) {
    cv::fastFree(local_90);
  }
  return;
}


```

## {'procedure': '0x180012b70'}
Evidence: `ev_95adec5bb101242d5660b807890989f7587d5f12c237cd164e7fa36be4fc5651`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180012b70(longlong param_1,longlong *param_2,longlong param_3,Mat *param_4,longlong param_5
                  ,int param_6)

{
  int *piVar1;
  double dVar2;
  undefined4 uVar3;
  longlong lVar4;
  bool bVar5;
  longlong lVar6;
  undefined8 uVar7;
  undefined8 uVar8;
  undefined8 uVar9;
  undefined8 uVar10;
  undefined8 uVar11;
  undefined8 uVar12;
  undefined8 uVar13;
  Mat *pMVar14;
  char *pcVar15;
  longlong lVar16;
  longlong lVar17;
  undefined1 *puVar18;
  uint uVar19;
  ulonglong uVar20;
  ulonglong uVar21;
  longlong lVar22;
  ulonglong uVar23;
  float *pfVar24;
  longlong lVar25;
  longlong lVar26;
  longlong lVar27;
  longlong lVar28;
  int iVar29;
  longlong lVar30;
  int iVar31;
  int iVar32;
  int iVar33;
  float fVar34;
  float fVar35;
  undefined4 uVar36;
  undefined4 uVar37;
  undefined1 auStackY_928 [32];
  undefined8 local_8e8;
  int local_8e0;
  int local_8dc;
  int local_8d8;
  int local_8d4;
  int local_8d0;
  int local_8cc;
  int local_8c8;
  int local_8c4;
  longlong local_8c0;
  longlong local_8b8;
  undefined4 uStack_8b0;
  undefined4 uStack_8ac;
  longlong local_8a8;
  undefined4 uStack_8a0;
  longlong local_898;
  undefined4 uStack_890;
  undefined4 uStack_88c;
  longlong local_888;
  undefined4 uStack_880;
  undefined4 uStack_87c;
  longlong local_878;
  undefined4 uStack_870;
  longlong local_868;
  undefined4 uStack_860;
  undefined4 uStack_85c;
  longlong local_858;
  int iStack_850;
  undefined8 local_848;
  undefined8 local_840;
  Mat *local_838;
  undefined4 local_830 [2];
  longlong local_828;
  undefined8 local_820;
  undefined4 local_818 [2];
  longlong local_810;
  undefined8 local_808;
  longlong local_7f8;
  undefined4 uStack_7f0;
  undefined4 uStack_7ec;
  longlong local_7e8;
  undefined4 uStack_7e0;
  undefined4 uStack_7dc;
  Mat local_7d8 [4];
  int local_7d4;
  undefined8 local_7c8;
  undefined8 local_7c0;
  undefined8 local_7b8;
  undefined8 local_7b0;
  longlong local_7a0;
  longlong local_798;
  undefined1 *local_790;
  undefined1 local_788 [16];
  undefined4 local_778;
  Mat local_768 [4];
  int local_764;
  undefined8 local_758;
  undefined8 local_750;
  undefined8 local_748;
  undefined8 local_740;
  longlong local_730;
  longlong local_728;
  undefined1 *local_720;
  undefined1 local_718 [16];
  undefined4 local_708;
  longlong local_6f8;
  int local_6f0;
  longlong local_6e8;
  undefined8 uStack_6e0;
  longlong local_6d8;
  undefined8 uStack_6d0;
  longlong local_6c8;
  int local_6c0;
  longlong local_6b8 [2];
  longlong local_6a8;
  int local_6a0;
  longlong local_698 [2];
  longlong local_688 [2];
  longlong local_678;
  undefined8 uStack_670;
  longlong local_668;
  undefined8 uStack_660;
  longlong local_658;
  undefined8 uStack_650;
  undefined8 local_648;
  longlong local_638;
  undefined8 uStack_630;
  longlong local_628;
  undefined8 uStack_620;
  undefined8 local_618;
  undefined8 uStack_610;
  longlong local_608;
  undefined8 uStack_600;
  undefined8 local_5f8;
  undefined8 uStack_5f0;
  undefined8 local_5e8;
  undefined8 uStack_5e0;
  longlong local_5d8;
  undefined8 uStack_5d0;
  undefined8 local_5c8;
  undefined8 uStack_5c0;
  undefined8 local_5b8;
  undefined8 uStack_5b0;
  longlong *local_5a8;
  undefined8 local_5a0;
  longlong local_598;
  longlong local_590;
  undefined8 local_588;
  undefined8 local_578;
  undefined8 uStack_570;
  longlong local_568;
  undefined8 uStack_560;
  undefined8 local_558;
  undefined8 uStack_550;
  undefined8 local_548;
  longlong lStack_540;
  undefined8 *local_538;
  undefined8 *local_530;
  undefined8 local_528;
  undefined8 local_520;
  undefined4 local_518;
  undefined8 local_508;
  undefined8 uStack_500;
  longlong local_4f8;
  undefined8 uStack_4f0;
  undefined8 local_4e8;
  undefined8 uStack_4e0;
  undefined8 local_4d8;
  longlong lStack_4d0;
  undefined8 *local_4c8;
  undefined8 *local_4c0;
  undefined8 local_4b8;
  undefined8 local_4b0;
  undefined4 local_4a8;
  undefined8 local_498;
  undefined8 uStack_490;
  undefined8 local_488;
  undefined8 uStack_480;
  undefined8 local_478;
  undefined8 uStack_470;
  undefined8 local_468;
  longlong lStack_460;
  undefined8 *local_458;
  undefined8 *local_450;
  undefined8 local_448;
  undefined8 local_440;
  undefined4 local_438;
  undefined4 local_428;
  int local_424;
  undefined4 local_420;
  undefined4 local_41c;
  longlong local_418;
  undefined8 local_410;
  undefined8 local_408;
  undefined8 local_400;
  undefined8 local_3f8;
  longlong local_3f0;
  undefined4 *local_3e8;
  undefined8 *local_3e0;
  undefined8 local_3d8;
  undefined8 local_3d0;
  undefined4 local_3c8;
  undefined4 local_3c0;
  int local_3bc;
  undefined4 local_3b8;
  undefined4 local_3b4;
  longlong local_3b0;
  undefined8 local_3a8;
  undefined8 local_3a0;
  undefined8 local_398;
  undefined8 local_390;
  longlong local_388;
  undefined4 *local_380;
  undefined8 *local_378;
  undefined8 local_370;
  undefined8 local_368;
  undefined4 local_360;
  undefined4 local_358;
  int local_354;
  undefined4 local_350;
  undefined4 local_34c;
  longlong local_348;
  undefined8 local_340;
  undefined8 local_338;
  undefined8 local_330;
  undefined8 local_328;
  longlong local_320;
  undefined8 *local_310;
  int local_308;
  undefined4 local_2f8;
  undefined4 local_2f0;
  int local_2ec;
  undefined4 local_2e8;
  undefined4 local_2e4;
  longlong local_2e0;
  undefined8 local_2d8;
  undefined8 local_2d0;
  undefined8 local_2c8;
  undefined8 local_2c0;
  longlong local_2b8;
  undefined8 *local_2a8;
  undefined4 local_2a0;
  undefined4 local_290;
  undefined4 local_288;
  int local_284;
  undefined4 local_280;
  undefined4 local_27c;
  longlong local_278;
  undefined8 local_270;
  undefined8 local_268;
  undefined8 local_260;
  undefined8 local_258;
  longlong local_250;
  undefined4 *local_248;
  undefined8 *local_240;
  undefined8 local_238;
  undefined8 local_230;
  undefined4 local_228;
  undefined4 local_220;
  int local_21c;
  undefined4 local_218;
  undefined4 local_214;
  longlong local_210;
  undefined8 local_208;
  undefined8 local_200;
  undefined8 local_1f8;
  undefined8 local_1f0;
  longlong local_1e8;
  undefined4 *local_1e0;
  undefined8 *local_1d8;
  undefined8 local_1d0;
  undefined8 local_1c8;
  undefined4 local_1c0;
  undefined4 local_1b8;
  int local_1b4;
  undefined4 local_1b0;
  undefined4 local_1ac;
  longlong local_1a8;
  undefined8 local_1a0;
  undefined8 local_198;
  undefined8 local_190;
  undefined8 local_188;
  longlong local_180;
  undefined8 *local_170;
  int local_168;
  undefined4 local_158;
  undefined4 local_150;
  int local_14c;
  undefined4 local_148;
  undefined4 local_144;
  longlong local_140;
  undefined8 local_138;
  undefined8 local_130;
  undefined8 local_128;
  undefined8 local_120;
  longlong local_118;
  undefined8 *local_108;
  undefined4 local_100;
  undefined4 local_f0;
  ulonglong local_e8;
  
  local_648 = 0xfffffffffffffffe;
  local_e8 = DAT_18008b080 ^ (ulonglong)auStackY_928;
  local_8d0 = *(int *)(param_1 + 0x5e4);
  local_8d8 = *(int *)(param_1 + 0x5e8);
  local_8cc = *(int *)(param_1 + 0x5ec);
  lVar25 = (longlong)local_8cc;
  local_8e0 = *(int *)(param_1 + 0x5f0);
  iVar33 = 0;
  local_8c0 = param_3;
  local_888 = param_1;
  local_838 = param_4;
  if (param_6 == 0) {
    uVar36 = (undefined4)*(undefined8 *)(param_1 + 0x80);
    uVar37 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0x80) >> 0x20);
  }
  else {
    if (param_6 != 1) {
      local_848 = 0;
      local_840 = 0;
      pcVar15 = cv::String::allocate((String *)&local_848,0x31);
      builtin_strncpy(pcVar15,"nDepthType == DEPTH_3D || nDepthType == DEPTH_EFI",0x31);
      cv::error(-0xd7,(String *)&local_848,"sg::cpu::InterpTextureLessCPU","SweepGuideCpu.cpp",0x171
               );
      cv::String::deallocate((String *)&local_848);
    }
    uVar36 = (undefined4)*(undefined8 *)(param_1 + 0x88);
    uVar37 = (undefined4)((ulonglong)*(undefined8 *)(param_1 + 0x88) >> 0x20);
  }
  dVar2 = *(double *)(param_1 + 0xa0);
  local_8dc = *(int *)(param_1 + 0x9c);
  FUN_1800726b8((longlong)&local_358,0x68,2,&LAB_180006700);
  FUN_1800726b8((longlong)&local_1b8,0x68,2,&LAB_180006700);
  memset(&local_508,0,0x68);
  local_508 = 0x42ff0000;
  uStack_500 = 0;
  local_4f8 = 0;
  uStack_4f0 = 0;
  local_4e8 = 0;
  uStack_4e0 = 0;
  local_4d8 = 0;
  lStack_4d0 = 0;
  local_4c8 = &uStack_500;
  local_4c0 = &local_4b8;
  local_4b0 = 0;
  local_4b8 = 0;
  local_4a8 = 0;
  local_588 = 0;
  local_5a0 = 0x20;
  local_598 = 0;
  local_590 = param_2[1] + *param_2;
  local_5a8 = param_2;
  FUN_180023f10((longlong *)&local_5a8,param_1,(Mat *)&local_358,(Mat *)&local_1b8,(Mat *)&local_508
               );
  memset(&local_578,0,0x68);
  local_578 = 0x42ff0000;
  uStack_570 = 0;
  local_568 = 0;
  uStack_560 = 0;
  local_558 = 0;
  uStack_550 = 0;
  local_548 = 0;
  lStack_540 = 0;
  local_538 = &uStack_570;
  local_530 = &local_528;
  local_520 = 0;
  local_528 = 0;
  local_518 = 0;
  memset(&local_498,0,0x68);
  local_498 = 0x42ff0000;
  uStack_490 = 0;
  local_488 = 0;
  uStack_480 = 0;
  local_478 = 0;
  uStack_470 = 0;
  local_468 = 0;
  lStack_460 = 0;
  local_458 = &uStack_490;
  local_450 = &local_448;
  local_440 = 0;
  local_448 = 0;
  local_438 = 0;
  local_878 = local_4f8;
  local_6c8 = local_4f8;
  local_8c4 = (int)local_4b8;
  local_6c0 = (int)local_4b8;
  local_8b8 = local_1a8;
  local_6a8 = local_1a8;
  local_8c8 = local_168;
  local_6a0 = local_168;
  local_8a8 = local_348;
  local_6f8 = local_348;
  local_8d4 = local_308;
  local_6f0 = local_308;
  lVar26 = *(longlong *)(param_5 + 0x10);
  uStack_7e0 = (undefined4)*(undefined8 *)(param_5 + 0x50);
  lVar4 = *(longlong *)(param_4 + 0x10);
  uStack_7f0 = (undefined4)*(undefined8 *)(param_4 + 0x50);
  lVar6 = *(longlong *)(param_1 + 0x608);
  uVar7 = *(undefined8 *)(param_1 + 0x610);
  uVar8 = *(undefined8 *)(param_1 + 0x618);
  uVar9 = *(undefined8 *)(param_1 + 0x620);
  uVar10 = *(undefined8 *)(param_1 + 0x628);
  uVar11 = *(undefined8 *)(param_1 + 0x630);
  local_8e8 = 0;
  lVar16 = (longlong)local_8e0;
  local_858 = lVar16;
  local_7f8 = lVar4;
  local_7e8 = lVar26;
  if (0 < lVar16) {
    uVar13 = CONCAT44(uStack_7dc,uStack_7e0);
    uVar12 = CONCAT44(uStack_7ec,uStack_7f0);
    lVar22 = local_1a8;
    lVar28 = local_4f8;
    lVar30 = local_348;
    iVar29 = (int)local_4b8;
    iVar31 = local_308;
    do {
      iVar32 = 0;
      lVar27 = 0;
      if (0 < lVar25) {
        pfVar24 = (float *)(local_8e8 * local_168 + lVar22);
        lVar17 = (((longlong)iVar31 - (longlong)local_168) * local_8e8 - lVar22) + lVar30;
        local_868 = iVar29 * local_8e8 + lVar28;
        local_898 = lVar17;
        local_6d8 = lVar26;
        uStack_6d0 = uVar13;
        local_638 = lVar4;
        uStack_630 = uVar12;
        local_608 = lVar6;
        uStack_600 = uVar7;
        local_5f8 = uVar8;
        uStack_5f0 = uVar9;
        local_5e8 = uVar10;
        uStack_5e0 = uVar11;
        local_5d8 = lVar6;
        uStack_5d0 = uVar7;
        local_5c8 = uVar8;
        uStack_5c0 = uVar9;
        local_5b8 = uVar10;
        uStack_5b0 = uVar11;
        do {
          fVar34 = FUN_180023d80(iVar32,iVar33,&local_5d8,&local_6d8);
          fVar35 = FUN_180023d80(iVar32,iVar33,&local_608,&local_638);
          fVar35 = fVar35 + 1.0;
          bVar5 = (float)(double)CONCAT44(uVar37,uVar36) <= fVar34;
          if (!bVar5) {
            fVar35 = 0.0;
          }
          *(float *)(lVar17 + (longlong)pfVar24) = fVar35;
          *pfVar24 = (float)bVar5;
          *(bool *)(lVar27 + local_868) = bVar5;
          iVar32 = iVar32 + 1;
          lVar27 = lVar27 + 1;
          pfVar24 = pfVar24 + 1;
          lVar16 = local_858;
          lVar22 = local_8b8;
          lVar28 = local_878;
          lVar30 = local_8a8;
          iVar29 = local_8c4;
          local_168 = local_8c8;
          iVar31 = local_8d4;
        } while (lVar27 < lVar25);
      }
      iVar33 = iVar33 + 1;
      local_8e8 = local_8e8 + 1;
      param_1 = local_888;
    } while (local_8e8 < lVar16);
  }
  if (local_8dc < 1) {
    FUN_180001780((Mat *)&local_578,(Mat *)&local_358);
    local_518 = local_2f8;
    FUN_180001780((Mat *)&local_498,(Mat *)&local_1b8);
    local_438 = local_158;
  }
  else {
    while( true ) {
      local_288 = local_358;
      local_284 = local_354;
      local_280 = local_350;
      local_27c = local_34c;
      local_278 = local_348;
      local_270 = local_340;
      local_268 = local_338;
      local_260 = local_330;
      local_258 = local_328;
      local_250 = local_320;
      local_248 = &local_280;
      local_240 = &local_238;
      local_230 = 0;
      local_238 = 0;
      if (local_320 != 0) {
        LOCK();
        *(int *)(local_320 + 0x14) = *(int *)(local_320 + 0x14) + 1;
        UNLOCK();
      }
      if (local_354 < 3) {
        local_238 = *local_310;
        local_230 = local_310[1];
      }
      else {
        local_284 = 0;
        cv::Mat::copySize((Mat *)&local_288,(Mat *)&local_358);
      }
      local_228 = local_2f8;
      local_220 = local_1b8;
      local_21c = local_1b4;
      local_218 = local_1b0;
      local_214 = local_1ac;
      local_210 = local_1a8;
      local_208 = local_1a0;
      local_200 = local_198;
      local_1f8 = local_190;
      local_1f0 = local_188;
      local_1e8 = local_180;
      local_1e0 = &local_218;
      local_1d8 = &local_1d0;
      local_1c8 = 0;
      local_1d0 = 0;
      if (local_180 != 0) {
        LOCK();
        *(int *)(local_180 + 0x14) = *(int *)(local_180 + 0x14) + 1;
        UNLOCK();
      }
      if (local_1b4 < 3) {
        local_1d0 = *local_170;
        local_1c8 = local_170[1];
      }
      else {
        local_21c = 0;
        cv::Mat::copySize((Mat *)&local_220,(Mat *)&local_1b8);
      }
      local_1c0 = local_158;
      local_428 = local_2f0;
      local_424 = local_2ec;
      local_420 = local_2e8;
      local_41c = local_2e4;
      local_418 = local_2e0;
      local_410 = local_2d8;
      local_408 = local_2d0;
      local_400 = local_2c8;
      local_3f8 = local_2c0;
      local_3f0 = local_2b8;
      local_3e8 = &local_420;
      local_3e0 = &local_3d8;
      local_3d0 = 0;
      local_3d8 = 0;
      if (local_2b8 != 0) {
        LOCK();
        *(int *)(local_2b8 + 0x14) = *(int *)(local_2b8 + 0x14) + 1;
        UNLOCK();
      }
      if (local_2ec < 3) {
        local_3d8 = *local_2a8;
        local_3d0 = local_2a8[1];
      }
      else {
        local_424 = 0;
        cv::Mat::copySize((Mat *)&local_428,(Mat *)&local_2f0);
      }
      local_3c8 = local_290;
      local_3c0 = local_150;
      local_3bc = local_14c;
      local_3b8 = local_148;
      local_3b4 = local_144;
      local_3b0 = local_140;
      local_3a8 = local_138;
      local_3a0 = local_130;
      local_398 = local_128;
      local_390 = local_120;
      local_388 = local_118;
      local_380 = &local_3b8;
      local_378 = &local_370;
      local_368 = 0;
      local_370 = 0;
      if (local_118 != 0) {
        LOCK();
        *(int *)(local_118 + 0x14) = *(int *)(local_118 + 0x14) + 1;
        UNLOCK();
      }
      if (local_14c < 3) {
        local_370 = *local_108;
        local_368 = local_108[1];
      }
      else {
        local_3bc = 0;
        cv::Mat::copySize((Mat *)&local_3c0,(Mat *)&local_150);
      }
      local_360 = local_f0;
      uVar3 = *(undefined4 *)(param_1 + 0x2a0);
      lVar26 = 2;
      lVar25 = 0;
      do {
        local_820 = 0;
        local_830[0] = 0x2010000;
        local_828 = (longlong)&local_428 + lVar25;
        local_808 = 0;
        local_818[0] = 0x1010000;
        local_810 = (longlong)&local_288 + lVar25;
        local_8e8 = CONCAT44(uVar3,uVar3);
        cv::GaussianBlur(local_818,local_830,&local_8e8,0);
        lVar25 = lVar25 + 0x68;
        lVar26 = lVar26 + -1;
      } while (lVar26 != 0);
      local_888 = local_4f8;
      uStack_880 = (int)local_4b8;
      uStack_660 = CONCAT44(uStack_87c,(int)local_4b8);
      local_668 = local_4f8;
      local_8a8 = local_140;
      uStack_8a0 = local_100;
      local_6b8[0] = local_140;
      local_878 = local_2e0;
      uStack_870 = local_2a0;
      local_688[0] = local_2e0;
      local_858 = local_348;
      iStack_850 = local_308;
      local_698[0] = local_348;
      fVar34 = FUN_1800121b0(local_8cc,local_8e0,local_698,local_688,local_6b8,&local_668);
      if ((double)fVar34 < dVar2) break;
      FUN_1800241f0((Mat *)&local_358,(Mat *)&local_2f0);
      FUN_1800241f0((Mat *)&local_1b8,(Mat *)&local_150);
      _eh_vector_destructor_iterator_(&local_428,0x68,2,FUN_180001750);
      _eh_vector_destructor_iterator_(&local_288,0x68,2,FUN_180001750);
    }
    FUN_180001780((Mat *)&local_578,(Mat *)&local_2f0);
    local_518 = local_290;
    FUN_180001780((Mat *)&local_498,(Mat *)&local_150);
    local_438 = local_f0;
    _eh_vector_destructor_iterator_(&local_428,0x68,2,FUN_180001750);
    _eh_vector_destructor_iterator_(&local_288,0x68,2,FUN_180001750);
  }
  pMVar14 = local_838;
  uVar21 = 0;
  local_8b8 = *(longlong *)(local_838 + 0x10);
  uStack_8b0 = (undefined4)*(undefined8 *)(local_838 + 0x50);
  uStack_670 = CONCAT44(uStack_8ac,uStack_8b0);
  local_868 = *(longlong *)(param_5 + 0x10);
  uStack_860 = (undefined4)*(undefined8 *)(param_5 + 0x50);
  uStack_650 = CONCAT44(uStack_85c,uStack_860);
  local_898 = local_568;
  uStack_890 = (undefined4)local_528;
  uStack_6e0 = CONCAT44(uStack_88c,(undefined4)local_528);
  local_6e8 = local_568;
  local_628 = *(longlong *)(param_1 + 0x690);
  uStack_620 = *(undefined8 *)(param_1 + 0x698);
  local_618 = *(undefined8 *)(param_1 + 0x6a0);
  uStack_610 = *(undefined8 *)(param_1 + 0x6a8);
  local_678 = local_8b8;
  local_658 = local_868;
  FUN_1800124d0(local_8d0,local_8d8,&local_628,(float)(double)CONCAT44(uVar37,uVar36),&local_6e8,
                &local_658,&local_678);
  if (*(char *)(param_1 + 0x11c) == '\x01') {
    if (param_6 == 0) {
      FUN_1800013f0(local_768,pMVar14);
      local_708 = *(undefined4 *)(pMVar14 + 0x60);
      FUN_180004b70(param_1,local_8c0,local_768,5);
      if (local_730 != 0) {
        LOCK();
        piVar1 = (int *)(local_730 + 0x14);
        iVar33 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar33 == 1) {
          cv::Mat::deallocate(local_768);
        }
      }
      local_730 = 0;
      local_758 = 0;
      local_740 = 0;
      local_748 = 0;
      local_750 = 0;
      uVar20 = uVar21;
      uVar23 = uVar21;
      if (0 < local_764) {
        do {
          *(undefined4 *)(uVar23 + local_728) = 0;
          uVar19 = (int)uVar20 + 1;
          uVar20 = (ulonglong)uVar19;
          uVar23 = uVar23 + 4;
        } while ((int)uVar19 < local_764);
      }
      puVar18 = local_718;
    }
    else {
      if (param_6 != 1) goto LAB_180013a30;
      FUN_1800013f0(local_7d8,pMVar14);
      local_778 = *(undefined4 *)(pMVar14 + 0x60);
      FUN_180004b70(param_1,local_8c0,local_7d8,10);
      if (local_7a0 != 0) {
        LOCK();
        piVar1 = (int *)(local_7a0 + 0x14);
        iVar33 = *piVar1;
        *piVar1 = *piVar1 + -1;
        UNLOCK();
        if (iVar33 == 1) {
          cv::Mat::deallocate(local_7d8);
        }
      }
      local_7a0 = 0;
      local_7c8 = 0;
      local_7b0 = 0;
      local_7b8 = 0;
      local_7c0 = 0;
      uVar20 = uVar21;
      uVar23 = uVar21;
      if (0 < local_7d4) {
        do {
          *(undefined4 *)(uVar23 + local_798) = 0;
          uVar19 = (int)uVar20 + 1;
          uVar20 = (ulonglong)uVar19;
          uVar23 = uVar23 + 4;
        } while ((int)uVar19 < local_7d4);
      }
      puVar18 = local_788;
      local_720 = local_790;
    }
    if (local_720 != puVar18) {
      cv::fastFree(local_720);
    }
  }
LAB_180013a30:
  if (lStack_460 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_460 + 0x14);
    iVar33 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar33 == 1) {
      cv::Mat::deallocate((Mat *)&local_498);
    }
  }
  lStack_460 = 0;
  local_488 = 0;
  uStack_470 = 0;
  local_478 = 0;
  uStack_480 = 0;
  uVar20 = uVar21;
  uVar23 = uVar21;
  if (0 < local_498._4_4_) {
    do {
      *(undefined4 *)(uVar23 + (longlong)local_458) = 0;
      uVar19 = (int)uVar20 + 1;
      uVar20 = (ulonglong)uVar19;
      uVar23 = uVar23 + 4;
    } while ((int)uVar19 < local_498._4_4_);
  }
  if (local_450 != &local_448) {
    cv::fastFree(local_450);
  }
  if (lStack_540 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_540 + 0x14);
    iVar33 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar33 == 1) {
      cv::Mat::deallocate((Mat *)&local_578);
    }
  }
  lStack_540 = 0;
  local_568 = 0;
  uStack_550 = 0;
  local_558 = 0;
  uStack_560 = 0;
  uVar20 = uVar21;
  uVar23 = uVar21;
  if (0 < local_578._4_4_) {
    do {
      *(undefined4 *)(uVar23 + (longlong)local_538) = 0;
      uVar19 = (int)uVar20 + 1;
      uVar20 = (ulonglong)uVar19;
      uVar23 = uVar23 + 4;
    } while ((int)uVar19 < local_578._4_4_);
  }
  if (local_530 != &local_528) {
    cv::fastFree(local_530);
  }
  local_5a8[1] = local_5a8[1] - local_598;
  if (lStack_4d0 != 0) {
    LOCK();
    piVar1 = (int *)(lStack_4d0 + 0x14);
    iVar33 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar33 == 1) {
      cv::Mat::deallocate((Mat *)&local_508);
    }
  }
  lStack_4d0 = 0;
  local_4f8 = 0;
  uStack_4e0 = 0;
  local_4e8 = 0;
  uStack_4f0 = 0;
  uVar20 = uVar21;
  if (0 < local_508._4_4_) {
    do {
      *(undefined4 *)(uVar20 + (longlong)local_4c8) = 0;
      uVar19 = (int)uVar21 + 1;
      uVar21 = (ulonglong)uVar19;
      uVar20 = uVar20 + 4;
    } while ((int)uVar19 < local_508._4_4_);
  }
  if (local_4c0 != &local_4b8) {
    cv::fastFree(local_4c0);
  }
  _eh_vector_destructor_iterator_(&local_1b8,0x68,2,FUN_180001750);
  _eh_vector_destructor_iterator_(&local_358,0x68,2,FUN_180001750);
  return;
}


```

## {'procedure': '0x180024c90'}
Evidence: `ev_9ab8b0d2afa0a3a2e8bb58e8a19e4a745fbd95db198a960dd69ec759d98baea2`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180024c90(longlong param_1,float param_2,int param_3,undefined8 param_4,longlong param_5,
                  longlong param_6,int param_7,longlong param_8,longlong param_9,longlong param_10,
                  longlong param_11,longlong param_12,longlong param_13,undefined8 param_14,
                  longlong *param_15,longlong *param_16)

{
  ushort uVar1;
  ushort uVar2;
  longlong lVar3;
  int iVar4;
  uint uVar5;
  float *pfVar6;
  longlong lVar7;
  longlong lVar8;
  float *pfVar9;
  uint uVar10;
  longlong lVar11;
  longlong lVar12;
  ulonglong uVar13;
  float *pfVar14;
  float *pfVar15;
  int iVar16;
  ushort *puVar17;
  short sVar18;
  uint uVar19;
  float *pfVar20;
  longlong lVar21;
  longlong lVar22;
  float fVar23;
  float fVar24;
  undefined1 auStackY_128 [32];
  int local_f8;
  int local_f4;
  undefined4 uStack_9c;
  longlong local_88;
  undefined8 uStack_80;
  float local_78 [2];
  longlong *local_70;
  undefined8 local_68;
  ulonglong local_60;
  float *local_58;
  undefined8 local_50;
  ulonglong local_48;
  
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_128;
  iVar16 = (int)param_4;
  lVar21 = (longlong)iVar16;
  local_70 = (longlong *)0x0;
  local_68 = 0;
  local_60 = 0;
  local_58 = (float *)0x0;
  local_50 = 0;
  iVar4 = omp_get_thread_num();
  local_70 = (longlong *)(*(longlong *)(param_1 + 0x30) + (longlong)iVar4 * 0x18);
  local_68 = CONCAT44(local_68._4_4_,0x20);
  pfVar20 = (float *)(*local_70 + local_70[1]);
  local_60 = (longlong)(iVar16 * 4) * 4 + 0x1fU & 0xffffffffffffffe0;
  lVar3 = local_70[1];
  local_70[1] = local_60 + lVar3;
  local_58 = pfVar20;
  if (local_60 + lVar3 <= (ulonglong)local_70[2]) {
    local_f8 = 0;
    fVar24 = param_2;
    if (0 < param_3) {
      do {
        local_f4 = 0;
        if (0 < (int)param_2) {
          do {
            lVar3 = (longlong)
                    ((int)((ulonglong)(longlong)*(int *)(param_5 + 8) >> 2) * local_f8 + local_f4) *
                    4;
            fVar23 = (float)rintf();
            lVar22 = (longlong)(int)fVar23;
            uVar1 = *(ushort *)(param_8 + lVar22 * 2);
            local_78[0] = (float)(uint)uVar1;
            lVar12 = 0;
            if (3 < lVar21) {
              lVar11 = (longlong)param_7;
              pfVar9 = pfVar20 + 2;
              pfVar14 = (float *)(lVar11 * (lVar21 + -2) + lVar3 + param_6);
              lVar7 = lVar21 - (lVar21 + -2);
              lVar8 = (lVar21 - 4U >> 2) + 1;
              lVar12 = lVar8 * 4;
              do {
                pfVar9[-2] = *(float *)((longlong)pfVar14 + (lVar7 + -1) * lVar11);
                pfVar9[-1] = *pfVar14;
                *pfVar9 = *(float *)((longlong)pfVar14 + (lVar7 + -3) * lVar11);
                pfVar9[1] = *(float *)((longlong)pfVar14 + (lVar7 + -4) * lVar11);
                pfVar9 = pfVar9 + 4;
                pfVar14 = pfVar14 + -lVar11;
                lVar8 = lVar8 + -1;
              } while (lVar8 != 0);
            }
            if (lVar12 < lVar21) {
              pfVar9 = (float *)(((lVar21 - lVar12) + -1) * (longlong)param_7 + lVar3 + param_6);
              do {
                pfVar20[lVar12] = *pfVar9;
                lVar12 = lVar12 + 1;
                pfVar9 = (float *)((longlong)pfVar9 - (longlong)param_7);
              } while (lVar12 < lVar21);
            }
            uVar5 = (uint)*(ushort *)(param_9 + lVar22 * 2);
            uVar2 = *(ushort *)(param_10 + lVar22 * 2);
            if (uVar2 == 0) {
              local_78[0] = *pfVar20;
              lVar12 = 1;
              if (1 < lVar21) {
                if (3 < lVar21 + -1) {
                  pfVar9 = pfVar20 + 3;
                  lVar22 = (lVar21 - 5U >> 2) + 1;
                  lVar12 = lVar22 * 4 + 1;
                  do {
                    pfVar14 = local_78;
                    if (local_78[0] < pfVar9[-2]) {
                      pfVar14 = pfVar9 + -2;
                    }
                    local_78[0] = *pfVar14;
                    pfVar6 = local_78;
                    if (*pfVar14 < pfVar9[-1]) {
                      pfVar6 = pfVar9 + -1;
                    }
                    local_78[0] = *pfVar6;
                    pfVar14 = local_78;
                    if (*pfVar6 < *pfVar9) {
                      pfVar14 = pfVar9;
                    }
                    local_78[0] = *pfVar14;
                    pfVar6 = local_78;
                    if (*pfVar14 < pfVar9[1]) {
                      pfVar6 = pfVar9 + 1;
                    }
                    local_78[0] = *pfVar6;
                    pfVar9 = pfVar9 + 4;
                    lVar22 = lVar22 + -1;
                  } while (lVar22 != 0);
                }
                if (lVar12 < lVar21) {
                  pfVar9 = pfVar20 + lVar12;
                  lVar12 = lVar21 - lVar12;
                  do {
                    pfVar14 = local_78;
                    if (local_78[0] < *pfVar9) {
                      pfVar14 = pfVar9;
                    }
                    local_78[0] = *pfVar14;
                    pfVar9 = pfVar9 + 1;
                    lVar12 = lVar12 + -1;
                  } while (lVar12 != 0);
                }
              }
              sVar18 = 0;
            }
            else {
              uVar13 = (ulonglong)uVar5;
              pfVar9 = pfVar20 + lVar21;
              pfVar14 = pfVar20 + lVar21 * 2;
              pfVar6 = pfVar20 + lVar21 * 3;
              uStack_80 = CONCAT44(uStack_9c,(uint)uVar1);
              local_88 = param_11;
              FUN_180013fd0(iVar16,&local_88,(longlong)pfVar20,pfVar9,pfVar14,(longlong)pfVar6);
              iVar4 = (uint)*(ushort *)(param_12 + uVar13 * 2) - (int)local_78[0];
              fVar24 = *(float *)(param_13 + uVar13 * 4);
              if (iVar4 < 0) {
                iVar4 = 0;
                fVar24 = 0.0;
              }
              fVar24 = ((fVar24 * pfVar6[iVar4] + pfVar14[iVar4]) * fVar24 + pfVar9[iVar4]) * fVar24
                       + pfVar20[iVar4];
              uVar19 = 0;
              sVar18 = 0;
              uVar10 = 1;
              if (1 < uVar2) {
                pfVar15 = (float *)(param_13 + 4 + uVar13 * 4);
                puVar17 = (ushort *)(param_12 + uVar13 * 2);
                do {
                  puVar17 = puVar17 + 1;
                  iVar4 = (uint)*puVar17 - (int)local_78[0];
                  fVar23 = *pfVar15;
                  fVar23 = ((fVar23 * pfVar6[iVar4] + pfVar14[iVar4]) * fVar23 + pfVar9[iVar4]) *
                           fVar23 + pfVar20[iVar4];
                  if (fVar24 < fVar23) {
                    uVar19 = uVar10 & 0xffff;
                    fVar24 = fVar23;
                  }
                  sVar18 = (short)uVar19;
                  uVar10 = uVar10 + 1;
                  pfVar15 = pfVar15 + 1;
                } while ((int)uVar10 < (int)(uint)uVar2);
              }
            }
            *(float *)(lVar3 + *param_15) = (float)(int)((int)sVar18 + uVar5);
            *(float *)(lVar3 + *param_16) = fVar24;
            local_f4 = local_f4 + 1;
          } while (local_f4 < (int)param_2);
        }
        local_f8 = local_f8 + 1;
      } while (local_f8 < param_3);
    }
    local_70[1] = local_70[1] - local_60;
    return;
  }
                    /* WARNING: Subroutine does not return */
  FUN_180001b90("Allocating memory exceeding the limit ...",(longlong)iVar4 * 3,local_70,param_4);
}


```

## {'procedure': '0x180008f80'}
Evidence: `ev_9b962c6583d27fb5de564d0d5bf904ca2912a62b165260574c7d0d4b39d97b75`

```c

void FUN_180008f80(undefined8 param_1,longlong param_2,longlong param_3,longlong param_4,
                  longlong param_5,longlong param_6)

{
  longlong local_68;
  undefined8 uStack_60;
  longlong local_58;
  undefined8 uStack_50;
  longlong local_48;
  undefined8 uStack_40;
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  undefined8 local_18;
  undefined8 uStack_10;
  
  local_58 = *(longlong *)(param_5 + 0x10);
  local_18 = *(undefined8 *)(param_6 + 0x38);
  uStack_10 = *(undefined8 *)(param_6 + 0x40);
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_5 + 0x50));
  local_48 = *(longlong *)(param_4 + 0x10);
  uStack_50 = uStack_60;
  local_28 = *(longlong *)(param_6 + 0x28);
  uStack_20 = *(undefined8 *)(param_6 + 0x30);
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_4 + 0x50));
  local_38 = *(longlong *)(param_3 + 0x10);
  uStack_40 = uStack_60;
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_68 = *(longlong *)(param_2 + 0x10);
  uStack_30 = uStack_60;
  uStack_60 = CONCAT44(uStack_60._4_4_,*(undefined4 *)(param_2 + 0x50));
  FUN_18000e280(*(int *)(param_6 + 0xc),*(int *)(param_6 + 0x10),&local_68,&local_38,&local_48,
                &local_28,&local_58);
  return;
}


```

## {'procedure': 'EFI_Construct'}
Evidence: `ev_9d6b707e62a524ff038d0401beec2d2deb9bf5e31b4e8cb522725f795086635f`

```c

undefined8 EFI_Construct(longlong *param_1,longlong param_2,longlong param_3)

{
  undefined8 uVar1;
  
                    /* 0x6620  2  EFI_Construct */
  if (((param_1 == (longlong *)0x0) || (param_2 == 0)) || (param_3 == 0)) {
    uVar1 = 0x11;
  }
  else {
    uVar1 = (**(code **)(*param_1 + 8))();
  }
  return uVar1;
}


```

## {'procedure': '0x18001c3b0'}
Evidence: `ev_9e7bbde1e039a3c936cf904f0fd699a0c49207e81fb2e330c6c6a0af6e8e4aa4`

```c

void FUN_18001c3b0(longlong param_1,undefined8 param_2,longlong param_3,longlong param_4,
                  undefined4 *param_5)

{
  ulonglong uVar1;
  longlong lVar2;
  undefined4 local_res10;
  undefined4 uStackX_14;
  undefined4 local_58 [2];
  longlong local_50;
  undefined8 local_48;
  undefined4 local_40 [2];
  longlong local_38;
  undefined8 local_30;
  
  if (0 < (int)*(uint *)(param_1 + 0x28)) {
    lVar2 = param_4 - param_3;
    uVar1 = (ulonglong)*(uint *)(param_1 + 0x28);
    do {
      local_50 = lVar2 + param_3;
      local_48 = 0;
      local_58[0] = 0x2010000;
      _local_res10 = CONCAT44(*param_5,*param_5);
      local_30 = 0;
      local_40[0] = 0x1010000;
      local_38 = param_3;
      cv::GaussianBlur(local_40,local_58,&local_res10,0,0,1);
      param_3 = param_3 + 0x68;
      uVar1 = uVar1 - 1;
    } while (uVar1 != 0);
  }
  return;
}


```

## {'procedure': '0x1800124d0'}
Evidence: `ev_ab529eea9e1889b6b4cdb18d12785af07981c0801b179463a8ff4cdf99b054ca`

```c

void FUN_1800124d0(int param_1,int param_2,longlong *param_3,float param_4,longlong *param_5,
                  longlong *param_6,longlong *param_7)

{
  float *pfVar1;
  longlong lVar2;
  float fVar3;
  float fVar4;
  int iVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  longlong lVar12;
  
  if (0 < param_2) {
    lVar8 = (longlong)param_1;
    lVar9 = 0;
    do {
      lVar7 = 0;
      if (3 < lVar8) {
        do {
          lVar2 = lVar7 * 4;
          pfVar1 = (float *)((int)param_6[1] * lVar9 + lVar2 + *param_6);
          if (*pfVar1 <= param_4 && param_4 != *pfVar1) {
            lVar11 = (longlong)*(int *)(lVar2 + param_3[2]);
            iVar5 = *(int *)(param_3[3] + lVar9 * 4);
            fVar3 = *(float *)(lVar2 + *param_3);
            fVar4 = *(float *)(param_3[1] + lVar9 * 4);
            lVar10 = (longlong)(iVar5 * (int)param_5[1]) + *param_5;
            lVar6 = (longlong)((iVar5 + 1) * (int)param_5[1]) + *param_5;
            *(float *)((int)param_7[1] * lVar9 + lVar2 + *param_7) =
                 ((1.0 - fVar4) * fVar3 * *(float *)(lVar10 + 4 + lVar11 * 4) +
                  (1.0 - fVar3) * (1.0 - fVar4) * *(float *)(lVar10 + lVar11 * 4) +
                  (1.0 - fVar3) * fVar4 * *(float *)(lVar6 + lVar11 * 4) +
                 fVar4 * fVar3 * *(float *)(lVar6 + 4 + lVar11 * 4)) - 1.0;
          }
          pfVar1 = (float *)((int)param_6[1] * lVar9 + lVar2 + 4 + *param_6);
          if (*pfVar1 <= param_4 && param_4 != *pfVar1) {
            lVar11 = (longlong)*(int *)(lVar2 + 4 + param_3[2]);
            iVar5 = *(int *)(param_3[3] + lVar9 * 4);
            fVar3 = *(float *)(lVar2 + 4 + *param_3);
            fVar4 = *(float *)(param_3[1] + lVar9 * 4);
            lVar10 = (longlong)(iVar5 * (int)param_5[1]) + *param_5;
            lVar6 = (longlong)((iVar5 + 1) * (int)param_5[1]) + *param_5;
            *(float *)((int)param_7[1] * lVar9 + lVar2 + 4 + *param_7) =
                 ((1.0 - fVar4) * fVar3 * *(float *)(lVar10 + 4 + lVar11 * 4) +
                  (1.0 - fVar3) * (1.0 - fVar4) * *(float *)(lVar10 + lVar11 * 4) +
                  (1.0 - fVar3) * fVar4 * *(float *)(lVar6 + lVar11 * 4) +
                 fVar4 * fVar3 * *(float *)(lVar6 + 4 + lVar11 * 4)) - 1.0;
          }
          lVar6 = lVar2 + 8;
          pfVar1 = (float *)((int)param_6[1] * lVar9 + *param_6 + lVar6);
          if (*pfVar1 <= param_4 && param_4 != *pfVar1) {
            lVar12 = (longlong)*(int *)(lVar6 + param_3[2]);
            iVar5 = *(int *)(param_3[3] + lVar9 * 4);
            fVar3 = *(float *)(lVar6 + *param_3);
            fVar4 = *(float *)(param_3[1] + lVar9 * 4);
            lVar11 = (longlong)(iVar5 * (int)param_5[1]) + *param_5;
            lVar10 = (longlong)((iVar5 + 1) * (int)param_5[1]) + *param_5;
            *(float *)((int)param_7[1] * lVar9 + lVar6 + *param_7) =
                 ((1.0 - fVar4) * fVar3 * *(float *)(lVar11 + 4 + lVar12 * 4) +
                  (1.0 - fVar3) * (1.0 - fVar4) * *(float *)(lVar11 + lVar12 * 4) +
                  (1.0 - fVar3) * fVar4 * *(float *)(lVar10 + lVar12 * 4) +
                 fVar4 * fVar3 * *(float *)(lVar10 + 4 + lVar12 * 4)) - 1.0;
          }
          pfVar1 = (float *)((int)param_6[1] * lVar9 + *param_6 + 0xc + lVar2);
          if (*pfVar1 <= param_4 && param_4 != *pfVar1) {
            lVar11 = (longlong)*(int *)(lVar2 + 0xc + param_3[2]);
            iVar5 = *(int *)(param_3[3] + lVar9 * 4);
            fVar3 = *(float *)(lVar2 + 0xc + *param_3);
            fVar4 = *(float *)(param_3[1] + lVar9 * 4);
            lVar10 = (longlong)(iVar5 * (int)param_5[1]) + *param_5;
            lVar6 = (longlong)((iVar5 + 1) * (int)param_5[1]) + *param_5;
            *(float *)((int)param_7[1] * lVar9 + *param_7 + 0xc + lVar2) =
                 ((1.0 - fVar4) * fVar3 * *(float *)(lVar10 + 4 + lVar11 * 4) +
                  (1.0 - fVar3) * (1.0 - fVar4) * *(float *)(lVar10 + lVar11 * 4) +
                  (1.0 - fVar3) * fVar4 * *(float *)(lVar6 + lVar11 * 4) +
                 fVar4 * fVar3 * *(float *)(lVar6 + 4 + lVar11 * 4)) - 1.0;
          }
          lVar7 = lVar7 + 4;
        } while (lVar7 < lVar8 + -3);
      }
      for (; lVar7 < lVar8; lVar7 = lVar7 + 1) {
        lVar2 = lVar7 * 4;
        pfVar1 = (float *)((int)param_6[1] * lVar9 + lVar2 + *param_6);
        if (*pfVar1 <= param_4 && param_4 != *pfVar1) {
          lVar11 = (longlong)*(int *)(lVar2 + param_3[2]);
          iVar5 = *(int *)(param_3[3] + lVar9 * 4);
          fVar3 = *(float *)(lVar2 + *param_3);
          fVar4 = *(float *)(param_3[1] + lVar9 * 4);
          lVar10 = (longlong)(iVar5 * (int)param_5[1]) + *param_5;
          lVar6 = (longlong)((iVar5 + 1) * (int)param_5[1]) + *param_5;
          *(float *)((int)param_7[1] * lVar9 + lVar2 + *param_7) =
               ((1.0 - fVar4) * fVar3 * *(float *)(lVar10 + 4 + lVar11 * 4) +
                (1.0 - fVar3) * (1.0 - fVar4) * *(float *)(lVar10 + lVar11 * 4) +
                (1.0 - fVar3) * fVar4 * *(float *)(lVar6 + lVar11 * 4) +
               fVar4 * fVar3 * *(float *)(lVar6 + 4 + lVar11 * 4)) - 1.0;
        }
      }
      lVar9 = lVar9 + 1;
    } while (lVar9 < param_2);
  }
  return;
}


```

## {'procedure': '0x180016a80'}
Evidence: `ev_adb1fbdf82e530d22881b8a86027753dd49106614d3d3d36fa97a076f11b7060`

```c

void FUN_180016a80(int *param_1,longlong param_2,int param_3,longlong param_4,longlong param_5,
                  longlong param_6)

{
  int *piVar1;
  int iVar2;
  int iVar3;
  int *piVar4;
  char *pcVar5;
  longlong lVar6;
  undefined8 local_108;
  undefined8 local_100;
  undefined8 local_f8;
  undefined8 local_f0;
  undefined8 local_e8;
  undefined8 local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  longlong local_c8;
  undefined4 uStack_c0;
  undefined4 uStack_bc;
  longlong local_b8;
  undefined4 uStack_b0;
  undefined4 uStack_ac;
  longlong local_a8;
  undefined4 uStack_a0;
  undefined4 uStack_9c;
  longlong local_98;
  undefined4 uStack_90;
  undefined4 uStack_8c;
  longlong local_88;
  undefined8 uStack_80;
  longlong local_78;
  undefined8 uStack_70;
  longlong local_68;
  undefined8 uStack_60;
  undefined8 local_58;
  longlong local_48;
  undefined8 uStack_40;
  
  local_58 = 0xfffffffffffffffe;
  piVar1 = param_1 + param_1[4];
  for (; param_1 != piVar1; param_1 = param_1 + 1) {
    lVar6 = (longlong)*param_1 * 0x68;
    piVar4 = *(int **)(lVar6 + 0x40 + param_2);
    iVar2 = *piVar4;
    iVar3 = piVar4[1];
    piVar4 = *(int **)(lVar6 + 0x40 + param_6);
    if ((piVar4[1] != iVar3) || (*piVar4 != iVar2)) {
      local_108 = 0;
      local_100 = 0;
      pcVar5 = cv::String::allocate((String *)&local_108,0x2e);
      builtin_strncpy(pcVar5,"matUpdateMask[i].size() == matLapPyr[i].size()",0x2e);
      cv::error(-0xd7,(String *)&local_108,"sg::cpu::SearchMaxLapCPU","SweepGuideCpu.cpp",0x660);
      cv::String::deallocate((String *)&local_108);
    }
    if ((*(uint *)(lVar6 + param_6) & 0xfff) != 0) {
      local_f8 = 0;
      local_f0 = 0;
      pcVar5 = cv::String::allocate((String *)&local_f8,0x22);
      builtin_strncpy(pcVar5,"matUpdateMask[i].type() == CV_8UC1",0x22);
      cv::error(-0xd7,(String *)&local_f8,"sg::cpu::SearchMaxLapCPU","SweepGuideCpu.cpp",0x661);
      cv::String::deallocate((String *)&local_f8);
    }
    if (*(longlong *)(lVar6 + 0x50 + param_2) != *(longlong *)(lVar6 + 0x50 + param_5)) {
      local_d8 = 0;
      local_d0 = 0;
      pcVar5 = cv::String::allocate((String *)&local_d8,0x29);
      builtin_strncpy(pcVar5,"matLapPyr[i].step == matMaxLapPyr[i].step",0x29);
      cv::error(-0xd7,(String *)&local_d8,"sg::cpu::SearchMaxLapCPU","SweepGuideCpu.cpp",0x662);
      cv::String::deallocate((String *)&local_d8);
    }
    if (*(longlong *)(lVar6 + 0x50 + param_2) != *(longlong *)(lVar6 + 0x50 + param_4)) {
      local_e8 = 0;
      local_e0 = 0;
      pcVar5 = cv::String::allocate((String *)&local_e8,0x28);
      builtin_strncpy(pcVar5,"matLapPyr[i].step == matDepthPyr[i].step",0x28);
      cv::error(-0xd7,(String *)&local_e8,"sg::cpu::SearchMaxLapCPU","SweepGuideCpu.cpp",0x663);
      cv::String::deallocate((String *)&local_e8);
    }
    local_a8 = *(longlong *)(lVar6 + 0x10 + param_6);
    uStack_a0 = *(undefined4 *)(lVar6 + 0x50 + param_6);
    uStack_80 = CONCAT44(uStack_9c,uStack_a0);
    local_c8 = *(longlong *)(lVar6 + 0x10 + param_4);
    uStack_c0 = *(undefined4 *)(lVar6 + 0x50 + param_4);
    uStack_70 = CONCAT44(uStack_bc,uStack_c0);
    local_98 = *(longlong *)(lVar6 + 0x10 + param_5);
    uStack_90 = *(undefined4 *)(lVar6 + 0x50 + param_5);
    uStack_60 = CONCAT44(uStack_8c,uStack_90);
    local_b8 = *(longlong *)(lVar6 + 0x10 + param_2);
    uStack_b0 = *(undefined4 *)(lVar6 + 0x50 + param_2);
    uStack_40 = CONCAT44(uStack_ac,uStack_b0);
    local_88 = local_a8;
    local_78 = local_c8;
    local_68 = local_98;
    local_48 = local_b8;
    FUN_180016840(iVar3,iVar2,param_3,&local_48,&local_68,&local_78,&local_88);
  }
  return;
}


```

## {'procedure': '0x180015960'}
Evidence: `ev_af649763122711c16f1ee288880da4ec0e8854314b84967d38ed88fd3c4f7266`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180015960(longlong param_1,longlong param_2,longlong param_3)

{
  ulonglong uVar1;
  int iVar2;
  int iVar3;
  uint uVar4;
  longlong lVar5;
  float *pfVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  ulonglong uVar11;
  longlong lVar12;
  longlong lVar13;
  longlong lVar14;
  int iVar15;
  longlong lVar16;
  longlong *plVar17;
  int iVar18;
  int iVar19;
  double dVar20;
  uint uVar21;
  undefined8 uVar22;
  uint uVar23;
  float fVar24;
  undefined1 auStack_108 [32];
  int local_e8;
  int local_e4;
  longlong local_e0;
  int local_d8;
  int local_d4;
  int local_d0;
  longlong local_c8;
  longlong local_c0;
  longlong local_b8;
  longlong local_b0;
  undefined8 local_a8;
  longlong *local_a0;
  undefined8 local_98;
  longlong local_90;
  longlong local_88;
  undefined8 local_80;
  longlong local_78 [8];
  
  local_a8 = 0xfffffffffffffffe;
  local_78[3] = DAT_18008b080 ^ (ulonglong)auStack_108;
  iVar15 = **(int **)(param_2 + 0x40);
  iVar18 = (*(int **)(param_2 + 0x40))[1];
  iVar19 = **(int **)(param_3 + 0x40);
  iVar2 = (*(int **)(param_3 + 0x40))[1];
  local_a0 = (longlong *)0x0;
  local_98 = 0;
  local_90 = 0;
  local_88 = 0;
  local_80 = 0;
  local_e8 = iVar18;
  local_d4 = iVar19;
  local_d0 = iVar15;
  local_c0 = param_3;
  local_b0 = param_2;
  iVar3 = omp_get_thread_num();
  plVar17 = (longlong *)(*(longlong *)(param_1 + 0x30) + (longlong)iVar3 * 0x18);
  local_98 = CONCAT44(local_98._4_4_,0x20);
  local_c8 = 0;
  local_88 = plVar17[1] + *plVar17;
  lVar7 = 0;
  uVar11 = (longlong)(iVar18 * 2) * 4 + 0x1fU & 0xffffffffffffffe0;
  do {
    local_c8 = local_c8 + uVar11;
    lVar14 = *plVar17 + plVar17[1];
    uVar1 = plVar17[1] + uVar11;
    plVar17[1] = uVar1;
    local_a0 = plVar17;
    local_90 = local_c8;
    if ((ulonglong)plVar17[2] < uVar1) {
                    /* WARNING: Subroutine does not return */
      FUN_180001b90("Allocating memory exceeding the limit ...",lVar7,uVar11,lVar14);
    }
    local_78[lVar7] = lVar14;
    lVar7 = lVar7 + 1;
  } while (lVar7 < 3);
  iVar3 = 0;
  do {
    FUN_1800157e0((float *)((longlong)((iVar3 + -1) * *(int *)(param_2 + 0x50)) +
                           *(longlong *)(param_2 + 0x10)),local_78,iVar18);
    iVar3 = iVar3 + 1;
  } while (iVar3 < 2);
  local_d8 = 0;
  if (0 < iVar15) {
    lVar7 = 0;
    local_e4 = 0;
    local_e0 = 1;
    uVar22 = 0x7fffffffffffffff;
    fVar24 = 0.015625;
    do {
      lVar14 = local_e0;
      iVar15 = local_e4;
      local_d8 = local_d8 + 1;
      FUN_1800157e0((float *)((longlong)(local_d8 * *(int *)(param_2 + 0x50)) +
                             *(longlong *)(param_2 + 0x10)),local_78,iVar18);
      uVar21 = (uint)uVar22;
      uVar23 = (uint)((ulonglong)uVar22 >> 0x20);
      if (iVar15 + 1 < iVar19) {
        lVar13 = *(longlong *)(param_3 + 0x10) + **(longlong **)(param_3 + 0x48) * lVar7 * 2;
        lVar12 = lVar14 * **(longlong **)(param_3 + 0x48) + *(longlong *)(param_3 + 0x10);
        iVar19 = 0;
        lVar5 = 0;
        if (3 < iVar2) {
          pfVar6 = (float *)(lVar13 + 4);
          lVar16 = local_78[0] - lVar13;
          lVar8 = local_78[1] - lVar13;
          lVar10 = local_78[2] - lVar13;
          lVar9 = lVar12 - lVar13;
          uVar4 = (iVar2 - 4U >> 2) + 1;
          uVar11 = (ulonglong)uVar4;
          iVar19 = uVar4 * 4;
          lVar5 = uVar11 * 4;
          do {
            dVar20 = (double)(pfVar6[-1] -
                             (*(float *)(lVar8 + -4 + (longlong)pfVar6) * 6.0 +
                              *(float *)(lVar16 + -4 + (longlong)pfVar6) +
                             *(float *)(lVar10 + -4 + (longlong)pfVar6)) * fVar24);
            pfVar6[-1] = (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                                 SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)((longlong)pfVar6 + lVar9 + -4) -
                             (*(float *)(lVar10 + -4 + (longlong)pfVar6) +
                             *(float *)(lVar8 + -4 + (longlong)pfVar6)) * 4.0 * fVar24);
            *(float *)((longlong)pfVar6 + lVar9 + -4) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*pfVar6 -
                             (*(float *)(lVar8 + (longlong)pfVar6) * 6.0 +
                              *(float *)(lVar16 + (longlong)pfVar6) +
                             *(float *)(lVar10 + (longlong)pfVar6)) * fVar24);
            *pfVar6 = (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                              SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)((longlong)pfVar6 + lVar9) -
                             (*(float *)(lVar10 + (longlong)pfVar6) +
                             *(float *)(lVar8 + (longlong)pfVar6)) * 4.0 * fVar24);
            *(float *)((longlong)pfVar6 + lVar9) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(pfVar6[1] -
                             (*(float *)(lVar8 + 4 + (longlong)pfVar6) * 6.0 +
                              *(float *)(lVar16 + 4 + (longlong)pfVar6) +
                             *(float *)(lVar10 + 4 + (longlong)pfVar6)) * fVar24);
            pfVar6[1] = (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                                SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)((longlong)pfVar6 + lVar9 + 4) -
                             (*(float *)(lVar10 + 4 + (longlong)pfVar6) +
                             *(float *)(lVar8 + 4 + (longlong)pfVar6)) * 4.0 * fVar24);
            *(float *)((longlong)pfVar6 + lVar9 + 4) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(pfVar6[2] -
                             (*(float *)(lVar8 + 8 + (longlong)pfVar6) * 6.0 +
                              *(float *)(lVar16 + 8 + (longlong)pfVar6) +
                             *(float *)(lVar10 + 8 + (longlong)pfVar6)) * fVar24);
            pfVar6[2] = (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                                SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)((longlong)pfVar6 + lVar9 + 8) -
                             (*(float *)(lVar8 + 8 + (longlong)pfVar6) +
                             *(float *)(lVar10 + 8 + (longlong)pfVar6)) * 4.0 * fVar24);
            *(float *)((longlong)pfVar6 + lVar9 + 8) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            pfVar6 = pfVar6 + 4;
            uVar11 = uVar11 - 1;
            lVar14 = local_e0;
            param_2 = local_b0;
            iVar15 = local_e4;
            local_b8 = lVar5;
          } while (uVar11 != 0);
        }
        iVar18 = local_e8;
        if (iVar19 < iVar2) {
          pfVar6 = (float *)(lVar13 + lVar5 * 4);
          lVar12 = lVar12 - lVar13;
          uVar11 = (ulonglong)(uint)(iVar2 - iVar19);
          do {
            dVar20 = (double)(*pfVar6 -
                             (*(float *)((local_78[1] - lVar13) + (longlong)pfVar6) * 6.0 +
                              *(float *)((local_78[0] - lVar13) + (longlong)pfVar6) +
                             *(float *)((local_78[2] - lVar13) + (longlong)pfVar6)) * fVar24);
            *pfVar6 = (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                              SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)(lVar12 + (longlong)pfVar6) -
                             (*(float *)((local_78[1] - lVar13) + (longlong)pfVar6) +
                             *(float *)((local_78[2] - lVar13) + (longlong)pfVar6)) * 4.0 * fVar24);
            *(float *)(lVar12 + (longlong)pfVar6) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            pfVar6 = pfVar6 + 1;
            uVar11 = uVar11 - 1;
          } while (uVar11 != 0);
        }
      }
      else if (iVar15 < iVar19) {
        lVar13 = *(longlong *)(param_3 + 0x10) + **(longlong **)(param_3 + 0x48) * lVar7 * 2;
        iVar19 = 0;
        lVar5 = 0;
        if (3 < iVar2) {
          pfVar6 = (float *)(local_78[0] + 4);
          lVar12 = local_78[1] - local_78[0];
          lVar8 = local_78[2] - local_78[0];
          lVar14 = lVar13 - local_78[0];
          uVar4 = (iVar2 - 4U >> 2) + 1;
          uVar11 = (ulonglong)uVar4;
          iVar19 = uVar4 * 4;
          lVar5 = uVar11 * 4;
          do {
            dVar20 = (double)(*(float *)((longlong)pfVar6 + lVar14 + -4) -
                             (*(float *)((longlong)pfVar6 + lVar12 + -4) * 6.0 + pfVar6[-1] +
                             *(float *)((longlong)pfVar6 + lVar8 + -4)) * fVar24);
            *(float *)((longlong)pfVar6 + lVar14 + -4) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)(lVar14 + (longlong)pfVar6) -
                             (*(float *)(lVar12 + (longlong)pfVar6) * 6.0 + *pfVar6 +
                             *(float *)(lVar8 + (longlong)pfVar6)) * fVar24);
            *(float *)(lVar14 + (longlong)pfVar6) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)(lVar14 + 4 + (longlong)pfVar6) -
                             (*(float *)(lVar12 + 4 + (longlong)pfVar6) * 6.0 + pfVar6[1] +
                             *(float *)(lVar8 + 4 + (longlong)pfVar6)) * fVar24);
            *(float *)(lVar14 + 4 + (longlong)pfVar6) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            dVar20 = (double)(*(float *)((longlong)pfVar6 + lVar14 + 8) -
                             (*(float *)(lVar12 + 8 + (longlong)pfVar6) * 6.0 + pfVar6[2] +
                             *(float *)(lVar8 + 8 + (longlong)pfVar6)) * fVar24);
            *(float *)((longlong)pfVar6 + lVar14 + 8) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            pfVar6 = pfVar6 + 4;
            uVar11 = uVar11 - 1;
          } while (uVar11 != 0);
        }
        lVar14 = local_e0;
        iVar18 = local_e8;
        if (iVar19 < iVar2) {
          pfVar6 = (float *)(local_78[0] + lVar5 * 4);
          lVar13 = lVar13 - local_78[0];
          uVar11 = (ulonglong)(uint)(iVar2 - iVar19);
          do {
            dVar20 = (double)(*(float *)(lVar13 + (longlong)pfVar6) -
                             (*(float *)((local_78[1] - local_78[0]) + (longlong)pfVar6) * 6.0 +
                              *pfVar6 + *(float *)((local_78[2] - local_78[0]) + (longlong)pfVar6))
                             * fVar24);
            *(float *)(lVar13 + (longlong)pfVar6) =
                 (float)(double)CONCAT44((uint)((ulonglong)dVar20 >> 0x20) & uVar23,
                                         SUB84(dVar20,0) & uVar21);
            pfVar6 = pfVar6 + 1;
            uVar11 = uVar11 - 1;
          } while (uVar11 != 0);
        }
      }
      local_e4 = iVar15 + 2;
      lVar7 = lVar7 + 1;
      local_e0 = lVar14 + 2;
      param_3 = local_c0;
      plVar17 = local_a0;
      iVar19 = local_d4;
    } while (local_d8 < local_d0);
  }
  plVar17[1] = plVar17[1] - local_c8;
  return;
}


```

## {'procedure': '0x180014890'}
Evidence: `ev_bb2e3c3e524897a281f424abbe687b7acd4355942d41c191c3f385a6fefc3501`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180014890(longlong param_1,longlong *param_2,int *param_3,longlong param_4,longlong param_5
                  ,longlong param_6,longlong param_7)

{
  float fVar1;
  float fVar2;
  float fVar3;
  uint uVar4;
  float *pfVar5;
  float *pfVar6;
  ulonglong uVar7;
  longlong *plVar8;
  int iVar9;
  float *pfVar10;
  longlong lVar11;
  longlong lVar12;
  longlong *plVar13;
  int iVar14;
  ulonglong uVar15;
  short *psVar16;
  uint uVar17;
  int iVar19;
  longlong lVar20;
  int *piVar21;
  longlong lVar22;
  longlong lVar23;
  float fVar24;
  float fVar25;
  float fVar26;
  undefined1 auStackY_2b8 [32];
  longlong in_stack_fffffffffffffdb0;
  int local_22c;
  float local_228;
  longlong local_218;
  undefined4 uStack_210;
  undefined4 uStack_20c;
  longlong local_208;
  longlong *local_200;
  int local_1f8 [2];
  int *local_1f0;
  ulonglong local_1e8;
  longlong local_1d8;
  undefined8 uStack_1d0;
  int local_1c8;
  longlong local_1c0;
  longlong local_1b8;
  short *local_1b0;
  longlong local_1a8;
  longlong local_1a0;
  longlong local_198;
  longlong local_188;
  undefined8 uStack_180;
  longlong local_178;
  ulonglong local_170;
  longlong local_168;
  undefined8 uStack_160;
  longlong local_158;
  undefined4 uStack_150;
  undefined4 uStack_14c;
  longlong local_148;
  longlong local_138;
  undefined4 uStack_130;
  undefined4 uStack_12c;
  longlong local_128;
  undefined4 uStack_120;
  undefined4 uStack_11c;
  longlong local_118;
  undefined4 uStack_110;
  undefined4 uStack_10c;
  longlong local_108;
  undefined4 uStack_100;
  undefined4 uStack_fc;
  longlong local_f8;
  undefined4 uStack_f0;
  undefined4 uStack_ec;
  longlong local_e8;
  undefined4 uStack_e0;
  undefined4 uStack_dc;
  longlong local_d8;
  undefined4 uStack_d0;
  undefined4 uStack_cc;
  longlong local_c8;
  undefined4 uStack_c0;
  undefined4 uStack_bc;
  longlong local_b8;
  undefined4 uStack_b0;
  undefined4 uStack_ac;
  undefined8 local_a8;
  longlong *local_a0;
  undefined8 local_98;
  longlong local_90;
  longlong local_88;
  undefined8 local_80;
  ulonglong local_78;
  ulonglong uVar18;
  
  local_a8 = 0xfffffffffffffffe;
  local_78 = DAT_18008b080 ^ (ulonglong)auStackY_2b8;
  local_148 = param_5;
  fVar3 = *(float *)(param_1 + 0x4dc);
  local_228 = *(float *)(param_1 + 0x4e0);
  local_22c = *param_3;
  iVar14 = param_3[1];
  uVar4 = *(uint *)(param_1 + 0x78);
  local_1c8 = (int)uVar4 / 2;
  iVar19 = iVar14 + -1;
  iVar9 = *(int *)(param_1 + 0x7c);
  pfVar5 = *(float **)(param_3 + 4);
  fVar1 = (float)param_3[0xb];
  local_80 = 0;
  local_98 = 0x20;
  uVar7 = 0;
  local_170 = 0;
  local_90 = 0;
  local_88 = param_2[1] + *param_2;
  local_208 = param_4;
  local_200 = param_2;
  local_1f0 = param_3;
  local_178 = param_1;
  local_a0 = param_2;
  FUN_180024560((longlong *)&local_a0,param_1,&local_1c0,local_22c);
  if (0 < iVar19) {
    uVar15 = uVar7;
    uVar18 = uVar7;
    pfVar6 = pfVar5;
    do {
      pfVar10 = pfVar6 + 1;
      *(float *)((local_1c0 - (longlong)pfVar5) + -4 + (longlong)pfVar10) = *pfVar10 - *pfVar6;
      for (; (longlong)uVar7 < (longlong)(iVar19 * iVar9 + 1); uVar7 = uVar7 + 1) {
        fVar24 = (float)(int)uVar15 * fVar1 + *pfVar5;
        if (*pfVar10 <= fVar24 && fVar24 != *pfVar10) break;
        *(short *)(local_1a0 + uVar7 * 2) = (short)uVar18;
        *(float *)(local_198 + uVar7 * 4) = fVar24 - *pfVar6;
        uVar15 = (ulonglong)((int)uVar15 + 1);
      }
      uVar17 = (int)uVar18 + 1;
      uVar18 = (ulonglong)uVar17;
      pfVar6 = pfVar10;
    } while ((int)uVar17 < iVar19);
  }
  lVar20 = (longlong)local_22c;
  lVar22 = local_198;
  lVar23 = local_1a0;
  if (0 < lVar20) {
    iVar14 = iVar14 - uVar4;
    local_1f8[0] = iVar14;
    local_22c = 0;
    lVar11 = local_1b8 - (longlong)local_1b0;
    lVar12 = local_1a8 - (longlong)local_1b0;
    local_1e8 = -(longlong)local_1b0;
    psVar16 = local_1b0;
    local_218 = lVar11;
    local_158 = lVar12;
    do {
      iVar9 = (uint)*(ushort *)(*(longlong *)(param_3 + 8) + local_1e8 + (longlong)psVar16) -
              local_1c8;
      local_218 = CONCAT44(local_218._4_4_,iVar9);
      plVar13 = (longlong *)&local_22c;
      if (-1 < iVar9) {
        plVar13 = &local_218;
      }
      plVar8 = (longlong *)local_1f8;
      if ((int)*plVar13 <= iVar14) {
        plVar8 = plVar13;
      }
      iVar9 = (int)*plVar8;
      local_218 = CONCAT44(local_218._4_4_,iVar9);
      fVar24 = *pfVar5;
      fVar25 = ceilf((pfVar5[iVar9] - fVar24) / fVar1);
      fVar2 = pfVar5[(longlong)(int)uVar4 + (longlong)iVar9 + -1];
      fVar26 = floorf((fVar2 - fVar24) / fVar1);
      for (iVar9 = (int)fVar26; fVar2 < (float)iVar9 * fVar1 + fVar24; iVar9 = iVar9 + -1) {
      }
      *(short *)(lVar11 + (longlong)psVar16) = (short)local_218;
      *psVar16 = (short)(int)fVar25;
      *(short *)(lVar12 + (longlong)psVar16) = ((short)iVar9 - (short)(int)fVar25) + 1;
      psVar16 = psVar16 + 1;
      lVar20 = lVar20 + -1;
      param_3 = local_1f0;
      lVar22 = local_198;
      lVar23 = local_1a0;
    } while (lVar20 != 0);
  }
  piVar21 = (int *)(local_178 + 0x244);
  local_1e8 = (ulonglong)
              ((longlong)(piVar21 + *(int *)(local_178 + 0x254)) + (3 - (longlong)piVar21)) >> 2;
  if (piVar21 + *(int *)(local_178 + 0x254) < piVar21) {
    local_1e8 = 0;
  }
  if (local_1e8 != 0) {
    do {
      lVar11 = (longlong)*piVar21;
      lVar20 = *(longlong *)(local_148 + lVar11 * 8);
      uVar17 = (int)*(undefined8 *)(lVar20 + 0x50) * (int)local_228 + 0x1fU & 0xffffffe0;
      if (*(char *)(local_178 + 0x24) == '\0') {
        if ((char)local_1f0[10] == '\0') {
          lVar11 = lVar11 * 0x68;
          local_1d8 = *(longlong *)(lVar11 + 0x10 + param_7);
          uStack_110 = (undefined4)*(undefined8 *)(lVar11 + 0x50 + param_7);
          uStack_1d0 = CONCAT44(uStack_10c,uStack_110);
          lVar12 = param_6 + lVar11;
          local_188 = *(longlong *)(lVar12 + 0x10);
          uStack_f0 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
          uStack_180 = CONCAT44(uStack_ec,uStack_f0);
          lVar11 = local_208 + lVar11;
          local_168 = *(longlong *)(lVar11 + 0x10);
          uStack_c0 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
          uStack_160 = CONCAT44(uStack_bc,uStack_c0);
          uVar7 = (ulonglong)(uint)local_228;
          local_118 = local_1d8;
          local_f8 = local_188;
          local_c8 = local_168;
          FUN_1800251b0((longlong)local_200,(int)fVar3,local_228,(ulonglong)uVar4,
                        (longlong)&local_168,*(longlong *)(lVar20 + 0x10),uVar17,local_1b8,
                        (longlong)local_1b0,local_1a8,local_1c0,lVar23,lVar22,
                        in_stack_fffffffffffffdb0,&local_188,&local_1d8);
        }
        else {
          lVar11 = lVar11 * 0x68;
          local_1d8 = *(longlong *)(lVar11 + 0x10 + param_7);
          uStack_b0 = (undefined4)*(undefined8 *)(lVar11 + 0x50 + param_7);
          uStack_1d0 = CONCAT44(uStack_ac,uStack_b0);
          lVar12 = param_6 + lVar11;
          local_188 = *(longlong *)(lVar12 + 0x10);
          uStack_100 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
          uStack_180 = CONCAT44(uStack_fc,uStack_100);
          lVar11 = local_208 + lVar11;
          local_168 = *(longlong *)(lVar11 + 0x10);
          uStack_120 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
          uStack_160 = CONCAT44(uStack_11c,uStack_120);
          uVar7 = (ulonglong)(uint)local_228;
          local_128 = local_168;
          local_108 = local_188;
          local_b8 = local_1d8;
          FUN_180024c90((longlong)local_200,fVar3,(int)local_228,(ulonglong)uVar4,
                        (longlong)&local_168,*(longlong *)(lVar20 + 0x10),uVar17,local_1b8,
                        (longlong)local_1b0,local_1a8,local_1c0,lVar23,lVar22,
                        in_stack_fffffffffffffdb0,&local_188,&local_1d8);
        }
      }
      else {
        lVar11 = lVar11 * 0x68;
        local_168 = *(longlong *)(lVar11 + 0x10 + param_7);
        uStack_150 = (undefined4)*(undefined8 *)(lVar11 + 0x50 + param_7);
        uStack_160 = CONCAT44(uStack_14c,uStack_150);
        lVar12 = param_6 + lVar11;
        local_218 = *(longlong *)(lVar12 + 0x10);
        uStack_210 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
        uStack_180 = CONCAT44(uStack_20c,uStack_210);
        lVar11 = local_208 + lVar11;
        local_1d8 = *(longlong *)(lVar11 + 0x10);
        uStack_d0 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
        uStack_1d0 = CONCAT44(uStack_cc,uStack_d0);
        in_stack_fffffffffffffdb0 = *(longlong *)(local_1f0 + 6);
        uVar7 = (ulonglong)(uint)local_228;
        local_188 = local_218;
        local_158 = local_168;
        local_d8 = local_1d8;
        FUN_180024770((longlong)local_200,fVar3,(int)local_228,(ulonglong)uVar4,(longlong)&local_1d8
                      ,*(longlong *)(lVar20 + 0x10),uVar17,local_1b8,(longlong)local_1b0,local_1a8,
                      local_1c0,lVar23,lVar22,in_stack_fffffffffffffdb0,&local_188,&local_168);
      }
      local_1d8 = *(longlong *)(lVar12 + 0x10);
      uStack_e0 = (undefined4)*(undefined8 *)(lVar12 + 0x50);
      uStack_1d0 = CONCAT44(uStack_dc,uStack_e0);
      local_188 = *(longlong *)(lVar11 + 0x10);
      uStack_130 = (undefined4)*(undefined8 *)(lVar11 + 0x50);
      uStack_180 = CONCAT44(uStack_12c,uStack_130);
      local_138 = local_188;
      local_e8 = local_1d8;
      FUN_180014730((int)fVar3,(int)local_228,uVar7,&local_1d8,*pfVar5,fVar1);
      piVar21 = piVar21 + 1;
      local_170 = local_170 + 1;
    } while (local_170 != local_1e8);
  }
  local_a0[1] = local_a0[1] - local_90;
  return;
}


```

## {'procedure': '0x1800157e0'}
Evidence: `ev_bef161eff7ef9bc56337ebf856bf218be55969bf34da0445febc03f6094ecb75`

```c

void FUN_1800157e0(float *param_1,longlong *param_2,int param_3)

{
  float *pfVar1;
  float *pfVar2;
  longlong lVar3;
  float *pfVar4;
  longlong lVar5;
  float *pfVar6;
  float *pfVar7;
  longlong lVar8;
  longlong lVar9;
  
  lVar3 = *param_2;
  lVar9 = (longlong)param_3;
  lVar8 = 0;
  if (3 < lVar9) {
    lVar5 = (lVar9 - 4U >> 2) + 1;
    lVar8 = lVar5 * 4;
    pfVar4 = (float *)(lVar3 + 8);
    pfVar6 = param_1;
    do {
      pfVar1 = pfVar6 + 2;
      pfVar2 = pfVar6 + 3;
      pfVar7 = pfVar6 + 4;
      pfVar4[-2] = *pfVar6 * 6.0 + pfVar6[-1] + pfVar6[1];
      pfVar4[-1] = (pfVar6[1] + *pfVar6) * 4.0;
      *pfVar4 = pfVar6[1] * 6.0 + *pfVar6 + *pfVar1;
      pfVar4[1] = (*pfVar1 + pfVar6[1]) * 4.0;
      pfVar4[2] = *pfVar1 * 6.0 + pfVar6[1] + *pfVar2;
      pfVar4[3] = (*pfVar2 + *pfVar1) * 4.0;
      pfVar4[4] = *pfVar2 * 6.0 + *pfVar1 + *pfVar7;
      pfVar4[5] = (*pfVar7 + *pfVar2) * 4.0;
      lVar5 = lVar5 + -1;
      pfVar4 = pfVar4 + 8;
      pfVar6 = pfVar7;
    } while (lVar5 != 0);
  }
  if (lVar8 < lVar9) {
    pfVar4 = param_1 + lVar8;
    do {
      lVar8 = lVar8 + 1;
      pfVar6 = pfVar4 + 1;
      *(float *)(lVar3 + -8 + lVar8 * 8) = *pfVar4 * 6.0 + pfVar4[-1] + *pfVar6;
      *(float *)(lVar3 + -4 + lVar8 * 8) = (*pfVar6 + *pfVar4) * 4.0;
      pfVar4 = pfVar6;
    } while (lVar8 < lVar9);
  }
  *param_2 = param_2[1];
  param_2[1] = param_2[2];
  param_2[2] = lVar3;
  return;
}


```

## {'procedure': '0x18000e280'}
Evidence: `ev_c7d850fe24a07a0f21002d77d53100ab34aaf09a9c404fc052321e8cc431efab`

```c

void FUN_18000e280(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5,
                  longlong *param_6,longlong *param_7)

{
  undefined8 uVar1;
  undefined8 uVar2;
  int iVar3;
  float fVar4;
  float fVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  longlong lVar12;
  longlong lVar13;
  longlong lVar14;
  longlong lVar15;
  longlong lVar16;
  longlong lVar17;
  float fVar18;
  float fVar19;
  float local_b0;
  float fStack_ac;
  float local_a0;
  float fStack_9c;
  
  if (0 < param_2) {
    lVar16 = 0;
    do {
      lVar14 = 0;
      if (0 < (longlong)param_1) {
        lVar6 = *param_6;
        lVar7 = param_6[1];
        lVar8 = param_6[2];
        lVar9 = param_6[3];
        lVar17 = 0;
        do {
          fVar18 = *(float *)(lVar6 + lVar14 * 4);
          fVar19 = *(float *)(lVar7 + lVar16 * 4);
          lVar14 = lVar14 + 1;
          lVar10 = (int)param_3[1] * lVar16 + *param_3;
          iVar3 = *(int *)(lVar9 + lVar16 * 4);
          uVar1 = *(undefined8 *)(lVar10 + lVar17);
          fVar4 = *(float *)(lVar10 + 8 + lVar17);
          local_a0 = (float)uVar1;
          lVar10 = (int)param_4[1] * lVar16 + *param_4;
          uVar2 = *(undefined8 *)(lVar10 + lVar17);
          fVar5 = *(float *)(lVar10 + 8 + lVar17);
          local_b0 = (float)uVar2;
          lVar15 = (longlong)((int)param_5[1] * iVar3) + *param_5;
          lVar11 = (longlong)*(int *)(lVar8 + -4 + lVar14 * 4);
          lVar12 = (longlong)((iVar3 + 1) * (int)param_5[1]) + *param_5;
          lVar10 = *param_7;
          lVar13 = (int)param_7[1] * lVar16 + lVar17;
          lVar17 = lVar17 + 0xc;
          fStack_ac = (float)((ulonglong)uVar2 >> 0x20);
          fVar19 = (1.0 - fVar19) * fVar18 * *(float *)(lVar15 + 4 + lVar11 * 4) +
                   (1.0 - fVar18) * (1.0 - fVar19) * *(float *)(lVar15 + lVar11 * 4) +
                   (1.0 - fVar18) * fVar19 * *(float *)(lVar12 + lVar11 * 4) +
                   fVar19 * fVar18 * *(float *)(lVar12 + 4 + lVar11 * 4);
          fStack_9c = (float)((ulonglong)uVar1 >> 0x20);
          fVar18 = 1.0 - fVar19;
          *(ulonglong *)(lVar13 + lVar10) =
               CONCAT44(fStack_9c * fVar19 + fStack_ac * fVar18,
                        local_a0 * fVar19 + local_b0 * fVar18);
          *(float *)(lVar13 + 8 + lVar10) = fVar4 * fVar19 + fVar5 * fVar18;
        } while (lVar14 < param_1);
      }
      lVar16 = lVar16 + 1;
    } while (lVar16 < param_2);
  }
  return;
}


```

## {'procedure': '0x18001e520'}
Evidence: `ev_cb8cb6a5a4ea85eecd77170fc655544ce46dcd723d2accc863961cfe9c166dad`

```c

/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

undefined8 FUN_18001e520(longlong param_1,undefined8 param_2,float param_3,longlong param_4)

{
  float fVar1;
  float fVar2;
  undefined8 uVar3;
  
  if (DAT_18008c518 != '\0') {
    if (DAT_18008c538 == (void *)0x0) {
      DAT_18008c538 = malloc(0x280000);
    }
    DAT_18008c530 = 0;
    _DAT_18008c520 = cv::getTickCount();
    _DAT_18008c528 = _DAT_18008c520;
    FUN_180001d80(&DAT_18007b328,0,0,0,0);
  }
  if (*(int *)(param_1 + 0x840) < (int)*(uint *)(param_1 + 0x24)) {
    if ((*(char *)(param_1 + 0x2c) == '\0') && (*(int *)(param_1 + 0x840) != 0)) {
      fVar1 = **(float **)(param_1 + 0x848);
      fVar2 = (*(float **)(param_1 + 0x848))[(longlong)*(int *)(param_1 + 0x840) + -1];
      if ((fVar1 != fVar2) && ((fVar2 != param_3 && (fVar2 < fVar1 != param_3 < fVar2)))) {
        return 0x17;
      }
    }
    *(undefined8 *)(param_1 + 0x810) = *(undefined8 *)(param_1 + 0xd40);
    FUN_180001d80("InputImage frameNo.%3d/%3d\n",*(int *)(param_1 + 0x840) + 1,
                  (ulonglong)*(uint *)(param_1 + 0x24),0,0);
    FUN_180022960(param_1,param_2,param_3);
    if (param_4 != 0) {
      FUN_180023480(param_1,param_4);
    }
    uVar3 = 0;
  }
  else {
    uVar3 = 0x14;
  }
  return uVar3;
}


```

## {'procedure': '0x180013fd0'}
Evidence: `ev_cf52a7fd57ecde67a70af79600563ab4fa5e70a8d741ab7f42b9c87f546e952f`

```c

void FUN_180013fd0(int param_1,longlong *param_2,longlong param_3,undefined4 *param_4,
                  undefined4 *param_5,longlong param_6)

{
  longlong lVar1;
  float *pfVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  float *pfVar7;
  longlong lVar8;
  float fVar9;
  float fVar10;
  
  lVar3 = 0;
  lVar5 = 1;
  lVar4 = (longlong)(param_1 + -1);
  param_5[lVar4] = 0;
  *param_5 = 0;
  lVar6 = 1;
  if (1 < lVar4) {
    if (3 < lVar4 + -1) {
      lVar8 = (longlong)param_5 - param_3;
      pfVar2 = (float *)(param_3 + 8);
      do {
        pfVar7 = pfVar2 + 4;
        lVar1 = (int)param_2[1] + lVar6;
        *(float *)(lVar8 + -0x14 + (longlong)pfVar7) =
             ((*pfVar2 - pfVar2[-1]) / *(float *)(*param_2 + lVar1 * 4) -
             (pfVar2[-1] - *(float *)(lVar8 + (longlong)pfVar2 + (param_3 - (longlong)param_5) + -8)
             ) / *(float *)(*param_2 + -4 + lVar1 * 4)) * 3.0;
        lVar1 = (int)param_2[1] + lVar6;
        *(float *)(lVar8 + (longlong)pfVar2) =
             ((pfVar2[1] - *pfVar2) / *(float *)(*param_2 + 4 + lVar1 * 4) -
             (*pfVar2 - pfVar2[-1]) / *(float *)(*param_2 + lVar1 * 4)) * 3.0;
        lVar1 = (int)param_2[1] + lVar6;
        *(float *)(lVar8 + -0xc + (longlong)pfVar7) =
             ((pfVar2[2] - pfVar2[1]) / *(float *)(*param_2 + 8 + lVar1 * 4) -
             (pfVar2[1] - *pfVar2) / *(float *)(*param_2 + 4 + lVar1 * 4)) * 3.0;
        lVar1 = (int)param_2[1] + lVar6;
        lVar6 = lVar6 + 4;
        *(float *)(lVar8 + -8 + (longlong)pfVar7) =
             ((pfVar2[3] - pfVar2[2]) / *(float *)(*param_2 + 0xc + lVar1 * 4) -
             (pfVar2[2] - pfVar2[1]) / *(float *)(*param_2 + 8 + lVar1 * 4)) * 3.0;
        pfVar2 = pfVar7;
      } while (lVar6 < lVar4 + -3);
    }
    if (lVar6 < lVar4) {
      pfVar2 = (float *)(param_3 + -4 + lVar6 * 4);
      do {
        lVar8 = (int)param_2[1] + lVar6;
        lVar6 = lVar6 + 1;
        *(float *)(((longlong)param_5 - param_3) + (longlong)(pfVar2 + 1)) =
             ((pfVar2[2] - pfVar2[1]) / *(float *)(*param_2 + lVar8 * 4) -
             (pfVar2[1] - *pfVar2) / *(float *)(*param_2 + -4 + lVar8 * 4)) * 3.0;
        pfVar2 = pfVar2 + 1;
      } while (lVar6 < lVar4);
    }
  }
  *param_4 = 0;
  if (1 < lVar4) {
    if (3 < lVar4 + -1) {
      pfVar2 = (float *)(param_5 + 2);
      lVar6 = (longlong)param_4 - (longlong)param_5;
      do {
        lVar8 = (int)param_2[1] + lVar5;
        fVar9 = *(float *)(*param_2 + -4 + lVar8 * 4);
        fVar10 = (fVar9 + *(float *)(*param_2 + lVar8 * 4)) * 2.0 -
                 fVar9 * *(float *)((longlong)pfVar2 + lVar6 + -8);
        pfVar2[-1] = (pfVar2[-1] - fVar9 * pfVar2[-2]) / fVar10;
        fVar10 = *(float *)(*param_2 + ((int)param_2[1] + lVar5) * 4) / fVar10;
        *(float *)(lVar6 + -4 + (longlong)pfVar2) = fVar10;
        lVar8 = (int)param_2[1] + lVar5;
        fVar9 = *(float *)(*param_2 + lVar8 * 4);
        fVar10 = (fVar9 + *(float *)(*param_2 + 4 + lVar8 * 4)) * 2.0 - fVar10 * fVar9;
        *pfVar2 = (*pfVar2 - fVar9 * pfVar2[-1]) / fVar10;
        fVar10 = *(float *)(*param_2 + 4 + ((int)param_2[1] + lVar5) * 4) / fVar10;
        *(float *)((longlong)pfVar2 + lVar6) = fVar10;
        lVar8 = (int)param_2[1] + lVar5;
        fVar9 = *(float *)(*param_2 + 4 + lVar8 * 4);
        fVar10 = (fVar9 + *(float *)(*param_2 + 8 + lVar8 * 4)) * 2.0 - fVar10 * fVar9;
        pfVar2[1] = (pfVar2[1] - fVar9 * *pfVar2) / fVar10;
        fVar10 = *(float *)(*param_2 + 8 + ((int)param_2[1] + lVar5) * 4) / fVar10;
        *(float *)(lVar6 + 4 + (longlong)pfVar2) = fVar10;
        lVar8 = (int)param_2[1] + lVar5;
        fVar9 = *(float *)(*param_2 + 8 + lVar8 * 4);
        fVar10 = (fVar9 + *(float *)(*param_2 + 0xc + lVar8 * 4)) * 2.0 - fVar10 * fVar9;
        pfVar2[2] = (pfVar2[2] - fVar9 * pfVar2[1]) / fVar10;
        *(float *)(lVar6 + 8 + (longlong)pfVar2) =
             *(float *)(*param_2 + 0xc + ((int)param_2[1] + lVar5) * 4) / fVar10;
        lVar5 = lVar5 + 4;
        pfVar2 = pfVar2 + 4;
      } while (lVar5 < lVar4 + -3);
    }
    if (lVar5 < lVar4) {
      pfVar2 = (float *)(param_5 + lVar5);
      do {
        lVar6 = (int)param_2[1] + lVar5;
        fVar9 = *(float *)(*param_2 + -4 + lVar6 * 4);
        fVar10 = (fVar9 + *(float *)(*param_2 + lVar6 * 4)) * 2.0 -
                 fVar9 * *(float *)((longlong)pfVar2 + ((longlong)param_4 - (longlong)param_5) + -4)
        ;
        *pfVar2 = (*pfVar2 - fVar9 * pfVar2[-1]) / fVar10;
        lVar6 = (int)param_2[1] + lVar5;
        lVar5 = lVar5 + 1;
        *(float *)((longlong)pfVar2 + ((longlong)param_4 - (longlong)param_5)) =
             *(float *)(*param_2 + lVar6 * 4) / fVar10;
        pfVar2 = pfVar2 + 1;
      } while (lVar5 < lVar4);
    }
  }
  lVar5 = (longlong)(param_1 + -2);
  if (3 < lVar5) {
    lVar6 = lVar5 + -1;
    lVar1 = (lVar5 - 4U >> 2) + 1;
    lVar8 = (longlong)param_4 - (longlong)param_5;
    lVar5 = lVar5 + lVar1 * -4;
    pfVar2 = (float *)(param_5 + lVar6);
    do {
      fVar9 = pfVar2[1] -
              *(float *)((longlong)pfVar2 + lVar8 + 4) *
              *(float *)((longlong)pfVar2 + lVar8 + (longlong)param_5 + (8 - (longlong)param_4));
      pfVar2[1] = fVar9;
      fVar9 = *pfVar2 - fVar9 * *(float *)((longlong)pfVar2 + lVar8);
      *pfVar2 = fVar9;
      fVar9 = pfVar2[-1] - fVar9 * *(float *)((longlong)pfVar2 + lVar8 + -4);
      pfVar2[-1] = fVar9;
      pfVar2[-2] = pfVar2[-2] - fVar9 * *(float *)(lVar8 + 8 + (longlong)(pfVar2 + -4));
      lVar1 = lVar1 + -1;
      pfVar2 = pfVar2 + -4;
    } while (lVar1 != 0);
  }
  if (0 < lVar5) {
    pfVar2 = (float *)(param_5 + lVar5);
    do {
      pfVar7 = (float *)(((longlong)param_4 - (longlong)param_5) + (longlong)pfVar2);
      lVar5 = lVar5 + -1;
      *pfVar2 = *pfVar2 - *(float *)((longlong)pfVar7 + (longlong)param_5 + (4 - (longlong)param_4))
                          * *pfVar7;
      pfVar2 = pfVar2 + -1;
    } while (0 < lVar5);
  }
  *(undefined4 *)(param_6 + lVar4 * 4) = 0;
  param_4[lVar4] = 0;
  if (3 < lVar4) {
    lVar8 = param_6 - (longlong)param_5;
    lVar6 = param_3 - (longlong)param_5;
    lVar5 = (longlong)param_4 - (longlong)param_5;
    pfVar2 = (float *)(param_5 + 1);
    do {
      pfVar7 = (float *)(lVar6 + (longlong)pfVar2);
      fVar10 = (*pfVar2 - pfVar2[-1]) / (*(float *)(*param_2 + ((int)param_2[1] + lVar3) * 4) * 3.0)
      ;
      *(float *)(lVar8 + -4 + (longlong)pfVar2) = fVar10;
      fVar9 = *(float *)(*param_2 + ((int)param_2[1] + lVar3) * 4);
      *(float *)((longlong)pfVar2 + lVar5 + -4) =
           (*pfVar7 - *(float *)(lVar6 + -4 + (longlong)pfVar2)) / fVar9 -
           (fVar9 * fVar10 + pfVar2[-1]) * fVar9;
      fVar10 = (pfVar2[1] - *pfVar2) /
               (*(float *)(*param_2 + 4 + ((int)param_2[1] + lVar3) * 4) * 3.0);
      *(float *)((longlong)pfVar7 + (param_6 - param_3)) = fVar10;
      fVar9 = *(float *)(*param_2 + 4 + ((int)param_2[1] + lVar3) * 4);
      *(float *)((longlong)pfVar2 + lVar5) =
           (*(float *)(lVar6 + 4 + (longlong)pfVar2) - *pfVar7) / fVar9 -
           (fVar10 * fVar9 + *pfVar2) * fVar9;
      fVar10 = (pfVar2[2] - pfVar2[1]) /
               (*(float *)(*param_2 + 8 + ((int)param_2[1] + lVar3) * 4) * 3.0);
      *(float *)(lVar8 + 4 + (longlong)pfVar2) = fVar10;
      fVar9 = *(float *)(*param_2 + 8 + ((int)param_2[1] + lVar3) * 4);
      *(float *)((longlong)pfVar2 + lVar5 + 4) =
           (*(float *)(lVar6 + 8 + (longlong)pfVar2) - *(float *)(lVar6 + 4 + (longlong)pfVar2)) /
           fVar9 - (fVar9 * fVar10 + pfVar2[1]) * fVar9;
      fVar10 = (pfVar2[3] - pfVar2[2]) /
               (*(float *)(*param_2 + 0xc + ((int)param_2[1] + lVar3) * 4) * 3.0);
      *(float *)(lVar8 + 8 + (longlong)pfVar2) = fVar10;
      fVar9 = *(float *)(*param_2 + 0xc + ((int)param_2[1] + lVar3) * 4);
      lVar3 = lVar3 + 4;
      *(float *)((longlong)pfVar2 + lVar5 + 8) =
           (*(float *)(lVar6 + 0xc + (longlong)pfVar2) - *(float *)(lVar6 + 8 + (longlong)pfVar2)) /
           fVar9 - (fVar9 * fVar10 + pfVar2[2]) * fVar9;
      pfVar2 = pfVar2 + 4;
    } while (lVar3 < lVar4 + -3);
  }
  if (lVar3 < lVar4) {
    lVar5 = (longlong)param_5 - param_3;
    pfVar2 = (float *)(param_3 + 4 + lVar3 * 4);
    do {
      pfVar7 = pfVar2 + 1;
      fVar10 = (*(float *)(lVar5 + (longlong)pfVar2) - *(float *)(lVar5 + -4 + (longlong)pfVar2)) /
               (*(float *)(*param_2 + ((int)param_2[1] + lVar3) * 4) * 3.0);
      *(float *)((param_6 - param_3) + -8 + (longlong)pfVar7) = fVar10;
      lVar6 = (int)param_2[1] + lVar3;
      lVar3 = lVar3 + 1;
      fVar9 = *(float *)(*param_2 + lVar6 * 4);
      *(float *)((longlong)param_4 + (-8 - param_3) + (longlong)pfVar7) =
           (*pfVar2 - pfVar2[-1]) / fVar9 -
           (fVar10 * fVar9 + *(float *)(lVar5 + -8 + (longlong)pfVar7)) * fVar9;
      pfVar2 = pfVar7;
    } while (lVar3 < lVar4);
  }
  return;
}


```

## {'procedure': '0x180022960'}
Evidence: `ev_d27841304e8f5d1fd2151c73ee8bc2442553886109e27002bf2b4338dec60216`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180022960(longlong param_1,undefined8 param_2,float param_3)

{
  Mat *pMVar1;
  int iVar2;
  longlong lVar3;
  _InputOutputArray *p_Var4;
  longlong *plVar5;
  Mat *pMVar6;
  ulonglong uVar7;
  int iVar8;
  ulonglong uVar9;
  uint uVar10;
  undefined8 uVar11;
  int *piVar12;
  longlong lVar13;
  ulonglong uVar14;
  longlong lVar15;
  int *piVar16;
  ulonglong uVar17;
  longlong lVar18;
  undefined1 auStackY_3e8 [32];
  int local_3a8;
  longlong local_3a0;
  longlong local_398;
  undefined4 local_390;
  undefined8 local_388;
  uint local_380;
  undefined4 local_37c;
  uint local_378;
  longlong local_370;
  longlong local_368;
  longlong local_360;
  longlong local_358;
  longlong local_350;
  longlong local_348;
  longlong local_340;
  longlong local_338;
  longlong local_330;
  Mat *local_328 [2];
  longlong local_318;
  longlong local_310;
  undefined4 local_308 [2];
  undefined8 *local_300;
  undefined4 local_2f8;
  undefined4 local_2f4;
  undefined4 local_2f0 [2];
  undefined8 *local_2e8;
  undefined4 local_2e0;
  undefined4 local_2dc;
  undefined4 local_2d8 [2];
  undefined8 *local_2d0;
  undefined4 local_2c8;
  undefined4 local_2c4;
  undefined4 local_2c0 [2];
  undefined8 *local_2b8;
  undefined4 local_2b0;
  undefined4 local_2ac;
  undefined4 local_2a8 [2];
  undefined4 *local_2a0;
  undefined8 local_298;
  undefined4 local_290 [2];
  longlong local_288;
  undefined8 local_280;
  undefined4 local_278 [2];
  undefined8 *local_270;
  undefined4 local_268;
  undefined4 local_264;
  Mat local_258 [4];
  int local_254;
  undefined8 local_248;
  undefined8 local_240;
  undefined8 local_238;
  undefined8 local_230;
  longlong local_220;
  longlong local_218;
  undefined1 *local_210;
  undefined1 local_208 [32];
  undefined8 local_1e8;
  undefined8 local_1e0;
  undefined8 uStack_1d8;
  undefined8 local_1d0;
  undefined8 uStack_1c8;
  undefined8 local_1c0;
  undefined8 uStack_1b8;
  undefined8 local_1b0;
  undefined8 uStack_1a8;
  undefined8 local_1a0;
  undefined8 uStack_198;
  undefined8 local_190;
  undefined8 uStack_188;
  undefined8 local_180;
  undefined8 uStack_178;
  undefined8 local_170;
  undefined8 uStack_168;
  undefined8 local_160;
  undefined8 uStack_158;
  undefined8 local_150;
  undefined8 uStack_148;
  int local_140 [2];
  undefined4 local_138;
  int local_134;
  undefined4 local_130;
  undefined4 local_12c;
  undefined8 local_128;
  undefined8 local_120;
  undefined8 local_118;
  undefined8 local_110;
  undefined8 local_108;
  longlong local_100;
  undefined4 *local_f8;
  undefined8 *local_f0;
  undefined8 local_e8;
  undefined8 local_e0;
  longlong *local_d8;
  undefined8 local_d0;
  longlong local_c8;
  longlong local_c0;
  undefined8 local_b8;
  Mat local_a8 [4];
  int local_a4;
  undefined8 local_98;
  undefined8 local_90;
  undefined8 local_88;
  undefined8 local_80;
  longlong local_70;
  longlong local_68;
  undefined1 *local_60;
  undefined1 local_58 [16];
  ulonglong local_48;
  
  local_1e8 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_3e8;
  local_3a8 = *(int *)(param_1 + 0x840);
  *(int *)(param_1 + 0x840) = local_3a8 + 1;
  local_380 = *(uint *)(param_1 + 8);
  local_378 = *(uint *)(param_1 + 0xc);
  local_390 = *(undefined4 *)(param_1 + 0x4e4);
  local_37c = *(undefined4 *)(param_1 + 0x4e8);
  local_350 = *(longlong *)(param_1 + 0xae8);
  local_360 = *(longlong *)(param_1 + 0xaf0);
  local_358 = *(longlong *)(param_1 + 0xaf8);
  lVar13 = *(longlong *)(param_1 + 0xb00);
  lVar18 = *(longlong *)(param_1 + 0xb08);
  local_318 = *(longlong *)(param_1 + 0xb10);
  local_310 = *(longlong *)(param_1 + 0xb18);
  local_b8 = 0;
  local_d8 = (longlong *)(param_1 + 0x808);
  local_d0 = 0x20;
  local_c8 = 0;
  local_c0 = *(longlong *)(param_1 + 0x810) + *local_d8;
  local_3a0 = lVar13;
  local_398 = lVar18;
  local_388 = param_2;
  local_368 = param_1;
  local_140[0] = local_3a8;
  FUN_18002b030((longlong *)&local_d8,param_1 + 8,(longlong *)local_328,&local_338,&local_340,
                &local_370,&local_348,&local_330);
  lVar15 = local_350;
  lVar3 = local_360;
  if (local_3a8 == 0) {
    piVar12 = (int *)(param_1 + 0x24c);
    piVar16 = piVar12 + *(int *)(param_1 + 0x25c);
    lVar13 = local_3a0;
    for (; local_3a0 = lVar13, piVar12 != piVar16; piVar12 = piVar12 + 1) {
      local_1e0 = 0;
      uStack_1d8 = 0;
      local_1d0 = 0;
      uStack_1c8 = 0;
      local_2c0[0] = 0xc1020006;
      local_2b8 = &local_1e0;
      local_2b0 = 1;
      local_2ac = 4;
      iVar8 = *piVar12;
      p_Var4 = cv::noArray();
      cv::Mat::setTo((Mat *)((longlong)iVar8 * 0x68 + lVar15),(_InputArray *)local_2c0,
                     (_InputArray *)p_Var4);
      local_1a0 = 0;
      uStack_198 = 0;
      local_190 = 0;
      uStack_188 = 0;
      local_278[0] = 0xc1020006;
      local_270 = &local_1a0;
      local_268 = 1;
      local_264 = 4;
      p_Var4 = cv::noArray();
      cv::Mat::setTo((Mat *)((longlong)iVar8 * 0x68 + lVar3),(_InputArray *)local_278,
                     (_InputArray *)p_Var4);
      lVar18 = local_398;
      lVar13 = local_3a0;
    }
    piVar12 = (int *)(param_1 + 0x238);
    iVar8 = *(int *)(param_1 + 0x248);
    for (piVar16 = piVar12; piVar16 != piVar12 + iVar8; piVar16 = piVar16 + 1) {
      iVar2 = *piVar16;
      local_180 = 0;
      uStack_178 = 0;
      local_170 = 0;
      uStack_168 = 0;
      local_2f0[0] = 0xc1020006;
      local_2e8 = &local_180;
      local_2e0 = 1;
      local_2dc = 4;
      p_Var4 = cv::noArray();
      cv::Mat::setTo((Mat *)((longlong)iVar2 * 0x68 + lVar13),(_InputArray *)local_2f0,
                     (_InputArray *)p_Var4);
      lVar18 = local_398;
    }
    local_160 = 0;
    uStack_158 = 0;
    local_150 = 0;
    uStack_148 = 0;
    local_308[0] = 0xc1020006;
    local_300 = &local_160;
    local_2f8 = 1;
    local_2f4 = 4;
    p_Var4 = cv::noArray();
    cv::Mat::setTo((Mat *)(param_1 + 0xa80),(_InputArray *)local_308,(_InputArray *)p_Var4);
    piVar16 = piVar12 + *(int *)(param_1 + 0x248);
    lVar13 = local_3a0;
    for (; local_3a0 = lVar13, piVar12 != piVar16; piVar12 = piVar12 + 1) {
      iVar8 = *piVar12;
      local_1c0 = 0;
      uStack_1b8 = 0;
      local_1b0 = 0;
      uStack_1a8 = 0;
      local_2d8[0] = 0xc1020006;
      local_2d0 = &local_1c0;
      local_2c8 = 1;
      local_2c4 = 4;
      p_Var4 = cv::noArray();
      cv::Mat::setTo((Mat *)((longlong)iVar8 * 0x68 + lVar18),(_InputArray *)local_2d8,
                     (_InputArray *)p_Var4);
      lVar13 = local_3a0;
    }
  }
  lVar18 = param_1 + 8;
  uVar9 = 0;
  FUN_180001d80("& MultiResDFFInput Init",0,0,0,0);
  iVar8 = *(int *)(param_1 + 0x28) + -1;
  local_3a0 = CONCAT44(local_3a0._4_4_,iVar8);
  plVar5 = &local_3a0;
  if (local_3a8 <= iVar8) {
    plVar5 = (longlong *)local_140;
  }
  lVar3 = *plVar5;
  memset(&local_138,0,0x60);
  pMVar6 = FUN_180012040(lVar18,local_258,(Mat *)(param_1 + 0x878),(int)lVar3);
  local_138 = *(undefined4 *)pMVar6;
  local_134 = *(int *)(pMVar6 + 4);
  pMVar1 = pMVar6 + 8;
  local_130 = *(undefined4 *)pMVar1;
  local_12c = *(undefined4 *)(pMVar6 + 0xc);
  local_128 = *(undefined8 *)(pMVar6 + 0x10);
  local_120 = *(undefined8 *)(pMVar6 + 0x18);
  local_118 = *(undefined8 *)(pMVar6 + 0x20);
  local_110 = *(undefined8 *)(pMVar6 + 0x28);
  local_108 = *(undefined8 *)(pMVar6 + 0x30);
  local_100 = *(longlong *)(pMVar6 + 0x38);
  local_f8 = &local_130;
  local_e0 = 0;
  local_e8 = 0;
  local_f0 = *(undefined8 **)(pMVar6 + 0x48);
  if (*(int *)(pMVar6 + 4) < 3) {
    local_e8 = *local_f0;
    local_e0 = *(undefined8 *)(*(longlong *)(pMVar6 + 0x48) + 8);
    local_f0 = &local_e8;
  }
  else {
    local_f8 = *(undefined4 **)(pMVar6 + 0x40);
    *(Mat **)(pMVar6 + 0x48) = pMVar6 + 0x50;
    *(Mat **)(pMVar6 + 0x40) = pMVar1;
  }
  *(undefined8 *)pMVar6 = 0x42ff0000;
  *(undefined4 *)(pMVar6 + 0xc) = 0;
  *(undefined4 *)pMVar1 = 0;
  *(undefined8 *)(pMVar6 + 0x10) = 0;
  *(undefined8 *)(pMVar6 + 0x18) = 0;
  *(undefined8 *)(pMVar6 + 0x20) = 0;
  *(undefined8 *)(pMVar6 + 0x28) = 0;
  *(undefined8 *)(pMVar6 + 0x30) = 0;
  *(undefined8 *)(pMVar6 + 0x38) = 0;
  if (local_220 != 0) {
    LOCK();
    piVar12 = (int *)(local_220 + 0x14);
    iVar8 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar8 == 1) {
      cv::Mat::deallocate(local_258);
    }
    local_3a8 = local_140[0];
  }
  local_220 = 0;
  local_248 = 0;
  local_230 = 0;
  local_238 = 0;
  local_240 = 0;
  uVar17 = uVar9;
  if (0 < local_254) {
    do {
      *(undefined4 *)(local_218 + uVar9) = 0;
      uVar10 = (int)uVar17 + 1;
      uVar9 = uVar9 + 4;
      uVar17 = (ulonglong)uVar10;
    } while ((int)uVar10 < local_254);
  }
  if (local_210 != local_208) {
    cv::fastFree(local_210);
  }
  memset(local_a8,0,0x60);
  if (*(int *)(param_1 + 0x1c) == 1) {
    uVar10 = 2;
    if (*(int *)(param_1 + 0x20) == 8) {
      uVar10 = 0;
    }
  }
  else {
    uVar10 = 0x12;
    if (*(int *)(param_1 + 0x20) == 8) {
      uVar10 = 0x10;
    }
  }
  FUN_1800014c0((uint *)local_a8,local_378,local_380,uVar10,local_388,
                (longlong)*(int *)(param_1 + 0x10));
  local_298 = 0;
  local_2a8[0] = 0x2010000;
  local_2a0 = &local_138;
  cv::Mat::copyTo(local_a8,(_OutputArray *)local_2a8);
  *(float *)(*(longlong *)(param_1 + 0x848) + (longlong)local_3a8 * 4) = param_3;
  if (((local_3a8 < 1) || (*(char *)(param_1 + 0x2c) != '\0')) ||
     (4.9303807e-32 <=
      ABS(*(float *)(*(longlong *)(param_1 + 0x848) + -4 + (longlong)local_3a8 * 4) - param_3))) {
    (*(code *)sg::cpu::SweepGuideCPU::vftable
              [(longlong)
               (int)((uint)(*(int *)(param_1 + 0x1c) != 1) +
                    (uint)(*(int *)(param_1 + 0x20) != 8) * 2) + 7])(&local_138);
    FUN_180001d80("& LoadInputImageCPU",0,0,0,0);
    FUN_180016000(lVar18,param_1 + 0x808,local_328[0],local_338);
    FUN_180001d80("& ConstructLapPyr",0,0,0,0);
    lVar3 = local_370;
    FUN_180016710(lVar18,local_328[0],local_390,local_37c,local_370);
    uVar11 = 0;
    FUN_180001d80("& ResizeLapPyr",0,0,0,0);
    lVar15 = local_348;
    FUN_18001c3b0(lVar18,uVar11,lVar3,local_348,(undefined4 *)(param_1 + 0x298));
    FUN_180001d80("& SmoothLapPyr",0,0,0,0);
    FUN_180016a80((int *)(param_1 + 0x238),lVar15,local_3a8,local_340,lVar13,local_330);
    FUN_180001d80("& SearchMaxLap",0,0,0,0);
    uVar14 = 0;
    iVar8 = *(int *)(param_1 + 0x20);
    iVar2 = *(int *)(param_1 + 0x1c);
    uVar9 = param_1 + 0x238;
    uVar17 = uVar9 + (longlong)*(int *)(param_1 + 0x248) * 4;
    uVar7 = (uVar17 - uVar9) + 3 >> 2;
    if (uVar17 < uVar9) {
      uVar7 = uVar14;
    }
    if (uVar7 != 0) {
      do {
        (*(code *)(&PTR_FUN_18007c130)[(int)((uint)(iVar2 != 1) + (uint)(iVar8 != 8) * 2)])(lVar18);
        uVar14 = uVar14 + 1;
        param_1 = local_368;
        lVar15 = local_348;
      } while (uVar14 != uVar7);
    }
    uVar11 = 0;
    FUN_180001d80("& CompFastEFI",0,0,0,0);
    FUN_18001c3b0(lVar18,uVar11,local_370,lVar15,(undefined4 *)(param_1 + 0x288));
    FUN_180001d80("& SmoothLapPyr",0,0,0,0);
    piVar12 = (int *)(param_1 + 0x24c);
    FUN_180016a80(piVar12,lVar15,local_3a8,local_350,local_360,local_330);
    FUN_180001d80("& SearchMaxLap",0,0,0,0);
    lVar13 = local_358;
    if (*(char *)(param_1 + 0x2c) == '\0') {
      iVar8 = *(int *)(param_1 + 0x844);
      *(int *)(param_1 + 0x844) = iVar8 + 1;
      FUN_180016f70(lVar18,lVar15,local_330,iVar8,*(int **)(param_1 + 0x870),local_358,local_318,
                    local_310);
    }
    else {
      uVar17 = 0;
      uVar9 = (ulonglong)((longlong)(piVar12 + *(int *)(param_1 + 0x25c)) + (3 - (longlong)piVar12))
              >> 2;
      if (piVar12 + *(int *)(param_1 + 0x25c) < piVar12) {
        uVar9 = uVar17;
      }
      if (uVar9 != 0) {
        do {
          local_288 = *(longlong *)(lVar13 + (longlong)*piVar12 * 8) + (longlong)local_3a8 * 0x68;
          local_280 = 0;
          local_290[0] = 0x2010000;
          cv::Mat::copyTo((Mat *)((longlong)*piVar12 * 0x68 + lVar15),(_OutputArray *)local_290);
          piVar12 = piVar12 + 1;
          uVar17 = uVar17 + 1;
          param_1 = local_368;
        } while (uVar17 != uVar9);
      }
      *(int *)(param_1 + 0x844) = *(int *)(param_1 + 0x844) + 1;
    }
    FUN_180001d80("& GetLapZStackNearMaxLap",0,0,0,0);
  }
  uVar9 = 0;
  if (local_70 != 0) {
    LOCK();
    piVar12 = (int *)(local_70 + 0x14);
    iVar8 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar8 == 1) {
      cv::Mat::deallocate(local_a8);
    }
  }
  local_70 = 0;
  local_98 = 0;
  local_80 = 0;
  local_88 = 0;
  local_90 = 0;
  uVar17 = uVar9;
  uVar7 = uVar9;
  if (0 < local_a4) {
    do {
      *(undefined4 *)(local_68 + uVar7) = 0;
      uVar10 = (int)uVar17 + 1;
      uVar17 = (ulonglong)uVar10;
      uVar7 = uVar7 + 4;
    } while ((int)uVar10 < local_a4);
  }
  if (local_60 != local_58) {
    cv::fastFree(local_60);
  }
  if (local_100 != 0) {
    LOCK();
    piVar12 = (int *)(local_100 + 0x14);
    iVar8 = *piVar12;
    *piVar12 = *piVar12 + -1;
    UNLOCK();
    if (iVar8 == 1) {
      cv::Mat::deallocate((Mat *)&local_138);
    }
  }
  local_100 = 0;
  local_128 = 0;
  local_110 = 0;
  local_118 = 0;
  local_120 = 0;
  uVar17 = uVar9;
  if (0 < local_134) {
    do {
      *(undefined4 *)((longlong)local_f8 + uVar17) = 0;
      uVar10 = (int)uVar9 + 1;
      uVar9 = (ulonglong)uVar10;
      uVar17 = uVar17 + 4;
    } while ((int)uVar10 < local_134);
  }
  if (local_f0 != &local_e8) {
    cv::fastFree(local_f0);
  }
  local_d8[1] = local_d8[1] - local_c8;
  return;
}


```

## {'procedure': '0x18001df60'}
Evidence: `ev_d9a44e2f73b547b9ac0b6d0623b54c27fef561de54e68b88cc1ce82c3ff72b03`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8 *
FUN_18001df60(undefined8 *param_1,undefined8 *param_2,longlong *param_3,char *param_4,
             undefined8 param_5,undefined8 param_6)

{
  longlong *plVar1;
  undefined8 uVar2;
  longlong lVar3;
  undefined8 uVar4;
  undefined8 uVar5;
  longlong lVar6;
  undefined8 *puVar7;
  undefined8 *puVar8;
  undefined1 auStack_88 [32];
  undefined8 *local_68 [2];
  undefined8 *local_58;
  undefined8 local_50;
  longlong *local_48;
  undefined8 local_40;
  undefined8 local_38;
  longlong local_30;
  undefined8 local_28;
  ulonglong local_20;
  
  local_50 = 0xfffffffffffffffe;
  local_20 = DAT_18008b080 ^ (ulonglong)auStack_88;
  *param_1 = sg::cpu::SweepGuideCPU::vftable;
  local_58 = param_1;
  FUN_1800026d0((int *)(param_1 + 1),param_2,param_3,param_4);
  uVar4 = *(undefined8 *)param_4;
  uVar5 = *(undefined8 *)(param_4 + 8);
  uVar2 = *(undefined8 *)(param_4 + 0x10);
  lVar6 = param_3[1];
  lVar3 = param_3[2];
  plVar1 = param_1 + 0x101;
  *plVar1 = *param_3;
  param_1[0x102] = lVar6;
  param_1[0x103] = lVar3;
  param_1[0x104] = uVar4;
  param_1[0x105] = uVar5;
  param_1[0x106] = uVar2;
  param_1[0x107] = param_6;
  puVar8 = param_1 + 1;
  FUN_18001bf10(param_1 + 0x108,(longlong)puVar8,plVar1);
  FUN_18001e240((longlong)(param_1 + 0x164));
  *(uint *)(param_1 + 0x51) = *(uint *)((longlong)param_1 + 0x3c);
  puVar7 = FUN_18001cf20(local_68,puVar8,plVar1,(ulonglong)*(uint *)((longlong)param_1 + 0x3c));
  uVar2 = puVar7[1];
  param_1[0x51] = *puVar7;
  param_1[0x52] = uVar2;
  *(uint *)(param_1 + 0x53) = *(uint *)(param_1 + 7);
  puVar7 = FUN_18001cf20(local_68,puVar8,plVar1,(ulonglong)*(uint *)(param_1 + 7));
  uVar2 = puVar7[1];
  param_1[0x53] = *puVar7;
  param_1[0x54] = uVar2;
  *(uint *)(param_1 + 0x55) = *(uint *)(param_1 + 0x14);
  puVar8 = FUN_18001cf20(local_68,puVar8,plVar1,(ulonglong)*(uint *)(param_1 + 0x14));
  uVar2 = puVar8[1];
  param_1[0x55] = *puVar8;
  param_1[0x56] = uVar2;
  local_68[0] = param_1;
  FUN_180029a60((int *)(param_1 + 1),(longlong *)local_68);
  param_1[0x1a9] = param_1[0x102];
  local_28 = 0;
  local_40 = 0x20;
  local_38 = 0;
  local_30 = param_1[0x102] + *plVar1;
  local_48 = plVar1;
  FUN_18002a070((longlong *)&local_48,(longlong)(param_1 + 1),(longlong)(param_1 + 0x108));
  param_1[0x1a8] = param_1[0x102];
  return param_1;
}


```

## {'procedure': '0x180015100'}
Evidence: `ev_e3ea72f1e8944cbc8bf62d4582151080b7754312d99e2d15acc670aeb8696bef`

```c

void FUN_180015100(longlong param_1,longlong *param_2,int param_3)

{
  longlong lVar1;
  float *pfVar2;
  float *pfVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  
  lVar1 = *param_2;
  lVar5 = (longlong)param_3;
  lVar4 = 0;
  if (3 < lVar5) {
    lVar6 = (lVar5 - 4U >> 2) + 1;
    lVar4 = lVar6 * 4;
    pfVar2 = (float *)(lVar1 + 8);
    pfVar3 = (float *)(param_1 + -4);
    do {
      pfVar2[-2] = (pfVar3[2] + *pfVar3) * 4.0 + pfVar3[-1] + pfVar3[3] + pfVar3[1] * 6.0;
      pfVar2[-1] = (pfVar3[4] + pfVar3[2]) * 4.0 + pfVar3[5] + pfVar3[1] + pfVar3[3] * 6.0;
      *pfVar2 = (pfVar3[6] + pfVar3[4]) * 4.0 + pfVar3[7] + pfVar3[3] + pfVar3[5] * 6.0;
      pfVar2[1] = pfVar3[9] + pfVar3[5] + (pfVar3[8] + pfVar3[6]) * 4.0 + pfVar3[7] * 6.0;
      lVar6 = lVar6 + -1;
      pfVar2 = pfVar2 + 4;
      pfVar3 = pfVar3 + 8;
    } while (lVar6 != 0);
  }
  if (lVar4 < lVar5) {
    pfVar2 = (float *)(param_1 + -4 + lVar4 * 8);
    do {
      lVar4 = lVar4 + 1;
      *(float *)(lVar1 + -4 + lVar4 * 4) =
           pfVar2[3] + pfVar2[-1] + (pfVar2[2] + *pfVar2) * 4.0 + pfVar2[1] * 6.0;
      pfVar2 = pfVar2 + 2;
    } while (lVar4 < lVar5);
  }
  *param_2 = param_2[1];
  param_2[1] = param_2[2];
  param_2[2] = param_2[3];
  lVar4 = param_2[4];
  param_2[4] = lVar1;
  param_2[3] = lVar4;
  return;
}


```

## {'procedure': '0x1800043f0'}
Evidence: `ev_e8aa9d5ceb05bf35c60b5e5826b5447c639b9c6d00990c35c300a66dfaf08620`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_1800043f0(undefined4 *param_1,longlong *param_2,int *param_3,longlong param_4,int param_5,
                  undefined8 param_6,undefined8 param_7,Mat *param_8)

{
  ulonglong uVar1;
  int iVar2;
  int iVar3;
  ulonglong uVar4;
  longlong *plVar5;
  undefined1 auStack_c8 [32];
  undefined8 local_a8;
  longlong *local_a0;
  undefined4 local_98;
  undefined8 local_90;
  undefined8 local_88;
  Mat *local_80;
  undefined8 local_78;
  longlong *local_70;
  undefined8 local_68;
  ulonglong local_60;
  longlong *local_58;
  undefined8 local_50;
  ulonglong local_48;
  
  local_78 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStack_c8;
  local_50 = 0;
  local_68 = 0x20;
  local_a0 = (longlong *)(*param_2 + param_2[1]);
  uVar4 = (longlong)*param_3 * 8 + 0x1fU & 0xffffffffffffffe0;
  uVar1 = param_2[1] + uVar4;
  param_2[1] = uVar1;
  local_70 = param_2;
  local_60 = uVar4;
  local_58 = local_a0;
  if (uVar1 <= (ulonglong)param_2[2]) {
    iVar3 = 0;
    plVar5 = local_a0;
    iVar2 = iVar3;
    if (0 < *param_3) {
      do {
        *plVar5 = (longlong)iVar3 + *(longlong *)(param_4 + 0x10);
        iVar2 = iVar2 + 1;
        iVar3 = iVar3 + param_5;
        plVar5 = plVar5 + 1;
      } while (iVar2 < *param_3);
    }
    local_80 = param_8;
    local_88 = param_7;
    local_90 = param_6;
    local_98 = (undefined4)*(undefined8 *)(param_4 + 0x50);
    local_a8 = *(undefined8 *)(param_3 + 6);
    (*(code *)(&PTR_FUN_18007bb30)[(int)((uint)(param_1[5] != 1) + (uint)(param_1[6] != 8) * 2)])
              (param_1,*(undefined1 *)((longlong)param_1 + 0x25),*param_1,param_1[1]);
    if (*(char *)(param_1 + 0x47) == '\x01') {
      FUN_180005740((longlong)param_1,param_8,2);
    }
    param_2[1] = param_2[1] - uVar4;
    return;
  }
                    /* WARNING: Subroutine does not return */
  FUN_180001b90("Allocating memory exceeding the limit ...",param_2,param_3,param_4);
}


```

## {'procedure': '0x180004050'}
Evidence: `ev_f2d110210008fba0d5e5e9013f47f0354527e985403d47c6f5da8df75215f708`

```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180004050(int *param_1,longlong *param_2,longlong param_3,longlong param_4,longlong param_5
                  ,longlong param_6)

{
  float fVar1;
  float fVar2;
  int iVar3;
  float *pfVar4;
  int *piVar5;
  int iVar6;
  longlong lVar7;
  ulonglong uVar8;
  longlong lVar9;
  ulonglong uVar10;
  ulonglong uVar11;
  float fVar12;
  undefined1 auStackY_118 [32];
  int local_a8 [2];
  longlong *local_a0;
  longlong local_98;
  undefined8 local_90;
  int local_88 [4];
  longlong *local_78;
  undefined8 uStack_70;
  ulonglong local_68;
  longlong lStack_60;
  undefined8 local_58;
  ulonglong local_48;
  
  local_90 = 0xfffffffffffffffe;
  local_48 = DAT_18008b080 ^ (ulonglong)auStackY_118;
  local_a0 = param_2;
  local_98 = param_6;
  iVar3 = *(int *)(param_3 + 4);
  lVar7 = (longlong)iVar3;
  local_58 = 0;
  local_78 = param_2;
  uStack_70 = 0x20;
  lVar9 = *param_2 + param_2[1];
  pfVar4 = *(float **)(param_3 + 0x10);
  uVar8 = lVar7 * 4 + 0x1fU & 0xffffffffffffffe0;
  uVar10 = param_2[1] + uVar8;
  param_2[1] = uVar10;
  if ((ulonglong)param_2[2] < uVar10) {
    local_68 = uVar8;
    lStack_60 = lVar9;
                    /* WARNING: Subroutine does not return */
    FUN_180001b90("Allocating memory exceeding the limit ...",lVar7,pfVar4,param_4);
  }
  fVar1 = *pfVar4;
  fVar2 = pfVar4[lVar7 + -1];
  fVar12 = (fVar2 - fVar1) / (float)iVar3;
  uVar10 = 0;
  if (0 < iVar3) {
    local_a8[0] = iVar3 + -1;
    uVar11 = uVar10;
    do {
      iVar6 = 1;
      local_88[0] = 1;
      if (1 < iVar3) {
        do {
          if ((float)(int)uVar11 * fVar12 + fVar1 < pfVar4[iVar6]) break;
          iVar6 = iVar6 + 1;
        } while (iVar6 < iVar3);
        local_88[0] = iVar6;
      }
      piVar5 = local_a8;
      if (local_88[0] <= iVar3 + -1) {
        piVar5 = local_88;
      }
      *(int *)(lVar9 + uVar10 * 4) = *piVar5;
      uVar11 = (ulonglong)((int)uVar11 + 1);
      uVar10 = uVar10 + 1;
    } while ((longlong)uVar10 < lVar7);
  }
  local_78 = *(longlong **)(param_1 + 0x160);
  uStack_70 = *(undefined8 *)(param_1 + 0x162);
  local_68 = *(ulonglong *)(param_1 + 0x164);
  lStack_60 = *(longlong *)(param_1 + 0x166);
  FUN_180003eb0(*(char *)((longlong)param_1 + 0x25),iVar3,*param_1,param_1[1],(longlong *)&local_78,
                *(longlong *)(param_3 + 0x10),lVar9,fVar1,fVar2,fVar12,param_4,param_5,param_6);
  param_2[1] = param_2[1] - uVar8;
  return;
}


```

## {'procedure': '0x18001ba70'}
Evidence: `ev_f69c6fe6de982545d16e426e8d8800cc02400ed30c51eaf0faa1b4ccb16e877a`

```c

void FUN_18001ba70(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5)

{
  float fVar1;
  float fVar2;
  int iVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  longlong lVar9;
  longlong lVar10;
  longlong lVar11;
  longlong lVar12;
  longlong lVar13;
  longlong lVar14;
  longlong lVar15;
  
  if (0 < param_2) {
    lVar12 = (longlong)param_1;
    lVar13 = 0;
    do {
      lVar11 = 0;
      if (3 < lVar12) {
        lVar4 = *param_3;
        lVar5 = param_3[1];
        lVar6 = param_3[2];
        lVar7 = param_3[3];
        do {
          fVar1 = *(float *)(lVar5 + lVar13 * 4);
          iVar3 = *(int *)(lVar7 + lVar13 * 4);
          lVar9 = lVar11 * 4;
          fVar2 = *(float *)(lVar9 + lVar4);
          lVar14 = (longlong)(iVar3 * (int)param_4[1]) + *param_4;
          lVar8 = (longlong)*(int *)(lVar9 + lVar6);
          lVar10 = (longlong)((iVar3 + 1) * (int)param_4[1]) + *param_4;
          *(float *)((int)param_5[1] * lVar13 + lVar9 + *param_5) =
               (1.0 - fVar1) * fVar2 * *(float *)(lVar14 + 4 + lVar8 * 4) +
               (1.0 - fVar2) * (1.0 - fVar1) * *(float *)(lVar14 + lVar8 * 4) +
               (1.0 - fVar2) * fVar1 * *(float *)(lVar10 + lVar8 * 4) +
               fVar1 * fVar2 * *(float *)(lVar10 + 4 + lVar8 * 4);
          fVar1 = *(float *)(lVar9 + 4 + lVar4);
          fVar2 = *(float *)(lVar5 + lVar13 * 4);
          iVar3 = *(int *)(lVar7 + lVar13 * 4);
          lVar14 = (longlong)(iVar3 * (int)param_4[1]) + *param_4;
          lVar8 = (longlong)*(int *)(lVar9 + 4 + lVar6);
          lVar10 = (longlong)((iVar3 + 1) * (int)param_4[1]) + *param_4;
          lVar15 = lVar9 + 0xc;
          *(float *)((int)param_5[1] * lVar13 + lVar9 + 4 + *param_5) =
               (1.0 - fVar2) * fVar1 * *(float *)(lVar14 + 4 + lVar8 * 4) +
               (1.0 - fVar1) * (1.0 - fVar2) * *(float *)(lVar14 + lVar8 * 4) +
               (1.0 - fVar1) * fVar2 * *(float *)(lVar10 + lVar8 * 4) +
               fVar2 * fVar1 * *(float *)(lVar10 + 4 + lVar8 * 4);
          fVar1 = *(float *)(lVar9 + 8 + lVar4);
          fVar2 = *(float *)(lVar5 + lVar13 * 4);
          iVar3 = *(int *)(lVar7 + lVar13 * 4);
          lVar11 = lVar11 + 4;
          lVar14 = (longlong)(iVar3 * (int)param_4[1]) + *param_4;
          lVar8 = (longlong)*(int *)(lVar9 + 8 + lVar6);
          lVar10 = (longlong)((iVar3 + 1) * (int)param_4[1]) + *param_4;
          *(float *)((int)param_5[1] * lVar13 + lVar9 + 8 + *param_5) =
               (1.0 - fVar2) * fVar1 * *(float *)(lVar14 + 4 + lVar8 * 4) +
               (1.0 - fVar1) * (1.0 - fVar2) * *(float *)(lVar14 + lVar8 * 4) +
               (1.0 - fVar1) * fVar2 * *(float *)(lVar10 + lVar8 * 4) +
               fVar2 * fVar1 * *(float *)(lVar10 + 4 + lVar8 * 4);
          fVar1 = *(float *)(lVar15 + lVar4);
          fVar2 = *(float *)(lVar5 + lVar13 * 4);
          iVar3 = *(int *)(lVar7 + lVar13 * 4);
          lVar10 = (longlong)(iVar3 * (int)param_4[1]) + *param_4;
          lVar9 = (longlong)*(int *)(lVar15 + lVar6);
          lVar8 = (longlong)((iVar3 + 1) * (int)param_4[1]) + *param_4;
          *(float *)((int)param_5[1] * lVar13 + lVar15 + *param_5) =
               (1.0 - fVar2) * fVar1 * *(float *)(lVar10 + 4 + lVar9 * 4) +
               (1.0 - fVar1) * (1.0 - fVar2) * *(float *)(lVar10 + lVar9 * 4) +
               (1.0 - fVar1) * fVar2 * *(float *)(lVar8 + lVar9 * 4) +
               fVar2 * fVar1 * *(float *)(lVar8 + 4 + lVar9 * 4);
        } while (lVar11 < lVar12 + -3);
      }
      if (lVar11 < lVar12) {
        lVar4 = *param_3;
        lVar5 = param_3[1];
        lVar6 = param_3[2];
        lVar7 = param_3[3];
        do {
          fVar1 = *(float *)(lVar5 + lVar13 * 4);
          iVar3 = *(int *)(lVar7 + lVar13 * 4);
          lVar9 = lVar11 * 4;
          lVar11 = lVar11 + 1;
          fVar2 = *(float *)(lVar9 + lVar4);
          lVar14 = (longlong)(iVar3 * (int)param_4[1]) + *param_4;
          lVar8 = (longlong)*(int *)(lVar9 + lVar6);
          lVar10 = (longlong)((iVar3 + 1) * (int)param_4[1]) + *param_4;
          *(float *)((int)param_5[1] * lVar13 + lVar9 + *param_5) =
               (1.0 - fVar1) * fVar2 * *(float *)(lVar14 + 4 + lVar8 * 4) +
               (1.0 - fVar2) * (1.0 - fVar1) * *(float *)(lVar14 + lVar8 * 4) +
               (1.0 - fVar2) * fVar1 * *(float *)(lVar10 + lVar8 * 4) +
               fVar1 * fVar2 * *(float *)(lVar10 + 4 + lVar8 * 4);
        } while (lVar11 < lVar12);
      }
      lVar13 = lVar13 + 1;
    } while (lVar13 < param_2);
  }
  return;
}


```
