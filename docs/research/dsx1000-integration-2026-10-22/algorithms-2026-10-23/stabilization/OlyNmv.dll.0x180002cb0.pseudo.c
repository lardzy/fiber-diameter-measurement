// ev_2cf0d31cc6e097e18339e06a5f97f035c131dbad129cf09a38f266b7786f5be2
// 3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180002cb0(undefined8 *param_1)

{
  undefined8 *puVar1;
  longlong lVar2;
  undefined8 *puVar3;
  int iVar4;
  ulonglong uVar5;
  undefined8 *puVar6;
  int iVar7;
  ulonglong uVar8;
  int iVar9;
  undefined1 auStack_318 [32];
  undefined8 *local_2f8;
  undefined8 *puStack_2f0;
  undefined8 local_2e8;
  _OutputArray local_2e0 [32];
  int local_2c0 [3];
  int local_2b4;
  int local_2b0;
  int local_2ac;
  int local_2a8;
  int local_2a4;
  int local_2a0 [3];
  int local_294;
  undefined8 local_290;
  int local_288;
  int local_284;
  undefined8 local_280;
  undefined8 local_278;
  undefined8 local_270;
  undefined8 local_268;
  int *piStack_260;
  undefined8 local_258;
  undefined8 local_250;
  undefined8 local_248;
  undefined8 local_240;
  undefined8 *local_238;
  undefined8 *local_230;
  undefined8 local_228;
  undefined8 local_220;
  Mat local_218 [16];
  undefined8 local_208;
  int *local_200;
  undefined8 local_1f8;
  undefined8 local_1f0;
  undefined8 local_1e8;
  undefined4 *local_1d8;
  undefined1 *local_1d0;
  undefined1 local_1c8 [16];
  Mat local_1b8 [16];
  undefined8 local_1a8;
  int *local_1a0;
  undefined8 local_198;
  undefined8 local_190;
  undefined8 local_188;
  undefined4 *local_178;
  undefined1 *local_170;
  undefined1 local_168 [16];
  Mat local_158 [8];
  int local_150;
  int local_14c;
  undefined8 local_148;
  int *local_140;
  undefined8 local_138;
  undefined8 local_130;
  undefined8 local_128;
  undefined4 *local_118;
  undefined1 *local_110;
  undefined1 local_108 [16];
  Mat local_f8 [16];
  undefined8 local_e8;
  int *local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  undefined8 local_c8;
  undefined4 *local_b8;
  undefined1 *local_b0;
  undefined1 local_a8 [16];
  Mat local_98 [16];
  undefined8 local_88;
  int *local_80;
  undefined8 local_78;
  undefined8 local_70;
  undefined8 local_68;
  undefined4 *local_58;
  undefined1 *local_50;
  undefined1 local_48 [16];
  ulonglong local_38;
  
  local_280 = 0xfffffffffffffffe;
  local_38 = DAT_18001a000 ^ (ulonglong)auStack_318;
  (**(code **)*param_1)(param_1,local_158,0xffffffff);
  uVar5 = 0;
  if ((local_150 != 1) || (local_14c != 1)) {
    local_2f8 = (undefined8 *)0x0;
    puStack_2f0 = (undefined8 *)0x0;
    local_2e8 = 0;
    cv::split(local_158,(vector<cv::Mat,std::allocator<cv::Mat>_> *)&local_2f8);
    iVar9 = local_150 >> 1;
    iVar7 = local_14c >> 1;
    if ((iVar7 == 0) || (iVar9 == 0)) {
      iVar7 = iVar7 + iVar9;
      lVar2 = (longlong)puStack_2f0 - (longlong)local_2f8 >> 0x3f;
      uVar8 = uVar5;
      if (((longlong)puStack_2f0 - (longlong)local_2f8) / 0x60 + lVar2 != lVar2) {
        do {
          local_238 = &local_270;
          local_230 = &local_228;
          local_220 = 0;
          local_228 = 0;
          local_278 = 0x42ff0000;
          local_270 = 0;
          local_248 = 0;
          local_250 = 0;
          local_258 = 0;
          local_268 = 0;
          piStack_260 = (int *)0x0;
          local_240 = 0;
          local_2c0[0] = 0;
          local_2c0[1] = 0;
          local_2b4 = 1;
          local_2c0[2] = iVar7;
          cv::Mat::Mat(local_218,(Mat *)(uVar5 + (longlong)local_2f8),(Rect_<int> *)local_2c0);
          local_2ac = 0;
          local_2a4 = 1;
          local_2b0 = iVar7;
          local_2a8 = iVar7;
          cv::Mat::Mat(local_1b8,(Mat *)((longlong)local_2f8 + uVar5),(Rect_<int> *)&local_2b0);
          cv::_OutputArray::_OutputArray(local_2e0,(Mat *)&local_278);
          cv::Mat::copyTo(local_218,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,local_218);
          cv::Mat::copyTo(local_1b8,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,local_1b8);
          cv::Mat::copyTo((Mat *)&local_278,local_2e0);
          if ((local_1a0 != (int *)0x0) &&
             (iVar9 = cv::_interlockedExchangeAdd(local_1a0,-1), iVar9 == 1)) {
            cv::Mat::deallocate(local_1b8);
          }
          local_188 = 0;
          local_190 = 0;
          local_198 = 0;
          local_1a8 = 0;
          *local_178 = 0;
          local_1a0 = (int *)0x0;
          if (local_170 != local_168) {
            cv::fastFree(local_170);
          }
          if ((local_200 != (int *)0x0) &&
             (iVar9 = cv::_interlockedExchangeAdd(local_200,-1), iVar9 == 1)) {
            cv::Mat::deallocate(local_218);
          }
          local_1e8 = 0;
          local_1f0 = 0;
          local_1f8 = 0;
          local_208 = 0;
          *local_1d8 = 0;
          local_200 = (int *)0x0;
          if (local_1d0 != local_1c8) {
            cv::fastFree(local_1d0);
          }
          if ((piStack_260 != (int *)0x0) &&
             (iVar9 = cv::_interlockedExchangeAdd(piStack_260,-1), iVar9 == 1)) {
            cv::Mat::deallocate((Mat *)&local_278);
          }
          local_248 = 0;
          local_250 = 0;
          local_258 = 0;
          local_268 = 0;
          *(undefined4 *)local_238 = 0;
          piStack_260 = (int *)0x0;
          if (local_230 != &local_228) {
            cv::fastFree(local_230);
          }
          uVar8 = uVar8 + 1;
          uVar5 = uVar5 + 0x60;
        } while (uVar8 < (ulonglong)(((longlong)puStack_2f0 - (longlong)local_2f8) / 0x60));
      }
    }
    else {
      lVar2 = (longlong)puStack_2f0 - (longlong)local_2f8 >> 0x3f;
      uVar8 = uVar5;
      if (((longlong)puStack_2f0 - (longlong)local_2f8) / 0x60 + lVar2 != lVar2) {
        do {
          local_238 = &local_270;
          local_230 = &local_228;
          local_220 = 0;
          local_228 = 0;
          local_278 = 0x42ff0000;
          local_270 = 0;
          local_248 = 0;
          local_250 = 0;
          local_258 = 0;
          local_268 = 0;
          piStack_260 = (int *)0x0;
          local_240 = 0;
          local_290 = 0;
          local_288 = iVar7;
          local_284 = iVar9;
          cv::Mat::Mat(local_1b8,(Mat *)(uVar8 + (longlong)local_2f8),(Rect_<int> *)&local_290);
          local_2a0[1] = 0;
          local_2a0[0] = iVar7;
          local_2a0[2] = iVar7;
          local_294 = iVar9;
          cv::Mat::Mat(local_218,(Mat *)((longlong)local_2f8 + uVar8),(Rect_<int> *)local_2a0);
          local_2b0 = 0;
          local_2ac = iVar9;
          local_2a8 = iVar7;
          local_2a4 = iVar9;
          cv::Mat::Mat(local_f8,(Mat *)((longlong)local_2f8 + uVar8),(Rect_<int> *)&local_2b0);
          local_2c0[1] = iVar9;
          local_2c0[0] = iVar7;
          local_2c0[2] = iVar7;
          local_2b4 = iVar9;
          cv::Mat::Mat(local_98,(Mat *)((longlong)local_2f8 + uVar8),(Rect_<int> *)local_2c0);
          cv::_OutputArray::_OutputArray(local_2e0,(Mat *)&local_278);
          cv::Mat::copyTo(local_1b8,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,local_1b8);
          cv::Mat::copyTo(local_98,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,local_98);
          cv::Mat::copyTo((Mat *)&local_278,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,(Mat *)&local_278);
          cv::Mat::copyTo(local_218,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,local_218);
          cv::Mat::copyTo(local_f8,local_2e0);
          cv::_OutputArray::_OutputArray(local_2e0,local_f8);
          cv::Mat::copyTo((Mat *)&local_278,local_2e0);
          if ((local_80 != (int *)0x0) &&
             (iVar4 = cv::_interlockedExchangeAdd(local_80,-1), iVar4 == 1)) {
            cv::Mat::deallocate(local_98);
          }
          local_68 = 0;
          local_70 = 0;
          local_78 = 0;
          local_88 = 0;
          *local_58 = 0;
          local_80 = (int *)0x0;
          if (local_50 != local_48) {
            cv::fastFree(local_50);
          }
          if ((local_e0 != (int *)0x0) &&
             (iVar4 = cv::_interlockedExchangeAdd(local_e0,-1), iVar4 == 1)) {
            cv::Mat::deallocate(local_f8);
          }
          local_c8 = 0;
          local_d0 = 0;
          local_d8 = 0;
          local_e8 = 0;
          *local_b8 = 0;
          local_e0 = (int *)0x0;
          if (local_b0 != local_a8) {
            cv::fastFree(local_b0);
          }
          if ((local_200 != (int *)0x0) &&
             (iVar4 = cv::_interlockedExchangeAdd(local_200,-1), iVar4 == 1)) {
            cv::Mat::deallocate(local_218);
          }
          local_1e8 = 0;
          local_1f0 = 0;
          local_1f8 = 0;
          local_208 = 0;
          *local_1d8 = 0;
          local_200 = (int *)0x0;
          if (local_1d0 != local_1c8) {
            cv::fastFree(local_1d0);
          }
          if ((local_1a0 != (int *)0x0) &&
             (iVar4 = cv::_interlockedExchangeAdd(local_1a0,-1), iVar4 == 1)) {
            cv::Mat::deallocate(local_1b8);
          }
          local_188 = 0;
          local_190 = 0;
          local_198 = 0;
          local_1a8 = 0;
          *local_178 = 0;
          local_1a0 = (int *)0x0;
          if (local_170 != local_168) {
            cv::fastFree(local_170);
          }
          if ((piStack_260 != (int *)0x0) &&
             (iVar4 = cv::_interlockedExchangeAdd(piStack_260,-1), iVar4 == 1)) {
            cv::Mat::deallocate((Mat *)&local_278);
          }
          local_248 = 0;
          local_250 = 0;
          local_258 = 0;
          local_268 = 0;
          *(undefined4 *)local_238 = 0;
          piStack_260 = (int *)0x0;
          if (local_230 != &local_228) {
            cv::fastFree(local_230);
          }
          uVar5 = uVar5 + 1;
          uVar8 = uVar8 + 0x60;
        } while (uVar5 < (ulonglong)(((longlong)puStack_2f0 - (longlong)local_2f8) / 0x60));
      }
    }
    cv::_OutputArray::_OutputArray(local_2e0,local_158);
    cv::merge((vector<cv::Mat,std::allocator<cv::Mat>_> *)&local_2f8,local_2e0);
    puVar3 = puStack_2f0;
    if (local_2f8 != (undefined8 *)0x0) {
      if (local_2f8 != puStack_2f0) {
        puVar6 = local_2f8 + 5;
        do {
          if (((int *)puVar6[-2] != (int *)0x0) &&
             (iVar7 = cv::_interlockedExchangeAdd((int *)puVar6[-2],-1), iVar7 == 1)) {
            cv::Mat::deallocate((Mat *)(puVar6 + -5));
          }
          puVar6[1] = 0;
          *puVar6 = 0;
          puVar6[-1] = 0;
          puVar6[-3] = 0;
          *(undefined4 *)puVar6[3] = 0;
          puVar6[-2] = 0;
          if ((undefined8 *)puVar6[4] != puVar6 + 5) {
            cv::fastFree((undefined8 *)puVar6[4]);
          }
          puVar1 = puVar6 + 7;
          puVar6 = puVar6 + 0xc;
        } while (puVar1 != puVar3);
      }
      operator_delete(local_2f8);
      local_2f8 = (undefined8 *)0x0;
      puStack_2f0 = (undefined8 *)0x0;
      local_2e8 = 0;
    }
  }
  if ((local_140 != (int *)0x0) && (iVar7 = cv::_interlockedExchangeAdd(local_140,-1), iVar7 == 1))
  {
    cv::Mat::deallocate(local_158);
  }
  local_128 = 0;
  local_130 = 0;
  local_138 = 0;
  local_148 = 0;
  *local_118 = 0;
  local_140 = (int *)0x0;
  if (local_110 != local_108) {
    cv::fastFree(local_110);
  }
  return;
}

