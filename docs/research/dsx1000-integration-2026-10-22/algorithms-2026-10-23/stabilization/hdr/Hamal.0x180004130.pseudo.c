// ev_4bf95baca06f7ed2580717fd5f7a030b7af21b9f2a5da6309aafb2b00c051371

ulonglong FUN_180004130(longlong param_1,longlong param_2,int *param_3)

{
  ulonglong uVar1;
  wchar_t *pwVar2;
  wchar_t local_28 [8];
  undefined8 local_18;
  undefined8 local_10;
  
  uVar1 = FUN_1800033a0(param_1,(longlong)param_3);
  if ((char)uVar1 != '\0') {
    if (param_2 == 0) {
      uVar1 = 0x13;
      pwVar2 = L"Input image is NULL";
    }
    else {
      if (param_3[0x4b6] == 0) {
LAB_180004187:
        return CONCAT71((int7)(uVar1 >> 8),1);
      }
      uVar1 = FUN_180004910(param_1,param_2,param_3);
      if ((char)uVar1 != '\0') goto LAB_180004187;
      uVar1 = 0x1f;
      pwVar2 = L"Error IridixThumScaleLpfMix8bit";
    }
    local_10 = 7;
    local_18 = 0;
    local_28[0] = L'\0';
    FUN_180006100((ulonglong *)local_28,pwVar2,uVar1);
    uVar1 = FUN_180009e10((longlong *)(param_1 + 0x78),4000,local_28,0);
  }
  return uVar1 & 0xffffffffffffff00;
}

