## procedure_pseudo_code {'procedure': '0x180004490'}
Evidence: `ev_09c4446010d435aa6c406bef93555cb198c9f2277268c247cde8e095ef24c0de`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
EFIMV_EndConstruct(undefined8 param_1,longlong param_2,undefined8 param_3,longlong param_4)

{
  undefined8 uVar1;
  undefined1 auStackY_68 [32];
  undefined8 local_38;
  longlong local_30 [2];
  longlong local_20;
  undefined8 local_18;
  ulonglong local_10;
  
                    /* 0x4490  4  EFIMV_EndConstruct */
  local_10 = DAT_180008000 ^ (ulonglong)auStackY_68;
  local_38 = 0;
  local_30[0] = 0;
  local_20 = param_4;
  local_18 = param_3;
  uVar1 = FUN_180002c70(param_2,param_3,param_4,&local_38,local_30);
  if ((int)uVar1 == 0) {
    uVar1 = EFI_EndConstruct(param_1,local_38,local_30[0]);
    if ((int)uVar1 == 0) {
      uVar1 = FUN_180002cc0(param_2,local_38,local_30[0],&local_18,&local_20);
      if ((int)uVar1 == 0) {
        uVar1 = 0;
      }
    }
  }
  return uVar1;
}


```

## procedure_pseudo_code {'procedure': '0x1800028b0'}
Evidence: `ev_4578638030f20c660b4f3b9025dbae51735f21c35168af7850cc810a9f08277f`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_1800028b0(longlong param_1,undefined8 param_2,undefined8 param_3,double param_4,double param_5)

{
  int *piVar1;
  uint uVar2;
  ulonglong uVar3;
  ulonglong uVar4;
  ulonglong uVar5;
  int iVar6;
  undefined1 auStackY_218 [32];
  undefined4 local_1d8;
  undefined4 local_1d4;
  undefined4 local_1d0 [2];
  Mat *local_1c8;
  undefined8 local_1c0;
  undefined4 local_1b8 [2];
  Mat *local_1b0;
  undefined8 local_1a8;
  undefined4 local_1a0 [2];
  Mat *local_198;
  undefined8 local_190;
  undefined8 local_188;
  undefined8 uStack_180;
  undefined8 local_178;
  undefined8 uStack_170;
  undefined8 local_168;
  Mat local_158 [4];
  int local_154;
  undefined4 *local_148;
  undefined8 local_140;
  undefined8 local_138;
  undefined8 local_130;
  longlong local_120;
  longlong local_118;
  longlong *local_110;
  longlong local_108 [2];
  Mat local_f8 [4];
  int local_f4;
  undefined8 local_e8;
  undefined8 local_e0;
  undefined8 local_d8;
  undefined8 local_d0;
  longlong local_c0;
  undefined4 *local_b8;
  undefined1 *local_b0;
  undefined1 local_a8 [16];
  Mat local_98 [4];
  int local_94;
  undefined8 local_88;
  undefined8 local_80;
  undefined8 local_78;
  undefined8 local_70;
  longlong local_60;
  longlong local_58;
  undefined1 *local_50;
  undefined1 local_48 [16];
  ulonglong local_38;
  
  local_168 = 0xfffffffffffffffe;
  local_38 = DAT_180008000 ^ (ulonglong)auStackY_218;
  uVar4 = 0;
  iVar6 = 2;
  if (*(int *)(param_1 + 200) < 9) {
    iVar6 = 0;
  }
  iVar6 = iVar6 + *(int *)(param_1 + 0xc4) * 8;
  memset(local_98,0,0x60);
  FUN_180001340((uint *)local_98,*(uint *)(param_1 + 0xb4),*(uint *)(param_1 + 0xb0),iVar6 - 8,
                param_2,(longlong)*(int *)(param_1 + 0xb8));
  memset(local_f8,0,0x60);
  FUN_180001340((uint *)local_f8,*(uint *)(param_1 + 0x2e4),*(uint *)(param_1 + 0x2e0),iVar6 - 8,
                param_3,(longlong)*(int *)(param_1 + 0x2e8));
  memset(local_158,0,0x60);
  FUN_180001280(local_158);
  *local_148 = 0x3f800000;
  local_148[1] = 0;
  local_148[2] = (float)(param_4 - (double)*(uint *)(param_1 + 0x84));
  *(undefined4 *)((longlong)local_148 + *local_110) = 0;
  *(undefined4 *)(*local_110 + 4 + (longlong)local_148) = 0x3f800000;
  *(float *)(*local_110 + 8 + (longlong)local_148) =
       (float)(param_5 - (double)*(uint *)(param_1 + 0x88));
  local_1a8 = 0;
  local_1b8[0] = 0x1010000;
  local_1b0 = local_158;
  local_190 = 0;
  local_1a0[0] = 0x2010000;
  local_198 = local_f8;
  local_1c0 = 0;
  local_1d0[0] = 0x1010000;
  local_1c8 = local_98;
  local_178 = 0;
  uStack_170 = 0;
  local_188 = 0;
  uStack_180 = 0;
  local_1d4 = *local_b8;
  local_1d8 = local_b8[1];
  cv::warpAffine(local_1d0,local_1a0,local_1b8,&local_1d8);
  if (local_120 != 0) {
    LOCK();
    piVar1 = (int *)(local_120 + 0x14);
    iVar6 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar6 == 1) {
      cv::Mat::deallocate(local_158);
    }
  }
  local_120 = 0;
  local_148 = (undefined4 *)0x0;
  local_130 = 0;
  local_138 = 0;
  local_140 = 0;
  uVar3 = uVar4;
  uVar5 = uVar4;
  if (0 < local_154) {
    do {
      *(undefined4 *)(uVar5 + local_118) = 0;
      uVar2 = (int)uVar3 + 1;
      uVar3 = (ulonglong)uVar2;
      uVar5 = uVar5 + 4;
    } while ((int)uVar2 < local_154);
  }
  if (local_110 != local_108) {
    cv::fastFree(local_110);
  }
  if (local_c0 != 0) {
    LOCK();
    piVar1 = (int *)(local_c0 + 0x14);
    iVar6 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar6 == 1) {
      cv::Mat::deallocate(local_f8);
    }
  }
  local_c0 = 0;
  local_e8 = 0;
  local_d0 = 0;
  local_d8 = 0;
  local_e0 = 0;
  uVar3 = uVar4;
  uVar5 = uVar4;
  if (0 < local_f4) {
    do {
      *(undefined4 *)(uVar5 + (longlong)local_b8) = 0;
      uVar2 = (int)uVar3 + 1;
      uVar3 = (ulonglong)uVar2;
      uVar5 = uVar5 + 4;
    } while ((int)uVar2 < local_f4);
  }
  if (local_b0 != local_a8) {
    cv::fastFree(local_b0);
  }
  if (local_60 != 0) {
    LOCK();
    piVar1 = (int *)(local_60 + 0x14);
    iVar6 = *piVar1;
    *piVar1 = *piVar1 + -1;
    UNLOCK();
    if (iVar6 == 1) {
      cv::Mat::deallocate(local_98);
    }
  }
  local_60 = 0;
  local_88 = 0;
  local_70 = 0;
  local_78 = 0;
  local_80 = 0;
  uVar3 = uVar4;
  if (0 < local_94) {
    do {
      *(undefined4 *)(uVar3 + local_58) = 0;
      uVar2 = (int)uVar4 + 1;
      uVar4 = (ulonglong)uVar2;
      uVar3 = uVar3 + 4;
    } while ((int)uVar2 < local_94);
  }
  if (local_50 != local_48) {
    cv::fastFree(local_50);
  }
  return 0;
}


```

## procedure_pseudo_code {'procedure': '0x180002490'}
Evidence: `ev_78891cdd2bcfd8566cec695e965adfefb45c73bedea96965b2540cdf05079728`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_180002490(longlong param_1,longlong param_2,undefined8 *param_3,ulonglong *param_4,
             undefined8 *param_5)

{
  int iVar1;
  undefined8 uVar2;
  undefined4 uVar3;
  uint uVar4;
  undefined1 auStack_138 [32];
  longlong local_118;
  longlong local_110;
  undefined4 local_108;
  undefined1 local_100;
  undefined8 local_f8;
  undefined1 local_f0;
  ulonglong local_e8;
  undefined8 local_e0;
  undefined8 local_d8 [2];
  undefined8 local_c8;
  undefined1 local_c0 [8];
  ulonglong local_b8;
  ulonglong local_58;
  
  local_58 = DAT_180008000 ^ (ulonglong)auStack_138;
  uVar3 = 0;
  uVar4 = 0;
  local_d8[0] = 0;
  local_e8 = 0;
  local_e0 = 0;
  if (*(int *)(param_1 + 0x7c) == 1) {
    local_c8 = 0;
    memset(local_c0,0,0x60);
    uVar2 = FUN_180002720(param_1,param_2,(longlong *)(param_1 + 0x5a8));
    if ((int)uVar2 != 0) {
      return uVar2;
    }
    if (*(int *)(param_1 + 0xc4) == 1) {
      iVar1 = (**(code **)(param_1 + 0x10))();
    }
    else {
      iVar1 = (**(code **)(param_1 + 0x18))(*(undefined8 *)(param_1 + 0x5a8),&local_c8);
    }
    if (iVar1 != 0) {
      if (iVar1 == -3) {
        return 0x2008;
      }
      if (iVar1 == -2) {
        return 0x2007;
      }
      if (iVar1 == -1) {
        return 0x2006;
      }
      return 0;
    }
    uVar3 = local_c0._0_4_;
    uVar4 = local_c0._4_4_ ^ 0x80000000;
    local_e8 = local_b8 ^ 0x8000000000000000;
  }
  else {
    if (*(int *)(param_1 + 0x7c) != 0) {
      return 0x11;
    }
    *(longlong *)(param_1 + 0x540) = param_2;
    if (*(int *)(param_1 + 0x510) == 0) {
      iVar1 = (**(code **)(param_1 + 0x60))
                        (param_1 + 0x548,param_1 + 0x528,*(undefined4 *)(param_1 + 0x98));
      if (iVar1 != 0) {
        if (iVar1 == -3) {
          return 0x2005;
        }
        if (iVar1 == -2) {
          return 0x2004;
        }
        if (iVar1 == -1) {
          return 0x2003;
        }
      }
      local_c8 = 0x3ff0000000000000;
      local_e8 = 0;
    }
    else {
      iVar1 = (**(code **)(param_1 + 0x60))
                        (param_1 + 0x568,param_1 + 0x528,*(undefined4 *)(param_1 + 0x98));
      if (iVar1 != 0) {
        if (iVar1 == -3) {
          return 0x2005;
        }
        if (iVar1 == -2) {
          return 0x2004;
        }
        if (iVar1 == -1) {
          return 0x2003;
        }
      }
      local_100 = *(undefined1 *)(param_1 + 0x9c);
      local_f8 = *(undefined8 *)(param_1 + 0xa0);
      local_f0 = 1;
      local_108 = *(undefined4 *)(param_1 + 0x98);
      local_110 = param_1 + 0x588;
      local_118 = param_1 + 0x568;
      iVar1 = (**(code **)(param_1 + 0x70))(local_d8,&local_e8,&local_e0,param_1 + 0x548);
      if (iVar1 != 0) {
        if (iVar1 == -3) {
          return 0x2005;
        }
        if (iVar1 == -2) {
          return 0x2004;
        }
        if (iVar1 == -1) {
          return 0x2003;
        }
      }
      uVar2 = *(undefined8 *)(param_1 + 0x560);
      uVar3 = (undefined4)local_d8[0];
      uVar4 = (uint)((ulonglong)local_d8[0] >> 0x20);
      *(undefined8 *)(param_1 + 0x560) = *(undefined8 *)(param_1 + 0x580);
      *(undefined8 *)(param_1 + 0x580) = uVar2;
      local_c8 = local_e0;
    }
  }
  *param_3 = CONCAT44(uVar4,uVar3);
  *param_4 = local_e8;
  *param_5 = local_c8;
  return 0;
}


```

## procedure_pseudo_code {'procedure': '0x180004290'}
Evidence: `ev_7f3d160f0de5958de31bc7413691275e66bdd40f35476331ae52066dc0d3afbe`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
EFIMV_Construct(undefined8 param_1,longlong param_2,undefined8 param_3,longlong param_4,
               undefined8 param_5)

{
  longlong lVar1;
  undefined8 uVar2;
  undefined1 auStackY_88 [32];
  undefined8 local_58;
  longlong local_50 [2];
  longlong local_40;
  undefined8 local_38;
  ulonglong local_30;
  
                    /* 0x4290  2  EFIMV_Construct */
  local_30 = DAT_180008000 ^ (ulonglong)auStackY_88;
  local_58 = 0;
  local_50[0] = 0;
  local_40 = param_4;
  local_38 = param_3;
  uVar2 = FUN_180002c70(param_2,param_3,param_4,&local_58,local_50);
  lVar1 = local_50[0];
  if ((int)uVar2 == 0) {
    uVar2 = EFI_Construct(param_1,local_58,local_50[0],param_5);
    if ((int)uVar2 == 0) {
      uVar2 = FUN_180002cc0(param_2,local_58,lVar1,&local_38,&local_40);
      if ((int)uVar2 == 0) {
        uVar2 = 0;
      }
    }
    else if (((int)uVar2 == 0x18) &&
            (uVar2 = FUN_180003060(param_2,param_3,param_4), (int)uVar2 == 0)) {
      uVar2 = 0x18;
    }
  }
  return uVar2;
}


```

## procedure_pseudo_code {'procedure': '0x1800041b0'}
Evidence: `ev_91f33c139396ccceb37d08389eeef6b98e6ed702c4437e3b1b0d7bf087165256`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
EFIMV_InputImage(undefined8 param_1,longlong param_2,void *param_3,undefined4 param_4,
                longlong param_5)

{
  undefined8 uVar1;
  undefined1 auStackY_78 [32];
  undefined8 local_48 [2];
  longlong local_38;
  ulonglong local_30;
  
                    /* 0x41b0  6  EFIMV_InputImage */
  local_30 = DAT_180008000 ^ (ulonglong)auStackY_78;
  local_38 = param_5;
  local_48[0] = 0;
  uVar1 = FUN_1800022d0(param_2,param_3,param_4,local_48);
  if ((int)uVar1 == 0) {
    if (param_5 == 0) {
      uVar1 = EFI_InputImage(param_1,local_48[0],param_4,0);
    }
    else {
      if ((*(char *)(param_2 + 0x78) != '\0') && (*(int *)(param_2 + 0x80) != 1)) {
        param_5 = *(longlong *)(param_2 + 0x5c8);
      }
      uVar1 = EFI_InputImage(param_1,local_48[0],param_4,param_5);
      if ((int)uVar1 == 0) {
        uVar1 = FUN_180002cc0(param_2,param_5,0,&local_38,(longlong *)0x0);
        if ((int)uVar1 == 0) {
          uVar1 = 0;
        }
      }
    }
  }
  return uVar1;
}


```

## procedure_pseudo_code {'procedure': '0x180004390'}
Evidence: `ev_a83a488aa01aa6b72624344785b62c68408c0cd0a65744645ce69c4a200dc489`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
EFIMV_InputImageForConstruct(undefined8 param_1,longlong param_2,int param_3,longlong param_4)

{
  undefined8 uVar1;
  int iVar2;
  undefined1 auStackY_88 [32];
  undefined8 local_48;
  ulonglong local_40;
  
                    /* 0x4390  7  EFIMV_InputImageForConstruct */
  local_40 = DAT_180008000 ^ (ulonglong)auStackY_88;
  if (param_2 == 0) {
    uVar1 = EFI_InputImageForConstruct(param_1,param_3,param_4);
  }
  else {
    for (iVar2 = 0; iVar2 < param_3; iVar2 = iVar2 + 1) {
      uVar1 = *(undefined8 *)(param_4 + (longlong)iVar2 * 8);
      local_48 = uVar1;
      if (*(char *)(param_2 + 0x78) != '\0') {
        local_48 = *(undefined8 *)(param_2 + 0x5c0);
        uVar1 = FUN_1800028b0(param_2,uVar1,local_48,
                              -*(double *)
                                (*(longlong *)(param_2 + 0x518) +
                                (longlong)*(int *)(param_2 + 0x514) * 8),
                              -*(double *)
                                (*(longlong *)(param_2 + 0x520) +
                                (longlong)*(int *)(param_2 + 0x514) * 8));
        if ((int)uVar1 != 0) {
          return uVar1;
        }
        *(int *)(param_2 + 0x514) = *(int *)(param_2 + 0x514) + 1;
      }
      uVar1 = EFI_InputImageForConstruct(param_1,1,&local_48);
      if ((int)uVar1 != 0) {
        return uVar1;
      }
    }
    uVar1 = 0;
  }
  return uVar1;
}


```

## procedure_pseudo_code {'procedure': '0x180001ce0'}
Evidence: `ev_adc6c1f6c3871960ee4242a662d1f59381cf3733b87f35ea142eaad19c3a88f0`
```c

undefined8 FUN_180001ce0(int *param_1,char *param_2,int *param_3)

{
  undefined8 uVar1;
  int *piVar2;
  int *piVar3;
  int iVar4;
  int *piVar5;
  int iVar6;
  longlong lVar7;
  int *piVar8;
  
  if (((param_1 != (int *)0x0) && (param_2 != (char *)0x0)) && (param_3 != (int *)0x0)) {
    lVar7 = 4;
    piVar2 = param_1;
    piVar3 = param_3;
    do {
      piVar8 = piVar3;
      piVar5 = piVar2;
      uVar1 = *(undefined8 *)(piVar5 + 2);
      *(undefined8 *)piVar8 = *(undefined8 *)piVar5;
      *(undefined8 *)(piVar8 + 2) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 6);
      *(undefined8 *)(piVar8 + 4) = *(undefined8 *)(piVar5 + 4);
      *(undefined8 *)(piVar8 + 6) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 10);
      *(undefined8 *)(piVar8 + 8) = *(undefined8 *)(piVar5 + 8);
      *(undefined8 *)(piVar8 + 10) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 0xe);
      *(undefined8 *)(piVar8 + 0xc) = *(undefined8 *)(piVar5 + 0xc);
      *(undefined8 *)(piVar8 + 0xe) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 0x12);
      *(undefined8 *)(piVar8 + 0x10) = *(undefined8 *)(piVar5 + 0x10);
      *(undefined8 *)(piVar8 + 0x12) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 0x16);
      *(undefined8 *)(piVar8 + 0x14) = *(undefined8 *)(piVar5 + 0x14);
      *(undefined8 *)(piVar8 + 0x16) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 0x1a);
      *(undefined8 *)(piVar8 + 0x18) = *(undefined8 *)(piVar5 + 0x18);
      *(undefined8 *)(piVar8 + 0x1a) = uVar1;
      uVar1 = *(undefined8 *)(piVar5 + 0x1e);
      *(undefined8 *)(piVar8 + 0x1c) = *(undefined8 *)(piVar5 + 0x1c);
      *(undefined8 *)(piVar8 + 0x1e) = uVar1;
      lVar7 = lVar7 + -1;
      piVar2 = piVar5 + 0x20;
      piVar3 = piVar8 + 0x20;
    } while (lVar7 != 0);
    uVar1 = *(undefined8 *)(piVar5 + 0x22);
    *(undefined8 *)(piVar8 + 0x20) = *(undefined8 *)(piVar5 + 0x20);
    *(undefined8 *)(piVar8 + 0x22) = uVar1;
    uVar1 = *(undefined8 *)(piVar5 + 0x26);
    *(undefined8 *)(piVar8 + 0x24) = *(undefined8 *)(piVar5 + 0x24);
    *(undefined8 *)(piVar8 + 0x26) = uVar1;
    uVar1 = *(undefined8 *)(piVar5 + 0x2a);
    *(undefined8 *)(piVar8 + 0x28) = *(undefined8 *)(piVar5 + 0x28);
    *(undefined8 *)(piVar8 + 0x2a) = uVar1;
    if (*param_2 != '\0') {
      iVar6 = *param_1 + *(int *)(param_2 + 0xc) * -2;
      *param_3 = iVar6;
      param_3[1] = param_1[1] + *(int *)(param_2 + 0x10) * -2;
      iVar4 = ((int)(param_1[6] + (param_1[6] >> 0x1f & 7U)) >> 3) * param_1[5] * iVar6;
      param_3[2] = iVar4;
      param_3[3] = iVar4;
      param_3[4] = iVar6 * 4;
    }
    return 0;
  }
  return 0x11;
}


```

## procedure_pseudo_code {'procedure': '0x180003ff0'}
Evidence: `ev_b5bc3d59db921553a1df5b8871f713baa9f3822a41bd558fcfe94af922f317ef`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

ulonglong EFIMV_CreateContext(uint *param_1,char *param_2,longlong param_3,ulonglong param_4,
                             undefined8 param_5,undefined8 param_6,undefined8 param_7,
                             longlong *param_8)

{
  int iVar1;
  undefined4 extraout_var;
  ulonglong uVar2;
  undefined1 auStackY_2d8 [32];
  longlong local_298;
  longlong local_290 [3];
  int local_278;
  undefined1 local_274 [556];
  ulonglong local_48;
  
                    /* 0x3ff0  3  EFIMV_CreateContext */
  local_48 = DAT_180008000 ^ (ulonglong)auStackY_2d8;
  local_298 = 0;
  local_290[0] = 0;
  local_278 = 0;
  memset(local_274,0,0x22c);
  iVar1 = FUN_180001690(param_1,param_2);
  uVar2 = CONCAT44(extraout_var,iVar1);
  if (iVar1 == 0) {
    if ((param_1 == (uint *)0x0) || (param_2 == (char *)0x0)) {
      uVar2 = 0x11;
    }
    else {
      uVar2 = FUN_180001de0((int *)param_1,param_2,param_3,param_4,&local_298,local_290,param_8);
      if ((((int)uVar2 == 0) &&
          (uVar2 = FUN_180001f20((int *)param_1,param_2,(undefined8 *)*param_8), (int)uVar2 == 0))
         && (uVar2 = FUN_180001ce0((int *)param_1,param_2,&local_278), (int)uVar2 == 0)) {
        uVar2 = EFI_CreateContext(&local_278,local_298,local_290[0],param_5);
      }
    }
  }
  return uVar2;
}


```

## procedure_pseudo_code {'procedure': '0x1800022d0'}
Evidence: `ev_dcff2c3c3436528e84cb307e4a47eb21452593b85dfc13f2d9f7f243bd865459`
```c

undefined8 FUN_1800022d0(longlong param_1,void *param_2,undefined4 param_3,undefined8 *param_4)

{
  int iVar1;
  undefined8 uVar2;
  longlong lVar3;
  double dVar4;
  double dVar5;
  double dVar6;
  double dVar7;
  double local_res8;
  double local_28;
  double local_20 [3];
  
  local_res8 = 0.0;
  local_28 = 0.0;
  local_20[0] = 0.0;
  if (*(int *)(param_1 + 0x510) == 0) {
    *(undefined4 *)(param_1 + 0x5b0) = param_3;
    memcpy(*(void **)(param_1 + 0x5b8),param_2,
           (longlong)(*(int *)(param_1 + 0xb8) * *(int *)(param_1 + 0xb4)));
  }
  if (*(char *)(param_1 + 0x78) == '\0') {
    *param_4 = param_2;
    *(int *)(param_1 + 0x510) = *(int *)(param_1 + 0x510) + 1;
    uVar2 = 0;
  }
  else {
    *param_4 = *(undefined8 *)(param_1 + 0x5c0);
    uVar2 = FUN_180002490(param_1,(longlong)param_2,&local_res8,(ulonglong *)&local_28,local_20);
    if ((int)uVar2 == 0) {
      iVar1 = *(int *)(param_1 + 0x510);
      lVar3 = (longlong)iVar1;
      dVar5 = (double)*(uint *)(param_1 + 0x84);
      dVar7 = (double)*(uint *)(param_1 + 0x88);
      dVar4 = local_28;
      dVar6 = local_res8;
      if (iVar1 != 0) {
        dVar4 = local_28 + *(double *)(*(longlong *)(param_1 + 0x520) + -8 + lVar3 * 8);
        dVar6 = local_res8 + *(double *)(*(longlong *)(param_1 + 0x518) + -8 + lVar3 * 8);
      }
      if (*(double *)(param_1 + 0x90) <= local_20[0]) {
        if (dVar6 < -dVar5) {
          dVar6 = -dVar5;
        }
        if (dVar5 < dVar6) {
          dVar6 = dVar5;
        }
        if (dVar4 < -dVar7) {
          dVar4 = -dVar7;
        }
        if (dVar7 < dVar4) {
          dVar4 = dVar7;
        }
      }
      else if (iVar1 == 0) {
        dVar6 = 0.0;
        dVar4 = 0.0;
      }
      else {
        dVar6 = *(double *)(*(longlong *)(param_1 + 0x518) + -8 + lVar3 * 8);
        dVar4 = *(double *)(*(longlong *)(param_1 + 0x520) + -8 + lVar3 * 8);
      }
      *(double *)(*(longlong *)(param_1 + 0x518) + lVar3 * 8) = dVar6;
      *(double *)(*(longlong *)(param_1 + 0x520) + (longlong)*(int *)(param_1 + 0x510) * 8) = dVar4;
      *(int *)(param_1 + 0x510) = *(int *)(param_1 + 0x510) + 1;
      uVar2 = FUN_1800028b0(param_1,param_2,*param_4,-dVar6,-dVar4);
    }
  }
  return uVar2;
}


```

## procedure_pseudo_code {'procedure': '0x180001f20'}
Evidence: `ev_de565fdd414f3bf42964692c4ba1543f0d936dbc0476d8924ef7e998e067f58b`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8 FUN_180001f20(int *param_1,char *param_2,undefined8 *param_3)

{
  undefined4 uVar1;
  undefined8 *puVar2;
  int *piVar3;
  int iVar4;
  undefined8 *puVar5;
  undefined8 uVar6;
  int *piVar7;
  ulonglong uVar8;
  uint uVar9;
  longlong lVar10;
  ulonglong uVar11;
  undefined1 auStack_108 [32];
  int local_e8;
  int local_e0;
  undefined4 local_d8;
  undefined4 local_d0;
  int local_c8;
  int local_c0;
  undefined4 local_b8;
  undefined4 uStack_b4;
  undefined4 uStack_b0;
  undefined4 uStack_ac;
  int local_a8;
  int local_a0;
  undefined4 local_98;
  undefined4 local_90;
  undefined4 local_88;
  undefined8 local_80;
  undefined8 uStack_78;
  undefined4 local_70;
  undefined4 local_68;
  int local_58;
  undefined8 uStack_54;
  undefined8 uStack_4c;
  undefined4 local_40;
  undefined8 uStack_3c;
  undefined8 uStack_34;
  ulonglong local_28;
  
  local_28 = DAT_180008000 ^ (ulonglong)auStack_108;
  if (((param_1 == (int *)0x0) || (param_2 == (char *)0x0)) || (param_3 == (undefined8 *)0x0)) {
    return 0x11;
  }
  uVar8 = 0;
  *(undefined4 *)(param_3 + 0xb6) = 0;
  uVar6 = *(undefined8 *)(param_2 + 8);
  param_3[0xf] = *(undefined8 *)param_2;
  param_3[0x10] = uVar6;
  uVar6 = *(undefined8 *)(param_2 + 0x18);
  param_3[0x11] = *(undefined8 *)(param_2 + 0x10);
  param_3[0x12] = uVar6;
  uVar6 = *(undefined8 *)(param_2 + 0x28);
  param_3[0x13] = *(undefined8 *)(param_2 + 0x20);
  param_3[0x14] = uVar6;
  lVar10 = 4;
  param_3[0x15] = *(undefined8 *)(param_2 + 0x30);
  puVar2 = param_3 + 0x16;
  piVar3 = param_1;
  do {
    piVar7 = piVar3;
    puVar5 = puVar2;
    uVar6 = *(undefined8 *)(piVar7 + 2);
    *puVar5 = *(undefined8 *)piVar7;
    puVar5[1] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 6);
    puVar5[2] = *(undefined8 *)(piVar7 + 4);
    puVar5[3] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 10);
    puVar5[4] = *(undefined8 *)(piVar7 + 8);
    puVar5[5] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 0xe);
    puVar5[6] = *(undefined8 *)(piVar7 + 0xc);
    puVar5[7] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 0x12);
    puVar5[8] = *(undefined8 *)(piVar7 + 0x10);
    puVar5[9] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 0x16);
    puVar5[10] = *(undefined8 *)(piVar7 + 0x14);
    puVar5[0xb] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 0x1a);
    puVar5[0xc] = *(undefined8 *)(piVar7 + 0x18);
    puVar5[0xd] = uVar6;
    uVar6 = *(undefined8 *)(piVar7 + 0x1e);
    puVar5[0xe] = *(undefined8 *)(piVar7 + 0x1c);
    puVar5[0xf] = uVar6;
    lVar10 = lVar10 + -1;
    puVar2 = puVar5 + 0x10;
    piVar3 = piVar7 + 0x20;
  } while (lVar10 != 0);
  uVar6 = *(undefined8 *)(piVar7 + 0x22);
  puVar5[0x10] = *(undefined8 *)(piVar7 + 0x20);
  puVar5[0x11] = uVar6;
  uVar6 = *(undefined8 *)(piVar7 + 0x26);
  puVar5[0x12] = *(undefined8 *)(piVar7 + 0x24);
  puVar5[0x13] = uVar6;
  uVar6 = *(undefined8 *)(piVar7 + 0x2a);
  puVar5[0x14] = *(undefined8 *)(piVar7 + 0x28);
  puVar5[0x15] = uVar6;
  uVar6 = FUN_180001ce0(param_1,param_2,(int *)(param_3 + 0x5c));
  if ((int)uVar6 != 0) {
    return uVar6;
  }
  param_3[0xa2] = 0;
  if (*param_2 != '\0') {
    uVar11 = uVar8;
    if (0 < *(int *)((longlong)param_3 + 0xcc)) {
      do {
        uVar9 = (int)uVar11 + 1;
        *(undefined8 *)(uVar8 + param_3[0xa3]) = 0;
        *(undefined8 *)(uVar8 + param_3[0xa4]) = 0;
        uVar8 = uVar8 + 8;
        uVar11 = (ulonglong)uVar9;
      } while ((int)uVar9 < *(int *)((longlong)param_3 + 0xcc));
    }
    uVar6 = FUN_180001740(*(int *)(param_2 + 4),param_3);
    if ((int)uVar6 != 0) {
      return uVar6;
    }
    if (*(int *)(param_2 + 4) == 1) {
      (*(code *)param_3[4])();
      local_e8 = *param_1;
      local_e0 = param_1[1];
      uVar1 = *(undefined4 *)(param_2 + 0x30);
      local_68 = 1;
      local_70 = 0;
      local_80 = 0;
      uStack_78 = 0;
      local_88 = 0;
      local_90 = 0;
      local_98 = 0;
      local_b8 = 0;
      uStack_b4 = 0x40240000;
      uStack_b0 = 0;
      uStack_ac = 0x40240000;
      local_d0 = 0;
      local_d8 = 0;
      local_c8 = local_e8;
      local_c0 = local_e0;
      local_a8 = local_e8;
      local_a0 = local_e0;
      iVar4 = (*(code *)param_3[1])(1,uVar1,uVar1,uVar1);
      if (iVar4 == 0) {
        return 0;
      }
      if (iVar4 == -3) {
        return 0x2008;
      }
      if (iVar4 == -2) {
        return 0x2007;
      }
      if (iVar4 == -1) {
        return 0x2006;
      }
      return 0;
    }
    if (*(int *)(param_2 + 4) != 0) {
      return 0x11;
    }
    local_58 = 0;
    local_40 = 0;
    uStack_54 = 0;
    uStack_4c = 0;
    uStack_3c = 0;
    uStack_34 = 0;
    (*(code *)param_3[10])();
    iVar4 = (*(code *)param_3[9])(0);
    if (iVar4 != 0) {
      if (iVar4 == -3) {
        return 0x2005;
      }
      if (iVar4 == -2) {
        return 0x2004;
      }
      if (iVar4 == -1) {
        return 0x2003;
      }
    }
    local_58 = *param_1;
    uStack_54._0_4_ = param_1[1];
    uStack_54._4_4_ = param_1[5];
    uStack_4c._0_4_ = (int)(param_1[6] + (param_1[6] >> 0x1f & 7U)) >> 3;
    uStack_4c._4_4_ = param_1[2];
    iVar4 = (*(code *)param_3[0xb])(&local_40,&local_58,*(undefined4 *)(param_2 + 0x20));
    if (iVar4 != 0) {
      if (iVar4 == -3) {
        return 0x2005;
      }
      if (iVar4 == -2) {
        return 0x2004;
      }
      if (iVar4 == -1) {
        return 0x2003;
      }
    }
    *(int *)(param_3 + 0xa5) = local_58;
    *(int *)((longlong)param_3 + 0x52c) = (int)uStack_54;
    *(int *)(param_3 + 0xa6) = uStack_54._4_4_;
    *(int *)((longlong)param_3 + 0x534) = (int)uStack_4c;
    *(undefined4 *)(param_3 + 0xa9) = local_40;
    *(undefined4 *)((longlong)param_3 + 0x54c) = (undefined4)uStack_3c;
    *(undefined4 *)(param_3 + 0xaa) = uStack_3c._4_4_;
    *(undefined4 *)((longlong)param_3 + 0x554) = (undefined4)uStack_34;
    *(undefined4 *)(param_3 + 0xad) = local_40;
    *(undefined4 *)((longlong)param_3 + 0x56c) = (undefined4)uStack_3c;
    *(undefined4 *)(param_3 + 0xae) = uStack_3c._4_4_;
    *(undefined4 *)((longlong)param_3 + 0x574) = (undefined4)uStack_34;
    *(int *)(param_3 + 0xa7) = uStack_4c._4_4_;
    *(undefined4 *)(param_3 + 0xab) = uStack_34._4_4_;
    *(undefined4 *)(param_3 + 0xaf) = uStack_34._4_4_;
    *(undefined4 *)(param_3 + 0xb1) = local_40;
    *(undefined4 *)((longlong)param_3 + 0x58c) = (undefined4)uStack_3c;
    *(undefined4 *)(param_3 + 0xb2) = uStack_3c._4_4_;
    *(undefined4 *)((longlong)param_3 + 0x594) = (undefined4)uStack_34;
    *(undefined4 *)(param_3 + 0xb3) = uStack_34._4_4_;
    iVar4 = (*(code *)param_3[0xd])();
    if (iVar4 != 0) {
      if (iVar4 == -3) {
        return 0x2005;
      }
      if (iVar4 == -2) {
        return 0x2004;
      }
      if (iVar4 == -1) {
        return 0x2003;
      }
    }
  }
  return 0;
}


```

## procedure_pseudo_code {'procedure': '0x180001740'}
Evidence: `ev_e10926e0c8ced8ddb986326c944d8ab6ccdfa111da8887cd88eeeb3811daceeb`
```c

undefined8 FUN_180001740(int param_1,undefined8 *param_2)

{
  HMODULE pHVar1;
  FARPROC pFVar2;
  
  if (param_1 == 1) {
    pHVar1 = LoadLibraryW(L"ALMIS3ol64.dll");
    *param_2 = pHVar1;
    if (pHVar1 != (HMODULE)0x0) {
      pFVar2 = GetProcAddress(pHVar1,"ALMIS3Initial");
      pHVar1 = (HMODULE)*param_2;
      param_2[1] = pFVar2;
      if (pFVar2 != (FARPROC)0x0) {
        pFVar2 = GetProcAddress(pHVar1,"ALMIS3Execute");
        pHVar1 = (HMODULE)*param_2;
        param_2[2] = pFVar2;
        if (pFVar2 != (FARPROC)0x0) {
          pFVar2 = GetProcAddress(pHVar1,"ALMIS3Execute_RGB");
          pHVar1 = (HMODULE)*param_2;
          param_2[3] = pFVar2;
          if (pFVar2 != (FARPROC)0x0) {
            pFVar2 = GetProcAddress(pHVar1,"ALMIS3End");
            pHVar1 = (HMODULE)*param_2;
            param_2[4] = pFVar2;
            if (pFVar2 != (FARPROC)0x0) {
              pFVar2 = GetProcAddress(pHVar1,"ALMIS3CheckSecurityUsbKey");
              pHVar1 = (HMODULE)*param_2;
              param_2[5] = pFVar2;
              if (pFVar2 != (FARPROC)0x0) {
                pFVar2 = GetProcAddress(pHVar1,"ALMIS3GetSecurityErrorString");
                pHVar1 = (HMODULE)*param_2;
                param_2[6] = pFVar2;
                if (pFVar2 != (FARPROC)0x0) {
                  pFVar2 = GetProcAddress(pHVar1,"ALMIS3GetSecurityErrorNumber");
                  param_2[7] = pFVar2;
                  if (pFVar2 != (FARPROC)0x0) {
                    return 0;
                  }
                  pHVar1 = (HMODULE)*param_2;
                }
              }
            }
          }
        }
      }
LAB_180001830:
      FreeLibrary(pHVar1);
      return 0x2002;
    }
  }
  else {
    if (param_1 != 0) {
      return 0x11;
    }
    pHVar1 = LoadLibraryW(L"OlyNmv.dll");
    param_2[8] = pHVar1;
    if (pHVar1 != (HMODULE)0x0) {
      pFVar2 = GetProcAddress(pHVar1,"OLYNMV_Initialize");
      pHVar1 = (HMODULE)param_2[8];
      param_2[9] = pFVar2;
      if (pFVar2 != (FARPROC)0x0) {
        pFVar2 = GetProcAddress(pHVar1,"OLYNMV_Uninitialize");
        pHVar1 = (HMODULE)param_2[8];
        param_2[10] = pFVar2;
        if (pFVar2 != (FARPROC)0x0) {
          pFVar2 = GetProcAddress(pHVar1,"OLYNMV_CalcDstImageInfo");
          pHVar1 = (HMODULE)param_2[8];
          param_2[0xb] = pFVar2;
          if (pFVar2 != (FARPROC)0x0) {
            pFVar2 = GetProcAddress(pHVar1,"OLYNMV_ConvertToDstImage");
            pHVar1 = (HMODULE)param_2[8];
            param_2[0xc] = pFVar2;
            if (pFVar2 != (FARPROC)0x0) {
              pFVar2 = GetProcAddress(pHVar1,"OLYNMV_CreateWindow");
              pHVar1 = (HMODULE)param_2[8];
              param_2[0xd] = pFVar2;
              if (pFVar2 != (FARPROC)0x0) {
                pFVar2 = GetProcAddress(pHVar1,"OLYNMV_EstimateMV");
                param_2[0xe] = pFVar2;
                if (pFVar2 != (FARPROC)0x0) {
                  return 0;
                }
                FreeLibrary((HMODULE)param_2[8]);
                return 0x2002;
              }
            }
          }
        }
      }
      goto LAB_180001830;
    }
  }
  return 0x2001;
}


```

## procedure_pseudo_code {'procedure': '0x180002cc0'}
Evidence: `ev_eddf0d080abe37827270d7f85d472566bea8cc351f8254c587d52bae04842e2e`
```c

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

undefined8
FUN_180002cc0(longlong param_1,undefined8 param_2,longlong param_3,undefined8 *param_4,
             longlong *param_5)

{
  int *piVar1;
  int iVar2;
  uint uVar3;
  ulonglong uVar4;
  ulonglong uVar5;
  ulonglong uVar6;
  undefined1 auStackY_288 [32];
  undefined4 local_258;
  undefined4 local_254;
  undefined4 local_250;
  undefined4 local_24c;
  undefined4 local_248 [2];
  Mat *local_240;
  undefined8 local_238;
  undefined4 local_230 [2];
  Mat *local_228;
  undefined8 local_220;
  undefined4 local_218 [2];
  Mat *local_210;
  undefined8 local_208;
  undefined4 local_200 [2];
  Mat *local_1f8;
  undefined8 local_1f0;
  undefined8 local_1e8;
  Mat local_1d8 [4];
  int local_1d4;
  undefined8 local_1c8;
  undefined8 local_1c0;
  undefined8 local_1b8;
  undefined8 local_1b0;
  longlong local_1a0;
  undefined4 *local_198;
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
  Mat local_118 [64];
  undefined4 *local_d8;
  Mat local_b8 [96];
  ulonglong local_58;
  
  local_1e8 = 0xfffffffffffffffe;
  local_58 = DAT_180008000 ^ (ulonglong)auStackY_288;
  if ((*(char *)(param_1 + 0x78) == '\0') || (*(int *)(param_1 + 0x80) == 1)) {
    *param_4 = param_2;
    if ((param_3 != 0) && (param_5 != (longlong *)0x0)) {
      *param_5 = param_3;
    }
  }
  else {
    uVar5 = 0;
    iVar2 = 2;
    if (*(int *)(param_1 + 200) < 9) {
      iVar2 = 0;
    }
    iVar2 = iVar2 + *(int *)(param_1 + 0xc4) * 8;
    memset(local_178,0,0x60);
    FUN_180001340((uint *)local_178,*(uint *)(param_1 + 0x2e4),*(uint *)(param_1 + 0x2e0),iVar2 - 8,
                  param_2,(longlong)*(int *)(param_1 + 0x2ec));
    memset(local_1d8,0,0x60);
    FUN_180001340((uint *)local_1d8,*(uint *)(param_1 + 0xb4),*(uint *)(param_1 + 0xb0),iVar2 - 8,
                  *param_4,(longlong)*(int *)(param_1 + 0xbc));
    local_208 = 0;
    local_218[0] = 0x2010000;
    local_210 = local_1d8;
    local_1f0 = 0;
    local_200[0] = 0x1010000;
    local_1f8 = local_178;
    local_254 = *local_198;
    local_258 = local_198[1];
    cv::resize(local_200,local_218,&local_258,0);
    if ((param_3 != 0) && (param_5 != (longlong *)0x0)) {
      memset(local_b8,0,0x60);
      FUN_180001340((uint *)local_b8,*(uint *)(param_1 + 0x2e4),*(uint *)(param_1 + 0x2e0),5,param_3
                    ,(longlong)*(int *)(param_1 + 0x2f0));
      memset(local_118,0,0x60);
      FUN_180001340((uint *)local_118,*(uint *)(param_1 + 0xb4),*(uint *)(param_1 + 0xb0),5,*param_5
                    ,(longlong)*(int *)(param_1 + 0xc0));
      local_238 = 0;
      local_248[0] = 0x2010000;
      local_240 = local_118;
      local_220 = 0;
      local_230[0] = 0x1010000;
      local_228 = local_b8;
      local_24c = *local_d8;
      local_250 = local_d8[1];
      cv::resize(local_230,local_248,&local_250,0);
      FUN_1800015d0(local_118);
      FUN_1800015d0(local_b8);
    }
    if (local_1a0 != 0) {
      LOCK();
      piVar1 = (int *)(local_1a0 + 0x14);
      iVar2 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar2 == 1) {
        cv::Mat::deallocate(local_1d8);
      }
    }
    local_1a0 = 0;
    local_1c8 = 0;
    local_1b0 = 0;
    local_1b8 = 0;
    local_1c0 = 0;
    uVar4 = uVar5;
    uVar6 = uVar5;
    if (0 < local_1d4) {
      do {
        *(undefined4 *)(uVar6 + (longlong)local_198) = 0;
        uVar3 = (int)uVar4 + 1;
        uVar4 = (ulonglong)uVar3;
        uVar6 = uVar6 + 4;
      } while ((int)uVar3 < local_1d4);
    }
    if (local_190 != local_188) {
      cv::fastFree(local_190);
    }
    if (local_140 != 0) {
      LOCK();
      piVar1 = (int *)(local_140 + 0x14);
      iVar2 = *piVar1;
      *piVar1 = *piVar1 + -1;
      UNLOCK();
      if (iVar2 == 1) {
        cv::Mat::deallocate(local_178);
      }
    }
    local_140 = 0;
    local_168 = 0;
    local_150 = 0;
    local_158 = 0;
    local_160 = 0;
    uVar4 = uVar5;
    if (0 < local_174) {
      do {
        *(undefined4 *)(uVar4 + local_138) = 0;
        uVar3 = (int)uVar5 + 1;
        uVar5 = (ulonglong)uVar3;
        uVar4 = uVar4 + 4;
      } while ((int)uVar3 < local_174);
    }
    if (local_130 != local_128) {
      cv::fastFree(local_130);
    }
  }
  return 0;
}


```
