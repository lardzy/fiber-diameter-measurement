// ev_872e226af9694ea5483ba8dc41e37cad0ba70928fc0561737b850b8e86bc10e1

/* WARNING: Function: __security_check_cookie replaced with injection: security_check_cookie */

void FUN_180002a70(longlong param_1,undefined8 param_2,undefined8 param_3)

{
  longlong lVar1;
  uint uVar2;
  ulonglong uVar3;
  uint uVar4;
  char cVar5;
  uint uVar6;
  uint uVar7;
  int iVar8;
  char cVar9;
  char cVar10;
  undefined1 auStack_88 [32];
  undefined4 *local_68;
  undefined4 *local_60;
  undefined8 local_58;
  undefined8 local_50;
  undefined4 local_48;
  undefined4 local_44;
  undefined4 local_40;
  undefined8 local_3c;
  undefined8 local_34;
  undefined8 local_2c;
  int local_24;
  ulonglong local_20;
  
  local_20 = DAT_180019000 ^ (ulonglong)auStack_88;
  cVar5 = '\0';
  local_40 = 0;
  local_3c = 0;
  local_34 = 0;
  local_2c = 0;
  local_44 = 0x12c0;
  local_48 = 0x1fff;
  local_24 = 0;
  local_60 = &local_44;
  local_68 = &local_48;
  local_58 = param_2;
  local_50 = param_3;
  _vcomp_fork(1,5,FUN_18000c2d0,&local_40);
  if (local_24 == 0) {
LAB_180002b55:
    *(undefined4 *)(param_1 + 0x28) = 0x1c00;
  }
  else if (local_24 < local_2c._4_4_ * 10) {
    if (local_24 < local_2c._4_4_ * 5) {
      if (local_24 * 3 < local_2c._4_4_ * 10) {
        if (local_24 * 2 < local_2c._4_4_ * 5) {
          if (local_24 < local_2c._4_4_ * 2) goto LAB_180002b55;
          *(undefined4 *)(param_1 + 0x28) = 0x1be0;
        }
        else {
          *(undefined4 *)(param_1 + 0x28) = 0x1ba0;
        }
      }
      else {
        *(undefined4 *)(param_1 + 0x28) = 0x1b60;
      }
    }
    else {
      *(undefined4 *)(param_1 + 0x28) = 0x1b20;
    }
  }
  else {
    *(undefined4 *)(param_1 + 0x28) = 0x1b00;
  }
  if (local_2c._4_4_ == 0) {
LAB_180002bbd:
    *(undefined4 *)(param_1 + 0x24) = 0x1800;
  }
  else if (local_2c._4_4_ < (int)local_2c * 10) {
    if (local_2c._4_4_ < (int)local_2c * 5) {
      if (local_2c._4_4_ * 3 < (int)local_2c * 10) {
        if (local_2c._4_4_ * 2 < (int)local_2c * 5) {
          if (local_2c._4_4_ < (int)local_2c * 2) goto LAB_180002bbd;
          *(undefined4 *)(param_1 + 0x24) = 0x17f0;
        }
        else {
          *(undefined4 *)(param_1 + 0x24) = 0x17d0;
        }
      }
      else {
        *(undefined4 *)(param_1 + 0x24) = 0x17b0;
      }
    }
    else {
      *(undefined4 *)(param_1 + 0x24) = 0x1790;
    }
  }
  else {
    *(undefined4 *)(param_1 + 0x24) = 0x1780;
  }
  if ((int)local_2c != 0) {
    if (local_34._4_4_ * 10 <= (int)local_2c) {
      *(undefined4 *)(param_1 + 0x20) = 0x13c0;
      goto LAB_180002c2c;
    }
    if (local_34._4_4_ * 5 <= (int)local_2c) {
      *(undefined4 *)(param_1 + 0x20) = 0x13c8;
      goto LAB_180002c2c;
    }
    if (local_34._4_4_ * 10 <= (int)local_2c * 3) {
      *(undefined4 *)(param_1 + 0x20) = 0x13d8;
      goto LAB_180002c2c;
    }
    if (local_34._4_4_ * 5 <= (int)local_2c * 2) {
      *(undefined4 *)(param_1 + 0x20) = 0x13e8;
      goto LAB_180002c2c;
    }
    if (local_34._4_4_ * 2 <= (int)local_2c) {
      *(undefined4 *)(param_1 + 0x20) = 0x13f8;
      goto LAB_180002c2c;
    }
  }
  *(undefined4 *)(param_1 + 0x20) = 0x1400;
LAB_180002c2c:
  uVar6 = 0x2000;
  cVar9 = '\0';
  uVar4 = 0x2000 - *(int *)(param_1 + 0x28);
  uVar7 = uVar4;
  if (uVar4 < 0x2000) {
    uVar2 = 0x2000;
    cVar9 = '\0';
    do {
      uVar2 = uVar2 >> 1 | (uint)((uVar2 & 1) != 0) << 0x1f;
      uVar7 = uVar7 * 2;
      cVar9 = cVar9 + '\x01';
    } while (uVar4 < uVar2);
  }
  uVar4 = *(int *)(param_1 + 0x28) - *(int *)(param_1 + 0x24);
  iVar8 = uVar7 - (uVar7 & 0xffffffc0);
  lVar1 = (ulonglong)((uVar7 & 0xffffffc0) - 0x2000 >> 6) * 4;
  cVar10 = '\0';
  *(int *)(param_1 + 0x4c) =
       (int)(((ulonglong)
              (uint)(*(int *)(&DAT_180014804 + lVar1) * iVar8 +
                    *(int *)(&DAT_180014800 + lVar1) * (0x40 - iVar8)) << 0x13) >>
            (0x26U - cVar9 & 0x3f));
  uVar7 = uVar4;
  if (uVar4 < 0x2000) {
    uVar2 = 0x2000;
    cVar10 = '\0';
    do {
      uVar2 = uVar2 >> 1 | (uint)((uVar2 & 1) != 0) << 0x1f;
      uVar7 = uVar7 * 2;
      cVar10 = cVar10 + '\x01';
    } while (uVar4 < uVar2);
  }
  uVar4 = *(int *)(param_1 + 0x24) - *(int *)(param_1 + 0x20);
  iVar8 = uVar7 - (uVar7 & 0xffffffc0);
  lVar1 = (ulonglong)((uVar7 & 0xffffffc0) - 0x2000 >> 6) * 4;
  cVar9 = '\0';
  *(int *)(param_1 + 0x48) =
       (int)(((ulonglong)
              (uint)(*(int *)(&DAT_180014800 + lVar1) * (0x40 - iVar8) +
                    *(int *)(&DAT_180014804 + lVar1) * iVar8) << 0x13) >> (0x26U - cVar10 & 0x3f));
  uVar7 = uVar4;
  if (uVar4 < 0x2000) {
    uVar2 = 0x2000;
    cVar9 = '\0';
    do {
      uVar2 = uVar2 >> 1 | (uint)((uVar2 & 1) != 0) << 0x1f;
      uVar7 = uVar7 * 2;
      cVar9 = cVar9 + '\x01';
    } while (uVar4 < uVar2);
  }
  uVar4 = *(int *)(param_1 + 0x20) - *(int *)(param_1 + 0x1c);
  iVar8 = uVar7 - (uVar7 & 0xffffffc0);
  lVar1 = (ulonglong)((uVar7 & 0xffffffc0) - 0x2000 >> 6) * 4;
  *(int *)(param_1 + 0x44) =
       (int)(((ulonglong)
              (uint)(*(int *)(&DAT_180014800 + lVar1) * (0x40 - iVar8) +
                    *(int *)(&DAT_180014804 + lVar1) * iVar8) << 0x13) >> (0x26U - cVar9 & 0x3f));
  uVar7 = uVar4;
  cVar9 = '\0';
  if (uVar4 < 0x2000) {
    do {
      uVar6 = uVar6 >> 1 | (uint)((uVar6 & 1) != 0) << 0x1f;
      uVar7 = uVar7 * 2;
      cVar5 = cVar9 + '\x01';
      cVar9 = cVar5;
    } while (uVar4 < uVar6);
  }
  iVar8 = uVar7 - (uVar7 & 0xffffffc0);
  uVar3 = (ulonglong)((uVar7 & 0xffffffc0) - 0x2000 >> 6);
  *(int *)(param_1 + 0x40) =
       (int)(((ulonglong)
              (uint)(*(int *)(&DAT_180014804 + uVar3 * 4) * iVar8 +
                    *(int *)(&DAT_180014800 + uVar3 * 4) * (0x40 - iVar8)) << 0x13) >>
            (0x26U - cVar5 & 0x3f));
  return;
}

