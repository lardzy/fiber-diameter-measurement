// ev_310fdb65f3b673c33d21caa2ab9189ec1546b7243617bc26be3b8419e61a7798

undefined8 FUN_1800065f0(undefined8 *param_1)

{
  int *piVar1;
  undefined4 *puVar2;
  undefined1 auVar3 [16];
  int iVar4;
  undefined8 *puVar5;
  wchar_t *pwVar6;
  undefined8 uVar7;
  longlong lVar8;
  longlong *plVar9;
  undefined8 *puVar10;
  undefined8 *puVar11;
  int iVar12;
  uint uVar13;
  ulonglong uVar14;
  int iVar15;
  char cVar16;
  int iVar17;
  char local_res8 [8];
  int local_res10 [2];
  int local_res18 [2];
  int local_res20 [2];
  int local_88 [2];
  longlong *local_80;
  LARGE_INTEGER local_78;
  LARGE_INTEGER local_70;
  LARGE_INTEGER local_68;
  undefined8 local_60;
  undefined8 *local_58;
  ulonglong local_50;
  int iVar18;
  
  local_60 = 0xfffffffffffffffe;
  local_res8[0] = '\0';
  if (param_1[6] == 0) {
    if (DAT_180019a40 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x59,L"true");
      DAT_180019a40 = '\x01';
    }
    pwVar6 = L"CCustomHalationRejectionCmd::Execute()  ApicalParam is NULL or invalid";
LAB_180006f51:
    OutputDebugLog(pwVar6);
    return 0x80070057;
  }
  piVar1 = (int *)param_1[3];
  if (piVar1 == (int *)0x0) {
    if (DAT_180019a41 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x5f,L"true");
      DAT_180019a41 = '\x01';
    }
    pwVar6 = L"CCustomHalationRejectionCmd::Execute()  InputImage is NULL or invalid";
    goto LAB_180006f51;
  }
  if (piVar1[5] == -1) {
    if (DAT_180019a42 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x65,L"true");
      DAT_180019a42 = '\x01';
    }
    pwVar6 = L"CCustomHalationRejectionCmd::Execute()  ImageType of InputImage is invalid";
    goto LAB_180006f51;
  }
  if (0x2697 < *piVar1 - 0x79U) {
    if (DAT_180019a43 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x6b,L"true");
      DAT_180019a43 = '\x01';
    }
    pwVar6 = L"CCustomHalationRejectionCmd::Execute()  Width of InputImage is NULL or invalid";
    goto LAB_180006f51;
  }
  if (0x26b5 < piVar1[1] - 0x5bU) {
    if (DAT_180019a44 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x71,L"true");
      DAT_180019a44 = '\x01';
    }
    pwVar6 = L"CCustomHalationRejectionCmd::Execute()  Height of InputImage is NULL or invalid";
    goto LAB_180006f51;
  }
  local_res8[0] = piVar1[5] == 1;
  local_res18[0] = 0xe - (uint)*(byte *)((longlong)param_1 + 0x39);
  puVar10 = (undefined8 *)0x0;
  if (*(byte *)((longlong)param_1 + 0x39) == 8) {
    local_res18[0] = 0;
  }
  local_88[0] = piVar1[2] * piVar1[1] * *piVar1;
  auVar3._8_8_ = 0;
  auVar3._0_8_ = (longlong)local_88[0] * 2;
  lVar8 = SUB168(ZEXT816(0x4a61d1491245f085) * auVar3,8);
  iVar18 = (int)(((ulonglong)((longlong)local_88[0] * 2 - lVar8) >> 1) + lVar8 >> 0x17);
  iVar17 = iVar18 + 1;
  if (iVar17 == 1) {
    iVar15 = piVar1[1];
  }
  else {
    iVar15 = piVar1[1] / iVar17 + 1;
  }
  local_res10[0] = iVar15 * piVar1[2] * *piVar1;
  if (param_1[2] != 0) {
    if (DAT_180019a45 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x93,L"true");
      DAT_180019a45 = '\x01';
    }
    (*(code *)**(undefined8 **)param_1[2])((undefined8 *)param_1[2],0,iVar18 + 3);
  }
  if ((*(int *)(param_1[3] + 0xc) == 0xe) || (local_res8[0] == '\x01')) {
    *(undefined8 *)(param_1[3] + 0x20) = param_1[9];
  }
  _vcomp_fork(1,4,FUN_18000d350,local_88,local_res18,local_res8,param_1);
  if (local_res8[0] == '\x01') {
    if (DAT_180019a46 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0xb8,L"true");
      DAT_180019a46 = '\x01';
    }
    puVar2 = (undefined4 *)param_1[3];
    FUN_18000a220(*(undefined8 *)(puVar2 + 8),*puVar2,puVar2[1],(byte)puVar2[3]);
  }
  uVar7 = 0x880;
  puVar5 = operator_new(0x880);
  puVar11 = puVar10;
  if (puVar5 != (undefined8 *)0x0) {
    *puVar5 = CCrdcApicDRCorrection::vftable;
    puVar5[0xf] = 0;
    *(undefined4 *)(puVar5 + 2) = 0x400;
    *(undefined4 *)((longlong)puVar5 + 0x14) = 0x800;
    *(undefined4 *)(puVar5 + 3) = 0xc00;
    *(undefined4 *)((longlong)puVar5 + 0x1c) = 0x1000;
    *(undefined4 *)(puVar5 + 4) = 0x1400;
    *(undefined4 *)((longlong)puVar5 + 0x24) = 0x1800;
    *(undefined4 *)(puVar5 + 5) = 0x1c00;
    *(undefined4 *)((longlong)puVar5 + 0x2c) = 0x2000;
    puVar5[6] = 0x20000000200;
    puVar5[7] = 0x20000000200;
    puVar5[8] = 0x20000000200;
    puVar5[9] = 0x20000000200;
    puVar11 = puVar5;
  }
  local_58 = (undefined8 *)0x0;
  local_50 = 0;
  local_58 = (undefined8 *)FUN_180008230(uVar7,(undefined8 *)0x0,(undefined8 *)0x0);
  puVar11[0xf] = &local_58;
  local_68.QuadPart = 0;
  local_70.QuadPart = 0;
  local_78.QuadPart = 0;
  QueryPerformanceFrequency(&local_68);
  QueryPerformanceCounter(&local_70);
  lVar8 = param_1[3];
  if (*(int *)(lVar8 + 0xc) == 8) {
    if (local_res8[0] == '\0') {
      uVar7 = FUN_180004130((longlong)puVar11,param_1[8],(int *)param_1[6]);
      cVar16 = (char)uVar7;
      goto LAB_1800069ae;
    }
    if (local_res8[0] == '\x01') {
      uVar7 = FUN_180004060((longlong)puVar11,*(longlong *)(lVar8 + 0x20),(int *)param_1[6]);
      cVar16 = (char)uVar7;
      goto LAB_1800069ae;
    }
  }
  uVar7 = FUN_180005210((longlong)puVar11,*(longlong *)(lVar8 + 0x20),(int *)param_1[6]);
  cVar16 = (char)uVar7;
LAB_1800069ae:
  if (cVar16 == '\0') {
    if (DAT_180019a47 == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0xd7,L"true");
      DAT_180019a47 = '\x01';
    }
    puVar10 = (undefined8 *)*local_58;
    if (0 < (int)local_50) {
      uVar14 = local_50 & 0xffffffff;
      do {
        plVar9 = puVar10 + 2;
        if (7 < (ulonglong)puVar10[5]) {
          plVar9 = (longlong *)*plVar9;
        }
        LogTraceV(&DAT_180014d80,plVar9);
        puVar10 = (undefined8 *)*puVar10;
        uVar14 = uVar14 - 1;
      } while (uVar14 != 0);
    }
    (**(code **)*puVar11)(puVar11,1);
    uVar7 = 0x80070057;
  }
  else {
    puVar5 = operator_new(0x28);
    param_1[4] = puVar5;
    *puVar5 = 0;
    puVar5[1] = 0;
    puVar5[2] = 0;
    puVar5[3] = 0;
    puVar5[4] = 0;
    *(undefined8 *)(param_1[4] + 0x20) = param_1[9];
    *(undefined4 *)param_1[4] = *(undefined4 *)param_1[3];
    *(undefined4 *)(param_1[4] + 4) = *(undefined4 *)(param_1[3] + 4);
    *(undefined4 *)(param_1[4] + 8) = *(undefined4 *)(param_1[3] + 8);
    *(undefined4 *)(param_1[4] + 0xc) = *(undefined4 *)(param_1[3] + 0xc);
    *(undefined4 *)(param_1[4] + 0x10) = *(undefined4 *)(param_1[3] + 0x10);
    *(undefined4 *)(param_1[4] + 0x14) = *(undefined4 *)(param_1[3] + 0x14);
    *(undefined4 *)(param_1[4] + 0x18) = *(undefined4 *)(param_1[3] + 0x18);
    *(undefined4 *)(param_1[4] + 0x1c) = *(undefined4 *)(param_1[3] + 0x1c);
    if (param_1[2] != 0) {
      if (DAT_180019a48 == '\0') {
        LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0xf3,L"true");
        DAT_180019a48 = '\x01';
      }
      (*(code *)**(undefined8 **)param_1[2])((undefined8 *)param_1[2],1,iVar18 + 3);
    }
    if (0 < iVar17) {
      do {
        local_80 = (longlong *)0x0;
        FUN_180006080(&local_80);
        iVar4 = (**(code **)(*local_80 + 0x10))(local_80,(longlong)local_res10[0]);
        iVar12 = (int)puVar10;
        if (iVar4 != 0) {
          if (local_80 != (longlong *)0x0) {
            (**(code **)(*local_80 + 8))();
          }
          uVar7 = 0;
          goto LAB_180006e5c;
        }
        uVar7 = (**(code **)(*local_80 + 0x28))();
        *(undefined8 *)(param_1[3] + 0x20) = uVar7;
        if (iVar12 < iVar18) {
          local_res20[0] = iVar12 * local_res10[0];
        }
        else {
          local_res20[0] = local_88[0] - local_res10[0];
        }
        _vcomp_fork(1,5,FUN_18000d440,local_res20,local_res10,local_res18,local_res8,param_1);
        if (local_res8[0] == '\x01') {
          if (DAT_180019a49 == '\0') {
            LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x130,L"true");
            DAT_180019a49 = '\x01';
          }
          puVar2 = (undefined4 *)param_1[3];
          FUN_18000a220(*(undefined8 *)(puVar2 + 8),*puVar2,iVar15,(byte)puVar2[3]);
        }
        uVar13 = iVar12 + 1;
        puVar10 = (undefined8 *)(ulonglong)uVar13;
        lVar8 = *(longlong *)(param_1[3] + 0x20);
        if (*(int *)(param_1[3] + 0xc) == 8) {
          uVar7 = FUN_180004200((longlong)puVar11,lVar8,(uint *)param_1[6],iVar17,uVar13,iVar15);
          cVar16 = (char)uVar7;
        }
        else {
          uVar7 = FUN_1800052e0((longlong)puVar11,lVar8,(uint *)param_1[6],iVar17,uVar13,iVar15);
          cVar16 = (char)uVar7;
        }
        if (local_res8[0] == '\x01') {
          if (DAT_180019a4a == '\0') {
            LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x140,L"true");
            DAT_180019a4a = '\x01';
          }
          FUN_18000a1e0(*(undefined8 *)((undefined4 *)param_1[3] + 8),*(undefined4 *)param_1[3],
                        iVar15);
        }
        _vcomp_fork(1,5,FUN_18000d560,local_res20,local_res10,local_res18,local_res8,param_1);
        plVar9 = local_80;
        if (local_80 != (longlong *)0x0) {
          local_80 = (longlong *)0x0;
          (**(code **)(*plVar9 + 8))();
        }
        *(undefined8 *)(param_1[3] + 0x20) = 0;
        if (param_1[2] != 0) {
          if (DAT_180019a4b == '\0') {
            LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x15c,L"true");
            DAT_180019a4b = '\x01';
          }
          (*(code *)**(undefined8 **)param_1[2])((undefined8 *)param_1[2],iVar12 + 2,iVar18 + 3);
        }
        if (local_80 != (longlong *)0x0) {
          (**(code **)(*local_80 + 8))();
        }
      } while ((int)uVar13 < iVar17);
    }
    QueryPerformanceCounter(&local_78);
    LogTraceV(L"CCustomHalationRejectionCmd::time(%d )\n",
              ((local_78.QuadPart - (longlong)local_70) * 1000) / local_68.QuadPart & 0xffffffff);
    (**(code **)*puVar11)(puVar11,1);
    if (cVar16 == '\0') {
      if (DAT_180019a4c == '\0') {
        LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x16d,L"true");
        DAT_180019a4c = '\x01';
      }
      puVar10 = (undefined8 *)*local_58;
      if (0 < (int)local_50) {
        uVar14 = local_50 & 0xffffffff;
        do {
          plVar9 = puVar10 + 2;
          if (7 < (ulonglong)puVar10[5]) {
            plVar9 = (longlong *)*plVar9;
          }
          LogTraceV(&DAT_180014d80,plVar9);
          puVar10 = (undefined8 *)*puVar10;
          uVar14 = uVar14 - 1;
        } while (uVar14 != 0);
      }
      uVar7 = 0x80070057;
    }
    else {
      OutputDebugLog(L"CCustomHalationRejectionCmd::Execute()\n");
      puVar10 = (undefined8 *)param_1[2];
      if (puVar10 != (undefined8 *)0x0) {
        (**(code **)*puVar10)(puVar10,iVar18 + 3,iVar18 + 3);
        (**(code **)*param_1)(param_1);
        (**(code **)(*(longlong *)param_1[2] + 8))((longlong *)param_1[2],0,param_1);
      }
      uVar7 = 0;
    }
  }
LAB_180006e5c:
  puVar10 = (undefined8 *)*local_58;
  *local_58 = local_58;
  local_58[1] = local_58;
  local_50 = 0;
  if (puVar10 != local_58) {
    do {
      puVar11 = (undefined8 *)*puVar10;
      if (7 < (ulonglong)puVar10[5]) {
        operator_delete((void *)puVar10[2]);
      }
      puVar10[5] = 7;
      puVar10[4] = 0;
      *(undefined2 *)(puVar10 + 2) = 0;
      operator_delete(puVar10);
      puVar10 = puVar11;
    } while (puVar11 != local_58);
  }
  operator_delete(local_58);
  return uVar7;
}

