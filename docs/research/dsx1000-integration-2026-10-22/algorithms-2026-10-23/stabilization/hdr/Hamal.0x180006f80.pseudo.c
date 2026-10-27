// ev_4f948285a3111b73ac30a5001a7a7084054037ce72aed771c45ba4a9c5990150

undefined8
FUN_180006f80(longlong param_1,longlong *param_2,longlong *param_3,uint *param_4,uint *param_5,
             uint *param_6,int *param_7)

{
  longlong lVar1;
  uint uVar2;
  undefined1 uVar3;
  undefined4 uVar4;
  int iVar5;
  void *pvVar6;
  undefined8 uVar7;
  undefined8 *puVar8;
  uint *puVar9;
  short *psVar10;
  undefined8 *puVar11;
  longlong lVar12;
  longlong lVar13;
  undefined8 *puVar14;
  
  if (DAT_180019a4d == '\0') {
    LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x1a6,L"true");
    DAT_180019a4d = '\x01';
  }
  puVar11 = (undefined8 *)0x0;
  if (*(void **)(param_1 + 0x18) != (void *)0x0) {
    operator_delete(*(void **)(param_1 + 0x18));
    *(undefined8 *)(param_1 + 0x18) = 0;
  }
  if (*(void **)(param_1 + 0x20) != (void *)0x0) {
    operator_delete(*(void **)(param_1 + 0x20));
    *(undefined8 *)(param_1 + 0x20) = 0;
  }
  if ((param_2 != (longlong *)0x0) && (param_3 != (longlong *)0x0)) {
    if (DAT_180019a4e == '\0') {
      LogTraceStdV(L"ONE TIME %s",L"[Cover]",L"CustomHalationRejectionCmd.cpp",0x1b4,L"true");
      DAT_180019a4e = '\x01';
    }
    pvVar6 = operator_new(0x28);
    *(void **)(param_1 + 0x18) = pvVar6;
    uVar4 = (**(code **)(*param_2 + 0x30))(param_2);
    **(undefined4 **)(param_1 + 0x18) = uVar4;
    uVar4 = (**(code **)(*param_2 + 0x38))(param_2);
    *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 4) = uVar4;
    uVar4 = (**(code **)(*param_2 + 0x48))(param_2);
    *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 8) = uVar4;
    *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0x10) = 1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0x1c) = 0;
    *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0x18) = 0;
    uVar3 = (**(code **)(*param_2 + 0x28))(param_2);
    *(undefined1 *)(param_1 + 0x38) = uVar3;
    uVar3 = (**(code **)(*param_2 + 0x50))(param_2);
    *(undefined1 *)(param_1 + 0x39) = uVar3;
    if (*(char *)(param_1 + 0x38) == '\x02') {
      *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0x14) = 0;
    }
    else if (*(char *)(param_1 + 0x38) == '\x04') {
      *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0x14) = 1;
    }
    else {
      *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0x14) = 0xffffffff;
    }
    iVar5 = (**(code **)(*param_2 + 0x50))(param_2);
    if (iVar5 == 8) {
      uVar4 = (**(code **)(*param_2 + 0x50))();
      *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0xc) = uVar4;
    }
    else {
      iVar5 = (**(code **)(*param_2 + 0x50))(param_2);
      if ((((iVar5 == 10) || (iVar5 = (**(code **)(*param_2 + 0x50))(param_2), iVar5 == 0xc)) ||
          (iVar5 = (**(code **)(*param_2 + 0x50))(param_2), iVar5 == 0xe)) ||
         (iVar5 = (**(code **)(*param_2 + 0x50))(param_2), iVar5 == 0x10)) {
        *(undefined4 *)(*(longlong *)(param_1 + 0x18) + 0xc) = 0xe;
      }
    }
    *(undefined8 *)(param_1 + 0x40) = 0;
    uVar7 = (**(code **)(*param_2 + 0x78))(param_2);
    *(undefined8 *)(param_1 + 0x48) = 0;
    *(longlong **)(param_1 + 0x50) = param_3;
    *(undefined8 *)(param_1 + 0x40) = uVar7;
    uVar7 = (**(code **)(*param_3 + 0x78))(param_3);
    *(undefined8 *)(param_1 + 0x48) = uVar7;
    if (*(void **)(param_1 + 0x30) != (void *)0x0) {
      operator_delete(*(void **)(param_1 + 0x30));
    }
    pvVar6 = operator_new(0x12dc);
    *(void **)(param_1 + 0x30) = pvVar6;
    memset(pvVar6,0,0x12dc);
    uVar4 = (**(code **)(*param_2 + 0x30))(param_2);
    **(undefined4 **)(param_1 + 0x30) = uVar4;
    uVar4 = (**(code **)(*param_2 + 0x38))(param_2);
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 4) = uVar4;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x14) = 1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x10) = 1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12d8) = 1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x18) = 0x7d1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1c) = 0x83;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x20) = 0x12;
    puVar8 = operator_new(0x880);
    puVar14 = puVar11;
    if (puVar8 != (undefined8 *)0x0) {
      *puVar8 = CCrdcApicDRCorrection::vftable;
      puVar8[0xf] = 0;
      *(undefined4 *)(puVar8 + 2) = 0x400;
      *(undefined4 *)((longlong)puVar8 + 0x14) = 0x800;
      *(undefined4 *)(puVar8 + 3) = 0xc00;
      *(undefined4 *)((longlong)puVar8 + 0x1c) = 0x1000;
      *(undefined4 *)(puVar8 + 4) = 0x1400;
      *(undefined4 *)((longlong)puVar8 + 0x24) = 0x1800;
      *(undefined4 *)(puVar8 + 5) = 0x1c00;
      *(undefined4 *)((longlong)puVar8 + 0x2c) = 0x2000;
      puVar8[6] = 0x20000000200;
      puVar8[7] = 0x20000000200;
      puVar8[8] = 0x20000000200;
      puVar8[9] = 0x20000000200;
      puVar14 = puVar8;
    }
    if ((param_6 == (uint *)0x0) || (uVar2 = *param_6, uVar2 == 0xffffffff)) {
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 8) = 0x14;
    }
    else if (uVar2 < 0x100) {
      *(uint *)(*(longlong *)(param_1 + 0x30) + 8) = uVar2;
    }
    else {
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 8) = 0xffffffff;
    }
    if ((param_5 == (uint *)0x0) || (uVar2 = *param_5, uVar2 == 0xffffffff)) {
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0xc) = 0x40;
    }
    else if (uVar2 < 0x100) {
      *(uint *)(*(longlong *)(param_1 + 0x30) + 0xc) = uVar2;
    }
    else {
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0xc) = 0xffffffff;
    }
    if ((param_4 == (uint *)0x0) || (uVar2 = *param_4, uVar2 == 0xffffffff)) {
      *(undefined4 *)(param_1 + 0x3c) = 0x32;
    }
    else if ((uVar2 < 0x65) && (uVar2 == ((int)uVar2 / 10) * 10)) {
      *(uint *)(param_1 + 0x3c) = uVar2;
    }
    else {
      *(undefined4 *)(param_1 + 0x3c) = 0xffffffff;
    }
    lVar12 = 0x1164;
    lVar13 = 0xf;
    do {
      if (*(int *)(param_1 + 0x3c) == -1) {
        *(undefined4 *)(*(longlong *)(param_1 + 0x30) + -0x3c + lVar12) = 0xffffffff;
        *(undefined4 *)(lVar12 + *(longlong *)(param_1 + 0x30)) = 0xffffffff;
        *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x3c + lVar12) = 0xffffffff;
      }
      else {
        iVar5 = (**(code **)(*param_2 + 0x50))(param_2);
        if (iVar5 == 8) {
          *(undefined4 *)(*(longlong *)(param_1 + 0x30) + -0x3c + lVar12) =
               *(undefined4 *)
                (&DAT_180011bc0 +
                ((longlong)(*(int *)(param_1 + 0x3c) / 10) * 0xf + (longlong)puVar11) * 4);
          *(undefined4 *)(lVar12 + *(longlong *)(param_1 + 0x30)) =
               *(undefined4 *)
                (&DAT_180011680 +
                ((longlong)(*(int *)(param_1 + 0x3c) / 10) * 0xf + (longlong)puVar11) * 4);
          uVar4 = *(undefined4 *)
                   (&DAT_180011920 +
                   ((longlong)(*(int *)(param_1 + 0x3c) / 10) * 0xf + (longlong)puVar11) * 4);
        }
        else {
          *(undefined4 *)(*(longlong *)(param_1 + 0x30) + -0x3c + lVar12) =
               *(undefined4 *)
                (&DAT_1800113e0 +
                ((longlong)(*(int *)(param_1 + 0x3c) / 10) * 0xf + (longlong)puVar11) * 4);
          *(undefined4 *)(lVar12 + *(longlong *)(param_1 + 0x30)) =
               *(undefined4 *)
                (&DAT_180010ea0 +
                ((longlong)(*(int *)(param_1 + 0x3c) / 10) * 0xf + (longlong)puVar11) * 4);
          uVar4 = *(undefined4 *)
                   (&DAT_180011140 +
                   ((longlong)(*(int *)(param_1 + 0x3c) / 10) * 0xf + (longlong)puVar11) * 4);
        }
        *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x3c + lVar12) = uVar4;
      }
      puVar11 = (undefined8 *)((longlong)puVar11 + 1);
      lVar12 = lVar12 + 4;
      lVar13 = lVar13 + -1;
    } while (lVar13 != 0);
    if ((((param_7 == (int *)0x0) || (puVar9 = (uint *)(param_7 + 0x10), param_7[0x10] == -1)) ||
        (*param_7 == -1)) || (param_7[8] == -1)) {
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11dc) = 0x19;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11fc) = 0xa3;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x121c) = 0x1a;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11e0) = 0x14;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1200) = 0xcc;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1220) = 0x21;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11e4) = 0xf;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1204) = 0x109;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1224) = 0x2a;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11e8) = 0xc;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1208) = 0x164;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1228) = 0x39;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11ec) = 8;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x120c) = 0x1f9;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x122c) = 0x51;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11f0) = 5;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1210) = 0x304;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1230) = 0x7b;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11f4) = 3;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1214) = 0x52e;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1234) = 0xd4;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x11f8) = 1;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1218) = 0xafb;
      *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1238) = 0x1c2;
    }
    else {
      lVar12 = 8;
      do {
        if (((*puVar9 < 0x400) && (puVar9[-0x10] < 0x4000)) && (puVar9[-8] < 0x2000)) {
          *(uint *)(*(longlong *)(param_1 + 0x30) + (0x119c - (longlong)param_7) + (longlong)puVar9)
               = puVar9[-0x10];
          *(uint *)(*(longlong *)(param_1 + 0x30) + (0x11bc - (longlong)param_7) + (longlong)puVar9)
               = puVar9[-8];
          *(uint *)(*(longlong *)(param_1 + 0x30) + (0x11dc - (longlong)param_7) + (longlong)puVar9)
               = *puVar9;
        }
        else {
          *(undefined4 *)
           (*(longlong *)(param_1 + 0x30) + (0x119c - (longlong)param_7) + (longlong)puVar9) =
               0xffffffff;
          *(undefined4 *)
           (*(longlong *)(param_1 + 0x30) + (0x11bc - (longlong)param_7) + (longlong)puVar9) =
               0xffffffff;
          *(undefined4 *)
           (*(longlong *)(param_1 + 0x30) + (0x11dc - (longlong)param_7) + (longlong)puVar9) =
               0xffffffff;
        }
        puVar9 = puVar9 + 1;
        lVar12 = lVar12 + -1;
      } while (lVar12 != 0);
    }
    if (puVar14 != (undefined8 *)0x0) {
      (**(code **)*puVar14)(puVar14,1);
    }
    psVar10 = &DAT_180010602;
    lVar13 = 0x16b;
    lVar12 = 0x28;
    do {
      lVar1 = lVar12 + 0xc;
      *(int *)(*(longlong *)(param_1 + 0x30) + -0x10 + lVar1) = (int)psVar10[-1];
      *(int *)(lVar12 + *(longlong *)(param_1 + 0x30)) = (int)*psVar10;
      *(int *)(*(longlong *)(param_1 + 0x30) + -8 + lVar1) = (int)psVar10[1];
      psVar10 = psVar10 + 3;
      lVar13 = lVar13 + -1;
      lVar12 = lVar1;
    } while (lVar13 != 0);
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x123c) = 0;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1240) = 0x6db;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1244) = 0xdb6;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1248) = 0x1492;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x124c) = 0x1b6d;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1250) = 0x2249;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1254) = 0x2924;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1258) = 0x3c;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x125c) = 0x2b;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1260) = 0x18;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1264) = 0xc;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1268) = 5;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x126c) = 2;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1270) = 1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1274) = 0;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1278) = 0x677;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x127c) = 0xb1d;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1280) = 0xdba;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1284) = 0xefe;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1288) = 0xf90;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x128c) = 0xfd0;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1290) = 0xda;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1294) = 0x2dc;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x1298) = 0x4a;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x129c) = 0xffffff8b;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12a0) = 0xfffffe75;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12a4) = 0x200;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12a8) = 0x200;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12ac) = 0xfffffe2f;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12b0) = 0xffffffd1;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12b4) = 0x400;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12b8) = 0;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12bc) = 0x64d;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12c0) = 0x400;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12c4) = 0xffffff41;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12c8) = 0xfffffe21;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12cc) = 0x400;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12d0) = 0x76d;
    *(undefined4 *)(*(longlong *)(param_1 + 0x30) + 0x12d4) = 0;
  }
  LogTraceV(L"CCustomHalationRejectionCmd::SetParam(%p)\n",param_2);
  return 0;
}

