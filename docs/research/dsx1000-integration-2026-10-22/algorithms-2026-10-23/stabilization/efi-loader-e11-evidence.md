## read_bytes {'address': '0x18007c4fc', 'length': 16}

Evidence: `ev_1fe78b560e8961129eaf7041dcb0894bb8083acc1535aac71b27070aac977793`

```text
{
  "address": "0x18007c4fc",
  "bytes_hex": "d578e93d8716993e0000003fa245163f",
  "complete": true,
  "requested_bytes": 16,
  "returned_bytes": 16
}
```

## procedure_assembly {'procedure': '0x18002c2f0'}

Evidence: `ev_2277be1da764d865d0fd1581ec71dc960d70e1313c74bd5ebbc7a34ec15b1d6e`

```text
0x18002c2f0: TEST EDX, EDX
0x18002c2f2: JLE 0x18002c78e
0x18002c2f8: MOV RAX, RSP
0x18002c2fb: MOV dword ptr [RAX + 0x10], EDX
0x18002c2fe: PUSH RBP
0x18002c2ff: PUSH R12
0x18002c301: PUSH R13
0x18002c303: PUSH R14
0x18002c305: PUSH R15
0x18002c307: MOV RBP, RSP
0x18002c30a: SUB RSP, 0x70
0x18002c30e: MOV qword ptr [RAX + 0x8], RBX
0x18002c312: MOV qword ptr [RAX + 0x18], RSI
0x18002c316: MOV RSI, qword ptr [RBP + 0x50]
0x18002c31a: MOV qword ptr [RAX + 0x20], RDI
0x18002c31e: MOVSXD R13, ECX
0x18002c321: XOR R12D, R12D
0x18002c324: MOV R14, R9
0x18002c327: MOV R15, R8
0x18002c32a: MOV R11D, EDX
0x18002c32d: MOVAPS xmmword ptr [RAX + -0x38], XMM6
0x18002c331: MOVSS XMM6, dword ptr [0x18007c500]
0x18002c339: MOVAPS xmmword ptr [RAX + -0x48], XMM7
0x18002c33d: MOVSS XMM7, dword ptr [0x18007c508]
0x18002c345: MOVAPS xmmword ptr [RAX + -0x58], XMM8
0x18002c34a: MOVSS XMM8, dword ptr [0x18007c4fc]
0x18002c353: MOV qword ptr [RBP + -0x50], R13
0x18002c357: XOR EDI, EDI
0x18002c359: NOP dword ptr [RAX]
0x18002c360: XOR EBX, EBX
0x18002c362: CMP R13, 0x4
0x18002c366: JL 0x18002c66e
0x18002c36c: LEA RAX, [R13 + -0x3]
0x18002c370: XOR R9D, R9D
0x18002c373: XOR R13D, R13D
0x18002c376: MOV qword ptr [RBP + -0x58], RAX
0x18002c37a: MOV qword ptr [RBP + -0x60], R9
0x18002c37e: NOP
0x18002c380: MOV RCX, qword ptr [R15]
0x18002c383: MOV R8, qword ptr [R14]
0x18002c386: MOVSXD RDX, dword ptr [R15 + 0x8]
0x18002c38a: IMUL RDX, RDI
0x18002c38e: ADD RDX, R13
0x18002c391: LEA R11, [RBX*0x4]
0x18002c399: MOV EAX, dword ptr [RDX + RCX*0x1]
0x18002c39c: MOV dword ptr [RBP + -0x70], EAX
0x18002c39f: MOVZX EAX, AX
0x18002c3a2: MOVD XMM4, EAX
0x18002c3a6: MOVZX EAX, word ptr [RBP + -0x6e]
0x18002c3aa: MOVD XMM3, EAX
0x18002c3ae: MOVZX EAX, word ptr [RDX + RCX*0x1 + 0x4]
0x18002c3b3: MOVSXD RDX, dword ptr [R14 + 0x8]
0x18002c3b7: CVTDQ2PS XMM4, XMM4
0x18002c3ba: MOVD XMM5, EAX
0x18002c3be: MOV EAX, EDX
0x18002c3c0: MOVAPS XMM2, XMM4
0x18002c3c3: CVTDQ2PS XMM3, XMM3
0x18002c3c6: MULSS XMM4, XMM8
0x18002c3cb: IMUL EAX, R12D
0x18002c3cf: MOVSXD RCX, EAX
0x18002c3d2: ADD RCX, R9
0x18002c3d5: MOVAPS XMM1, XMM3
0x18002c3d8: CVTDQ2PS XMM5, XMM5
0x18002c3db: ADDSS XMM2, dword ptr [RCX + R8*0x1]
0x18002c3e1: ADDSS XMM1, dword ptr [RCX + R8*0x1 + 0x4]
0x18002c3e8: MOVAPS XMM0, XMM5
0x18002c3eb: MULSS XMM3, XMM7
0x18002c3ef: MULSS XMM5, XMM6
0x18002c3f3: ADDSS XMM0, dword ptr [RCX + R8*0x1 + 0x8]
0x18002c3fa: ADDSS XMM5, XMM3
0x18002c3fe: MOV RCX, RDX
0x18002c401: IMUL RCX, RDI
0x18002c405: ADD RCX, R9
0x18002c408: MOVSS dword ptr [RBP + -0x40], XMM0
0x18002c40d: ADDSS XMM5, XMM4
0x18002c411: MOV EAX, dword ptr [RBP + -0x40]
0x18002c414: MOVAPS XMM0, XMM2
0x18002c417: UNPCKLPS XMM0, XMM1
0x18002c41a: MOVSD qword ptr [RCX + R8*0x1], XMM0
0x18002c420: MOV dword ptr [RCX + R8*0x1 + 0x8], EAX
0x18002c425: MOVSXD RCX, dword ptr [RSI + 0x8]
0x18002c429: MOV RAX, qword ptr [RSI]
0x18002c42c: IMUL RCX, RDI
0x18002c430: ADD RCX, R11
0x18002c433: MOVSS dword ptr [RCX + RAX*0x1], XMM5
0x18002c438: MOV RCX, qword ptr [R15]
0x18002c43b: MOV R8, qword ptr [R14]
0x18002c43e: MOVSXD RDX, dword ptr [R15 + 0x8]
0x18002c442: IMUL RDX, RDI
0x18002c446: ADD RDX, R13
0x18002c449: MOV EAX, dword ptr [RDX + RCX*0x1 + 0x6]
0x18002c44d: MOV dword ptr [RBP + -0x70], EAX
0x18002c450: MOVZX EAX, AX
0x18002c453: MOVD XMM4, EAX
0x18002c457: MOVZX EAX, word ptr [RBP + -0x6e]
0x18002c45b: MOVD XMM3, EAX
0x18002c45f: MOVZX EAX, word ptr [RDX + RCX*0x1 + 0xa]
0x18002c464: MOVSXD RDX, dword ptr [R14 + 0x8]
0x18002c468: CVTDQ2PS XMM4, XMM4
0x18002c46b: MOVD XMM5, EAX
0x18002c46f: MOVAPS XMM2, XMM4
0x18002c472: MOV EAX, EDX
0x18002c474: CVTDQ2PS XMM3, XMM3
0x18002c477: IMUL EAX, R12D
0x18002c47b: MOVSXD RCX, EAX
0x18002c47e: ADD RCX, R9
0x18002c481: MOVAPS XMM1, XMM3
0x18002c484: CVTDQ2PS XMM5, XMM5
0x18002c487: ADDSS XMM2, dword ptr [RCX + R8*0x1 + 0xc]
0x18002c48e: ADDSS XMM1, dword ptr [RCX + R8*0x1 + 0x10]
0x18002c495: MOVAPS XMM0, XMM5
0x18002c498: ADDSS XMM0, dword ptr [RCX + R8*0x1 + 0x14]
0x18002c49f: MOV RCX, RDX
0x18002c4a2: IMUL RCX, RDI
0x18002c4a6: MOVSS dword ptr [RBP + -0x40], XMM0
0x18002c4ab: MOVAPS XMM0, XMM2
0x18002c4ae: ADD RCX, R9
0x18002c4b1: UNPCKLPS XMM0, XMM1
0x18002c4b4: MOVSD qword ptr [RCX + R8*0x1 + 0xc], XMM0
0x18002c4bb: MOV EAX, dword ptr [RBP + -0x40]
0x18002c4be: MULSS XMM5, XMM6
0x18002c4c2: MULSS XMM3, XMM7
0x18002c4c6: MOV dword ptr [RCX + R8*0x1 + 0x14], EAX
0x18002c4cb: MOV RAX, qword ptr [RSI]
0x18002c4ce: MOVSXD RCX, dword ptr [RSI + 0x8]
0x18002c4d2: ADDSS XMM5, XMM3
0x18002c4d6: MULSS XMM4, XMM8
0x18002c4db: IMUL RCX, RDI
0x18002c4df: ADD RCX, R11
0x18002c4e2: ADDSS XMM5, XMM4
0x18002c4e6: MOVSS dword ptr [RCX + RAX*0x1 + 0x4], XMM5
0x18002c4ec: MOVSXD RDX, dword ptr [R14 + 0x8]
0x18002c4f0: MOVSXD RCX, dword ptr [R15 + 0x8]
0x18002c4f4: IMUL RCX, RDI
0x18002c4f8: ADD RCX, qword ptr [R15]
0x18002c4fb: MOV EAX, dword ptr [RCX + R13*0x1 + 0xc]
0x18002c500: MOV dword ptr [RBP + -0x70], EAX
0x18002c503: MOVZX EAX, AX
0x18002c506: MOVD XMM4, EAX
0x18002c50a: MOVZX EAX, word ptr [RBP + -0x6e]
0x18002c50e: MOVD XMM3, EAX
0x18002c512: MOVZX EAX, word ptr [RCX + R13*0x1 + 0x10]
0x18002c518: CVTDQ2PS XMM4, XMM4
0x18002c51b: MOVD XMM5, EAX
0x18002c51f: MOVAPS XMM2, XMM4
0x18002c522: CVTDQ2PS XMM3, XMM3
0x18002c525: MOV EAX, EDX
0x18002c527: MULSS XMM4, XMM8
0x18002c52c: IMUL EAX, R12D
0x18002c530: MOVSXD RCX, EAX
0x18002c533: ADD RCX, qword ptr [R14]
0x18002c536: MOVAPS XMM1, XMM3
0x18002c539: CVTDQ2PS XMM5, XMM5
0x18002c53c: ADDSS XMM2, dword ptr [RCX + R9*0x1 + 0x18]
0x18002c543: ADDSS XMM1, dword ptr [RCX + R9*0x1 + 0x1c]
0x18002c54a: MOVAPS XMM0, XMM5
0x18002c54d: MULSS XMM3, XMM7
0x18002c551: MULSS XMM5, XMM6
0x18002c555: ADDSS XMM0, dword ptr [RCX + R9*0x1 + 0x20]
0x18002c55c: ADDSS XMM5, XMM3
0x18002c560: MOV RCX, RDX
0x18002c563: IMUL RCX, RDI
0x18002c567: ADD RCX, qword ptr [R14]
0x18002c56a: MOVSS dword ptr [RBP + -0x40], XMM0
0x18002c56f: ADDSS XMM5, XMM4
0x18002c573: MOV EAX, dword ptr [RBP + -0x40]
0x18002c576: MOVAPS XMM0, XMM2
0x18002c579: UNPCKLPS XMM0, XMM1
0x18002c57c: MOVSD qword ptr [RCX + R9*0x1 + 0x18], XMM0
0x18002c583: MOV dword ptr [RCX + R9*0x1 + 0x20], EAX
0x18002c588: MOVSXD RAX, dword ptr [RSI + 0x8]
0x18002c58c: IMUL RAX, RDI
0x18002c590: ADD RAX, qword ptr [RSI]
0x18002c593: MOVSS dword ptr [RAX + R11*0x1 + 0x8], XMM5
0x18002c59a: MOVSXD RCX, dword ptr [R15 + 0x8]
0x18002c59e: MOVSXD RDX, dword ptr [R14 + 0x8]
0x18002c5a2: IMUL RCX, RDI
0x18002c5a6: ADD RCX, qword ptr [R15]
0x18002c5a9: MOV EAX, dword ptr [RCX + R13*0x1 + 0x12]
0x18002c5ae: MOV dword ptr [RBP + -0x70], EAX
0x18002c5b1: MOVZX EAX, AX
0x18002c5b4: MOVD XMM4, EAX
0x18002c5b8: MOVZX EAX, word ptr [RBP + -0x6e]
0x18002c5bc: MOVD XMM3, EAX
0x18002c5c0: MOVZX EAX, word ptr [RCX + R13*0x1 + 0x16]
0x18002c5c6: CVTDQ2PS XMM4, XMM4
0x18002c5c9: MOVD XMM5, EAX
0x18002c5cd: MOVAPS XMM2, XMM4
0x18002c5d0: MOV EAX, EDX
0x18002c5d2: CVTDQ2PS XMM3, XMM3
0x18002c5d5: IMUL EAX, R12D
0x18002c5d9: MOVSXD RCX, EAX
0x18002c5dc: ADD RCX, qword ptr [R14]
0x18002c5df: MOVAPS XMM1, XMM3
0x18002c5e2: CVTDQ2PS XMM5, XMM5
0x18002c5e5: ADDSS XMM2, dword ptr [RCX + R9*0x1 + 0x24]
0x18002c5ec: ADDSS XMM1, dword ptr [RCX + R9*0x1 + 0x28]
0x18002c5f3: MOVAPS XMM0, XMM5
0x18002c5f6: ADDSS XMM0, dword ptr [RCX + R9*0x1 + 0x2c]
0x18002c5fd: MOVSS dword ptr [RBP + -0x40], XMM0
0x18002c602: MOV EAX, dword ptr [RBP + -0x40]
0x18002c605: MULSS XMM5, XMM6
0x18002c609: MOV RCX, RDX
0x18002c60c: MOVAPS XMM0, XMM2
0x18002c60f: UNPCKLPS XMM0, XMM1
0x18002c612: MULSS XMM3, XMM7
0x18002c616: IMUL RCX, RDI
0x18002c61a: ADD RBX, 0x4
0x18002c61e: ADD R13, 0x18
0x18002c622: ADD RCX, qword ptr [R14]
0x18002c625: ADDSS XMM5, XMM3
0x18002c629: MULSS XMM4, XMM8
0x18002c62e: MOVSD qword ptr [RCX + R9*0x1 + 0x24], XMM0
0x18002c635: MOV dword ptr [RCX + R9*0x1 + 0x2c], EAX
0x18002c63a: MOVSXD RAX, dword ptr [RSI + 0x8]
0x18002c63e: MOV R9, qword ptr [RBP + -0x60]
0x18002c642: ADDSS XMM5, XMM4
0x18002c646: IMUL RAX, RDI
0x18002c64a: ADD RAX, qword ptr [RSI]
0x18002c64d: ADD R9, 0x30
0x18002c651: MOV qword ptr [RBP + -0x60], R9
0x18002c655: MOVSS dword ptr [RAX + R11*0x1 + 0xc], XMM5
0x18002c65c: CMP RBX, qword ptr [RBP + -0x58]
0x18002c660: JL 0x18002c380
0x18002c666: MOV R13, qword ptr [RBP + -0x50]
0x18002c66a: MOV R11D, dword ptr [RBP + 0x38]
0x18002c66e: CMP RBX, R13
0x18002c671: JGE 0x18002c74a
0x18002c677: LEA R9, [RBX + RBX*0x2]
0x18002c67b: LEA R10, [RBX + RBX*0x2]
0x18002c67f: SHL R9, 0x2
0x18002c683: ADD R10, R10
0x18002c686: MOVSXD RCX, dword ptr [R15 + 0x8]
0x18002c68a: MOVSXD RDX, dword ptr [R14 + 0x8]
0x18002c68e: ADD R10, 0x6
0x18002c692: ADD R9, 0xc
0x18002c696: IMUL RCX, RDI
0x18002c69a: ADD RCX, qword ptr [R15]
0x18002c69d: MOV EAX, dword ptr [RCX + R10*0x1 + -0x6]
0x18002c6a2: MOV dword ptr [RBP + -0x70], EAX
0x18002c6a5: MOVZX EAX, AX
0x18002c6a8: MOVD XMM4, EAX
0x18002c6ac: MOVZX EAX, word ptr [RBP + -0x6e]
0x18002c6b0: MOVD XMM3, EAX
0x18002c6b4: MOVZX EAX, word ptr [RCX + R10*0x1 + -0x2]
0x18002c6ba: CVTDQ2PS XMM4, XMM4
0x18002c6bd: MOVD XMM5, EAX
0x18002c6c1: MOVAPS XMM2, XMM4
0x18002c6c4: MOV EAX, EDX
0x18002c6c6: CVTDQ2PS XMM3, XMM3
0x18002c6c9: MULSS XMM4, XMM8
0x18002c6ce: IMUL EAX, R12D
0x18002c6d2: MOVSXD RCX, EAX
0x18002c6d5: ADD RCX, qword ptr [R14]
0x18002c6d8: MOVAPS XMM1, XMM3
0x18002c6db: CVTDQ2PS XMM5, XMM5
0x18002c6de: ADDSS XMM2, dword ptr [RCX + R9*0x1 + -0xc]
0x18002c6e5: ADDSS XMM1, dword ptr [RCX + R9*0x1 + -0x8]
0x18002c6ec: MOVAPS XMM0, XMM5
0x18002c6ef: MULSS XMM3, XMM7
0x18002c6f3: MULSS XMM5, XMM6
0x18002c6f7: ADDSS XMM0, dword ptr [RCX + R9*0x1 + -0x4]
0x18002c6fe: ADDSS XMM5, XMM3
0x18002c702: MOV RCX, RDX
0x18002c705: IMUL RCX, RDI
0x18002c709: ADD RCX, qword ptr [R14]
0x18002c70c: MOVSS dword ptr [RBP + -0x40], XMM0
0x18002c711: ADDSS XMM5, XMM4
0x18002c715: MOV EAX, dword ptr [RBP + -0x40]
0x18002c718: MOVAPS XMM0, XMM2
0x18002c71b: UNPCKLPS XMM0, XMM1
0x18002c71e: MOVSD qword ptr [RCX + R9*0x1 + -0xc], XMM0
0x18002c725: MOV dword ptr [RCX + R9*0x1 + -0x4], EAX
0x18002c72a: MOVSXD RAX, dword ptr [RSI + 0x8]
0x18002c72e: IMUL RAX, RDI
0x18002c732: LEA RCX, [RAX + RBX*0x4]
0x18002c736: MOV RAX, qword ptr [RSI]
0x18002c739: INC RBX
0x18002c73c: MOVSS dword ptr [RCX + RAX*0x1], XMM5
0x18002c741: CMP RBX, R13
0x18002c744: JL 0x18002c686
0x18002c74a: INC R12D
0x18002c74d: INC RDI
0x18002c750: CMP R12D, R11D
0x18002c753: JL 0x18002c360
0x18002c759: MOVAPS XMM8, xmmword ptr [RSP + 0x40]
0x18002c75f: MOVAPS XMM7, xmmword ptr [RSP + 0x50]
0x18002c764: MOVAPS XMM6, xmmword ptr [RSP + 0x60]
0x18002c769: MOV RDI, qword ptr [RSP + 0xb8]
0x18002c771: MOV RSI, qword ptr [RSP + 0xb0]
0x18002c779: MOV RBX, qword ptr [RSP + 0xa0]
0x18002c781: ADD RSP, 0x70
0x18002c785: POP R15
0x18002c787: POP R14
0x18002c789: POP R13
0x18002c78b: POP R12
0x18002c78d: POP RBP
0x18002c78e: RET
```

## procedure_pseudo_code {'procedure': '0x1800244d0'}

Evidence: `ev_45613b98d138d190a95f42ea3c949ca43b5bd318624973976f94d445ceaa2635`

```c

void FUN_1800244d0(longlong param_1,longlong param_2,longlong param_3)

{
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  longlong local_18;
  undefined8 uStack_10;
  
  local_28 = *(longlong *)(param_3 + 0x10);
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_18 = *(longlong *)(param_2 + 0x10);
  uStack_20 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_2 + 0x50));
  local_38 = *(longlong *)(param_1 + 0x10);
  uStack_10 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_1 + 0x50));
  FUN_18002c2f0((*(int **)(param_1 + 0x40))[1],**(int **)(param_1 + 0x40),&local_38,&local_18,
                &local_28);
  return;
}


```

## procedure_assembly {'procedure': '0x18002c0d0'}

Evidence: `ev_69af2bc5c4ffa73c29e55d21ca5083e5069699622e602d162cd4da87c7ac4d0a`

```text
0x18002c0d0: TEST EDX, EDX
0x18002c0d2: JLE 0x18002c2e1
0x18002c0d8: MOV qword ptr [RSP + 0x10], RBX
0x18002c0dd: MOV qword ptr [RSP + 0x18], RSI
0x18002c0e2: MOV qword ptr [RSP + 0x20], RDI
0x18002c0e7: PUSH R14
0x18002c0e9: MOV R11, qword ptr [RSP + 0x30]
0x18002c0ee: MOV RBX, R9
0x18002c0f1: MOV RDI, R8
0x18002c0f4: MOVSXD RSI, ECX
0x18002c0f7: XOR R10D, R10D
0x18002c0fa: MOVSXD R14, EDX
0x18002c0fd: MOV qword ptr [RSP + 0x10], RBP
0x18002c102: XOR R9D, R9D
0x18002c105: CMP RSI, 0x4
0x18002c109: JL 0x18002c262
0x18002c10f: LEA RBP, [RSI + -0x3]
0x18002c113: NOP dword ptr [RAX]
0x18002c117: NOP word ptr [RAX + RAX*0x1]
0x18002c120: MOV RAX, qword ptr [RDI]
0x18002c123: MOVSXD RCX, dword ptr [RDI + 0x8]
0x18002c127: LEA R8, [R9*0x4]
0x18002c12f: IMUL RCX, R10
0x18002c133: LEA RDX, [R9 + R9*0x1]
0x18002c137: ADD R9, 0x4
0x18002c13b: ADD RCX, RDX
0x18002c13e: MOVZX ECX, word ptr [RCX + RAX*0x1]
0x18002c142: MOV RAX, qword ptr [RBX]
0x18002c145: MOVD XMM1, ECX
0x18002c149: MOVSXD RCX, dword ptr [RBX + 0x8]
0x18002c14d: IMUL RCX, R10
0x18002c151: CVTDQ2PS XMM1, XMM1
0x18002c154: ADD RCX, R8
0x18002c157: MOVAPS XMM0, XMM1
0x18002c15a: ADDSS XMM0, dword ptr [RCX + RAX*0x1]
0x18002c15f: MOVSS dword ptr [RCX + RAX*0x1], XMM0
0x18002c164: MOVSXD RCX, dword ptr [R11 + 0x8]
0x18002c168: MOV RAX, qword ptr [R11]
0x18002c16b: IMUL RCX, R10
0x18002c16f: ADD RCX, R8
0x18002c172: MOVSS dword ptr [RCX + RAX*0x1], XMM1
0x18002c177: MOV RAX, qword ptr [RDI]
0x18002c17a: MOVSXD RCX, dword ptr [RDI + 0x8]
0x18002c17e: IMUL RCX, R10
0x18002c182: ADD RCX, RDX
0x18002c185: ADD RDX, 0x6
0x18002c189: MOVZX ECX, word ptr [RCX + RAX*0x1 + 0x2]
0x18002c18e: MOV RAX, qword ptr [RBX]
0x18002c191: MOVD XMM1, ECX
0x18002c195: MOVSXD RCX, dword ptr [RBX + 0x8]
0x18002c199: IMUL RCX, R10
0x18002c19d: CVTDQ2PS XMM1, XMM1
0x18002c1a0: ADD RCX, R8
0x18002c1a3: MOVAPS XMM0, XMM1
0x18002c1a6: ADDSS XMM0, dword ptr [RCX + RAX*0x1 + 0x4]
0x18002c1ac: MOVSS dword ptr [RCX + RAX*0x1 + 0x4], XMM0
0x18002c1b2: MOVSXD RCX, dword ptr [R11 + 0x8]
0x18002c1b6: MOV RAX, qword ptr [R11]
0x18002c1b9: IMUL RCX, R10
0x18002c1bd: ADD RCX, R8
0x18002c1c0: ADD R8, 0xc
0x18002c1c4: MOVSS dword ptr [RCX + RAX*0x1 + 0x4], XMM1
0x18002c1ca: MOV RAX, qword ptr [RDI]
0x18002c1cd: MOVSXD RCX, dword ptr [RDI + 0x8]
0x18002c1d1: IMUL RCX, R10
0x18002c1d5: ADD RCX, RDX
0x18002c1d8: MOVZX ECX, word ptr [RCX + RAX*0x1 + -0x2]
0x18002c1dd: MOVSXD RAX, dword ptr [RBX + 0x8]
0x18002c1e1: MOVD XMM1, ECX
0x18002c1e5: IMUL RAX, R10
0x18002c1e9: ADD RAX, qword ptr [RBX]
0x18002c1ec: CVTDQ2PS XMM1, XMM1
0x18002c1ef: MOVAPS XMM0, XMM1
0x18002c1f2: ADDSS XMM0, dword ptr [RAX + R8*0x1 + -0x4]
0x18002c1f9: MOVSS dword ptr [RAX + R8*0x1 + -0x4], XMM0
0x18002c200: MOVSXD RAX, dword ptr [R11 + 0x8]
0x18002c204: IMUL RAX, R10
0x18002c208: ADD RAX, qword ptr [R11]
0x18002c20b: MOVSS dword ptr [RAX + R8*0x1 + -0x4], XMM1
0x18002c212: MOV RAX, qword ptr [RDI]
0x18002c215: MOVSXD RCX, dword ptr [RDI + 0x8]
0x18002c219: IMUL RCX, R10
0x18002c21d: ADD RCX, RDX
0x18002c220: MOVZX ECX, word ptr [RCX + RAX*0x1]
0x18002c224: MOV RAX, qword ptr [RBX]
0x18002c227: MOVD XMM1, ECX
0x18002c22b: MOVSXD RCX, dword ptr [RBX + 0x8]
0x18002c22f: IMUL RCX, R10
0x18002c233: CVTDQ2PS XMM1, XMM1
0x18002c236: ADD RCX, R8
0x18002c239: MOVAPS XMM0, XMM1
0x18002c23c: ADDSS XMM0, dword ptr [RCX + RAX*0x1]
0x18002c241: MOVSS dword ptr [RCX + RAX*0x1], XMM0
0x18002c246: MOVSXD RCX, dword ptr [R11 + 0x8]
0x18002c24a: MOV RAX, qword ptr [R11]
0x18002c24d: IMUL RCX, R10
0x18002c251: ADD RCX, R8
0x18002c254: MOVSS dword ptr [RCX + RAX*0x1], XMM1
0x18002c259: CMP R9, RBP
0x18002c25c: JL 0x18002c120
0x18002c262: CMP R9, RSI
0x18002c265: JGE 0x18002c2bf
0x18002c267: MOVSXD RAX, dword ptr [RDI + 0x8]
0x18002c26b: LEA RDX, [R9*0x4]
0x18002c273: IMUL RAX, R10
0x18002c277: LEA RCX, [RAX + R9*0x2]
0x18002c27b: MOV RAX, qword ptr [RDI]
0x18002c27e: INC R9
0x18002c281: MOVZX ECX, word ptr [RCX + RAX*0x1]
0x18002c285: MOV RAX, qword ptr [RBX]
0x18002c288: MOVD XMM1, ECX
0x18002c28c: MOVSXD RCX, dword ptr [RBX + 0x8]
0x18002c290: IMUL RCX, R10
0x18002c294: CVTDQ2PS XMM1, XMM1
0x18002c297: ADD RCX, RDX
0x18002c29a: MOVAPS XMM0, XMM1
0x18002c29d: ADDSS XMM0, dword ptr [RCX + RAX*0x1]
0x18002c2a2: MOVSS dword ptr [RCX + RAX*0x1], XMM0
0x18002c2a7: MOVSXD RCX, dword ptr [R11 + 0x8]
0x18002c2ab: MOV RAX, qword ptr [R11]
0x18002c2ae: IMUL RCX, R10
0x18002c2b2: ADD RCX, RDX
0x18002c2b5: MOVSS dword ptr [RCX + RAX*0x1], XMM1
0x18002c2ba: CMP R9, RSI
0x18002c2bd: JL 0x18002c267
0x18002c2bf: INC R10
0x18002c2c2: CMP R10, R14
0x18002c2c5: JL 0x18002c102
0x18002c2cb: MOV RBP, qword ptr [RSP + 0x10]
0x18002c2d0: MOV RBX, qword ptr [RSP + 0x18]
0x18002c2d5: MOV RSI, qword ptr [RSP + 0x20]
0x18002c2da: MOV RDI, qword ptr [RSP + 0x28]
0x18002c2df: POP R14
0x18002c2e1: RET
```

## procedure_pseudo_code {'procedure': '0x18002c2f0'}

Evidence: `ev_7b9745d8c6c453454cdfe63a69991d86357b00fe887b36db9c0deaf7fca623fc`

```c

void FUN_18002c2f0(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5)

{
  float fVar1;
  uint uVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  longlong lVar7;
  longlong lVar8;
  int iVar9;
  longlong lVar10;
  longlong lVar11;
  float fVar12;
  float fVar13;
  float fVar14;
  longlong local_88;
  
  if (0 < param_2) {
    lVar10 = (longlong)param_1;
    iVar9 = 0;
    lVar7 = 0;
    do {
      lVar6 = 0;
      if (3 < lVar10) {
        lVar11 = 0;
        local_88 = 0;
        do {
          lVar4 = *param_4;
          lVar5 = (int)param_3[1] * lVar7 + lVar11;
          lVar8 = lVar6 * 4;
          uVar2 = *(uint *)(lVar5 + *param_3);
          fVar13 = (float)(uVar2 & 0xffff);
          fVar12 = (float)(uVar2 >> 0x10);
          lVar3 = (int)param_4[1] * iVar9 + local_88;
          fVar14 = (float)*(ushort *)(lVar5 + 4 + *param_3);
          fVar1 = *(float *)(lVar3 + 8 + lVar4);
          lVar5 = (int)param_4[1] * lVar7 + local_88;
          *(ulonglong *)(lVar5 + lVar4) =
               CONCAT44(fVar12 + *(float *)(lVar3 + 4 + lVar4),fVar13 + *(float *)(lVar3 + lVar4));
          *(float *)(lVar5 + 8 + lVar4) = fVar14 + fVar1;
          *(float *)((int)param_5[1] * lVar7 + lVar8 + *param_5) =
               fVar14 * 0.299 + fVar12 * 0.587 + fVar13 * 0.114;
          lVar4 = *param_4;
          lVar5 = (int)param_3[1] * lVar7 + lVar11;
          uVar2 = *(uint *)(lVar5 + 6 + *param_3);
          fVar13 = (float)(uVar2 & 0xffff);
          fVar12 = (float)(uVar2 >> 0x10);
          lVar3 = (int)param_4[1] * iVar9 + local_88;
          fVar14 = (float)*(ushort *)(lVar5 + 10 + *param_3);
          fVar1 = *(float *)(lVar3 + 0x14 + lVar4);
          lVar5 = (int)param_4[1] * lVar7 + local_88;
          *(ulonglong *)(lVar5 + 0xc + lVar4) =
               CONCAT44(fVar12 + *(float *)(lVar3 + 0x10 + lVar4),
                        fVar13 + *(float *)(lVar3 + 0xc + lVar4));
          *(float *)(lVar5 + 0x14 + lVar4) = fVar14 + fVar1;
          *(float *)((int)param_5[1] * lVar7 + lVar8 + 4 + *param_5) =
               fVar14 * 0.299 + fVar12 * 0.587 + fVar13 * 0.114;
          lVar4 = (int)param_3[1] * lVar7 + *param_3;
          uVar2 = *(uint *)(lVar4 + 0xc + lVar11);
          fVar13 = (float)(uVar2 & 0xffff);
          fVar12 = (float)(uVar2 >> 0x10);
          lVar3 = (longlong)((int)param_4[1] * iVar9) + *param_4;
          fVar14 = (float)*(ushort *)(lVar4 + 0x10 + lVar11);
          fVar1 = *(float *)(lVar3 + 0x20 + local_88);
          lVar4 = (int)param_4[1] * lVar7 + *param_4;
          *(ulonglong *)(lVar4 + 0x18 + local_88) =
               CONCAT44(fVar12 + *(float *)(lVar3 + 0x1c + local_88),
                        fVar13 + *(float *)(lVar3 + 0x18 + local_88));
          *(float *)(lVar4 + 0x20 + local_88) = fVar14 + fVar1;
          *(float *)((int)param_5[1] * lVar7 + *param_5 + 8 + lVar8) =
               fVar14 * 0.299 + fVar12 * 0.587 + fVar13 * 0.114;
          lVar4 = (int)param_3[1] * lVar7 + *param_3;
          uVar2 = *(uint *)(lVar4 + 0x12 + lVar11);
          fVar13 = (float)(uVar2 & 0xffff);
          fVar12 = (float)(uVar2 >> 0x10);
          lVar3 = (longlong)((int)param_4[1] * iVar9) + *param_4;
          fVar14 = (float)*(ushort *)(lVar4 + 0x16 + lVar11);
          fVar1 = *(float *)(lVar3 + 0x2c + local_88);
          lVar6 = lVar6 + 4;
          lVar11 = lVar11 + 0x18;
          lVar4 = (int)param_4[1] * lVar7 + *param_4;
          *(ulonglong *)(lVar4 + 0x24 + local_88) =
               CONCAT44(fVar12 + *(float *)(lVar3 + 0x28 + local_88),
                        fVar13 + *(float *)(lVar3 + 0x24 + local_88));
          *(float *)(lVar4 + 0x2c + local_88) = fVar14 + fVar1;
          local_88 = local_88 + 0x30;
          *(float *)((int)param_5[1] * lVar7 + *param_5 + 0xc + lVar8) =
               fVar14 * 0.299 + fVar12 * 0.587 + fVar13 * 0.114;
        } while (lVar6 < lVar10 + -3);
      }
      if (lVar6 < lVar10) {
        lVar8 = lVar6 * 0xc;
        lVar11 = lVar6 * 6;
        do {
          lVar11 = lVar11 + 6;
          lVar8 = lVar8 + 0xc;
          lVar4 = (int)param_3[1] * lVar7 + *param_3;
          uVar2 = *(uint *)(lVar4 + -6 + lVar11);
          fVar13 = (float)(uVar2 & 0xffff);
          fVar12 = (float)(uVar2 >> 0x10);
          lVar3 = (longlong)((int)param_4[1] * iVar9) + *param_4;
          fVar14 = (float)*(ushort *)(lVar4 + -2 + lVar11);
          fVar1 = *(float *)(lVar3 + -4 + lVar8);
          lVar4 = (int)param_4[1] * lVar7 + *param_4;
          *(ulonglong *)(lVar4 + -0xc + lVar8) =
               CONCAT44(fVar12 + *(float *)(lVar3 + -8 + lVar8),
                        fVar13 + *(float *)(lVar3 + -0xc + lVar8));
          *(float *)(lVar4 + -4 + lVar8) = fVar14 + fVar1;
          lVar4 = lVar6 * 4;
          lVar6 = lVar6 + 1;
          *(float *)((int)param_5[1] * lVar7 + lVar4 + *param_5) =
               fVar14 * 0.299 + fVar12 * 0.587 + fVar13 * 0.114;
        } while (lVar6 < lVar10);
      }
      iVar9 = iVar9 + 1;
      lVar7 = lVar7 + 1;
    } while (iVar9 < param_2);
  }
  return;
}


```

## procedure_pseudo_code {'procedure': '0x18002c0d0'}

Evidence: `ev_b83f6c9c469c6fd013342c8a33a333ff09d466cf34d4e114354163d507f2dc0a`

```c

void FUN_18002c0d0(int param_1,int param_2,longlong *param_3,longlong *param_4,longlong *param_5)

{
  longlong lVar1;
  longlong lVar2;
  longlong lVar3;
  longlong lVar4;
  longlong lVar5;
  longlong lVar6;
  float fVar7;
  
  if (0 < param_2) {
    lVar4 = (longlong)param_1;
    lVar6 = 0;
    do {
      lVar5 = 0;
      if (3 < lVar4) {
        do {
          lVar2 = lVar5 * 4;
          lVar1 = lVar5 * 2;
          lVar5 = lVar5 + 4;
          fVar7 = (float)*(ushort *)((int)param_3[1] * lVar6 + lVar1 + *param_3);
          lVar3 = (int)param_4[1] * lVar6 + lVar2;
          *(float *)(lVar3 + *param_4) = fVar7 + *(float *)(lVar3 + *param_4);
          *(float *)((int)param_5[1] * lVar6 + lVar2 + *param_5) = fVar7;
          fVar7 = (float)*(ushort *)((int)param_3[1] * lVar6 + lVar1 + 2 + *param_3);
          lVar3 = (int)param_4[1] * lVar6 + lVar2;
          *(float *)(lVar3 + 4 + *param_4) = fVar7 + *(float *)(lVar3 + 4 + *param_4);
          lVar3 = lVar2 + 0xc;
          *(float *)((int)param_5[1] * lVar6 + lVar2 + 4 + *param_5) = fVar7;
          lVar2 = (int)param_4[1] * lVar6 + *param_4;
          fVar7 = (float)*(ushort *)((int)param_3[1] * lVar6 + lVar1 + 6 + -2 + *param_3);
          *(float *)(lVar2 + -4 + lVar3) = fVar7 + *(float *)(lVar2 + -4 + lVar3);
          *(float *)((int)param_5[1] * lVar6 + *param_5 + -4 + lVar3) = fVar7;
          fVar7 = (float)*(ushort *)((int)param_3[1] * lVar6 + lVar1 + 6 + *param_3);
          lVar2 = (int)param_4[1] * lVar6 + lVar3;
          *(float *)(lVar2 + *param_4) = fVar7 + *(float *)(lVar2 + *param_4);
          *(float *)((int)param_5[1] * lVar6 + lVar3 + *param_5) = fVar7;
        } while (lVar5 < lVar4 + -3);
      }
      for (; lVar5 < lVar4; lVar5 = lVar5 + 1) {
        fVar7 = (float)*(ushort *)((int)param_3[1] * lVar6 + lVar5 * 2 + *param_3);
        lVar2 = (int)param_4[1] * lVar6 + lVar5 * 4;
        *(float *)(lVar2 + *param_4) = fVar7 + *(float *)(lVar2 + *param_4);
        *(float *)((int)param_5[1] * lVar6 + lVar5 * 4 + *param_5) = fVar7;
      }
      lVar6 = lVar6 + 1;
    } while (lVar6 < param_2);
  }
  return;
}


```

## read_bytes {'address': '0x18007c0c8', 'length': 32}

Evidence: `ev_c6e5c0c4cc790121912c86668d568289a0f6b49b4a0846fc766c0c082d1ec172`

```text
{
  "address": "0x18007c0c8",
  "bytes_hex": "2043028001000000b0430280010000004044028001000000d044028001000000",
  "complete": true,
  "requested_bytes": 32,
  "returned_bytes": 32
}
```

## procedure_pseudo_code {'procedure': '0x180024440'}

Evidence: `ev_d04d12e749bb9c41f940bd7b00d5b290ec371fd84f8919552b550de2e740551c`

```c

void FUN_180024440(longlong param_1,longlong param_2,longlong param_3)

{
  longlong local_38;
  undefined8 uStack_30;
  longlong local_28;
  undefined8 uStack_20;
  longlong local_18;
  undefined8 uStack_10;
  
  local_28 = *(longlong *)(param_3 + 0x10);
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_3 + 0x50));
  local_18 = *(longlong *)(param_2 + 0x10);
  uStack_20 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_2 + 0x50));
  local_38 = *(longlong *)(param_1 + 0x10);
  uStack_10 = uStack_30;
  uStack_30 = CONCAT44(uStack_30._4_4_,*(undefined4 *)(param_1 + 0x50));
  FUN_18002c0d0((*(int **)(param_1 + 0x40))[1],**(int **)(param_1 + 0x40),&local_38,&local_18,
                &local_28);
  return;
}


```
