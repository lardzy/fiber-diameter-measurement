# SGSharp managed anchors
Evidence: ev_379de98b66523eeb7bc419eabf4a809f30e787c4bec9fb63d97cafd76c9bb6b2
SHA256: cb03932b0582204728b20e68bf6342f4f6d1ed0b155b8652ba1bca0a34baaebc


## Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam..ctor 0x06000026
```
   0 ldarg.0  
   1 call 0x0a000014 ..ctor
   6 ret  
```

## Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.GetDefaultParam 0x06000083
```
   0 ldarg.0  
   1 ldsfld 0x04000078 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._EFI_PARAM_PATH_WORKDIR
   6 ldsfld 0x04000079 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._EFI_DEFAULTPARAM_FILE_NAME
  11 call 0x0a000044 .Concat
  16 stloc.0  
  17 ldsfld 0x0400007a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._PARAM_INI_SECTION_NAME_INPUT
  22 stloc.1  
  23 newobj 0x06000086 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam..ctor
  28 stloc.2  
  29 ldloc.2  
  30 ldc.i4 1200 
  35 callvirt 0x0600002a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefWidth
  40 ldloc.2  
  41 ldc.i4 1200 
  46 callvirt 0x0600002c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefHeight
  51 ldloc.2  
  52 ldc.i4.3  
  53 callvirt 0x0600002e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefChannel
  58 ldloc.2  
  59 ldc.i4.8  
  60 callvirt 0x06000030 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefColorBits
  65 ldloc.2  
  66 ldc.i4.s 100 
  68 callvirt 0x06000032 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefMaxImgNum
  73 ldloc.2  
  74 ldloc.2  
  75 callvirt 0x06000031 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefMaxImgNum
  80 callvirt 0x06000034 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefCacheImgNum
  85 ldloc.2  
  86 ldc.i4.0  
  87 callvirt 0x06000036 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefArbitraryZOrder
  92 ldloc.2  
  93 ldc.i4.1  
  94 callvirt 0x06000038 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefEFIBlendMode
  99 ldloc.2  
 100 ldc.i4.4  
 101 callvirt 0x0600003a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefLapPyrLevel
 106 ldloc.2  
 107 ldc.i4.3  
 108 callvirt 0x0600003c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefDepthMapPyrLevel
 113 ldloc.2  
 114 ldc.i4.3  
 115 callvirt 0x0600003e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefGaussFilterSizeFastEFI
 120 ldloc.2  
 121 ldc.i4.s 11 
 123 callvirt 0x06000040 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefGaussFilterSize3D
 128 ldarg.0  
 129 ldarg.1  
 130 ldc.i4.1  
 131 call 0x06000084 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._GetBandBlendParams
 136 stloc.3  
 137 ldloc.2  
 138 ldloc.3  
 139 ldc.i4.0  
 140 ldelem.r8  
 141 callvirt 0x06000042 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlendFastEFI0
 146 ldloc.2  
 147 ldloc.3  
 148 ldc.i4.1  
 149 ldelem.r8  
 150 callvirt 0x06000044 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlendFastEFI1
 155 ldloc.2  
 156 ldloc.3  
 157 ldc.i4.2  
 158 ldelem.r8  
 159 callvirt 0x06000046 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlendFastEFI2
 164 ldloc.2  
 165 ldloc.3  
 166 ldc.i4.3  
 167 ldelem.r8  
 168 callvirt 0x06000048 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlendFastEFI3
 173 ldarg.0  
 174 ldarg.1  
 175 ldc.i4.0  
 176 call 0x06000084 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._GetBandBlendParams
 181 stloc.s 4 
 183 ldloc.2  
 184 ldloc.s 4 
 186 ldc.i4.0  
 187 ldelem.r8  
 188 callvirt 0x0600004a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlend3D0
 193 ldloc.2  
 194 ldloc.s 4 
 196 ldc.i4.1  
 197 ldelem.r8  
 198 callvirt 0x0600004c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlend3D1
 203 ldloc.2  
 204 ldloc.s 4 
 206 ldc.i4.2  
 207 ldelem.r8  
 208 callvirt 0x0600004e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlend3D2
 213 ldloc.2  
 214 ldloc.s 4 
 216 ldc.i4.3  
 217 ldelem.r8  
 218 callvirt 0x06000050 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBandBlend3D3
 223 ldloc.2  
 224 ldc.i4.5  
 225 callvirt 0x06000052 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefDataNum
 230 ldloc.2  
 231 ldc.i4.8  
 232 callvirt 0x06000054 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefDataInterpNum
 237 ldloc.2  
 238 ldc.r8 1.8 
 247 callvirt 0x06000056 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefLapThresh3D
 252 ldloc.2  
 253 ldc.r8 1.5 
 262 callvirt 0x06000058 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefLapThreshEFI
 267 ldloc.2  
 268 ldc.r8 0.5 
 277 callvirt 0x0600005a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefDownSizeRateTexless
 282 ldloc.2  
 283 ldc.i4.s 9 
 285 callvirt 0x0600005c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefGaussFilterSizeTexless
 290 ldloc.2  
 291 ldc.i4 500 
 296 callvirt 0x0600005e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefItrMax
 301 ldloc.2  
 302 ldc.r8 10 
 311 callvirt 0x06000060 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefSadThresh
 316 ldloc.2  
 317 ldc.i4.3  
 318 callvirt 0x06000062 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefGuideImgChannel
 323 ldloc.2  
 324 ldc.r8 0.5 
 333 callvirt 0x06000064 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefDownSizeRateGuided
 338 ldloc.2  
 339 ldc.i4.5  
 340 callvirt 0x06000066 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefWMRadius3D
 345 ldloc.2  
 346 ldc.r8 0.01 
 355 callvirt 0x06000068 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefWMEps3D
 360 ldloc.2  
 361 ldc.i4.5  
 362 callvirt 0x0600006a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefMedFliterSize3D
 367 ldloc.2  
 368 ldc.i4.s 9 
 370 callvirt 0x0600006c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBilateralRadius3D
 375 ldloc.2  
 376 ldc.r8 0.0001 
 385 callvirt 0x0600006e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBilateralEps3D
 390 ldloc.2  
 391 ldc.i4.1  
 392 callvirt 0x06000070 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefWMRadiusEFI
 397 ldloc.2  
 398 ldc.r8 0.01 
 407 callvirt 0x06000072 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefWMEpsEFI
 412 ldloc.2  
 413 ldc.i4.3  
 414 callvirt 0x06000074 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefMedFliterSizeEFI
 419 ldloc.2  
 420 ldc.i4.1  
 421 callvirt 0x06000076 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBilateralRadiusEFI
 426 ldloc.2  
 427 ldc.r8 0.01 
 436 callvirt 0x06000078 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBilateralEpsEFI
 441 ldloc.2  
 442 ldc.r8 5 
 451 callvirt 0x0600007a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBlendLapThreshFastSweep
 456 ldloc.2  
 457 ldc.r8 1 
 466 callvirt 0x0600007c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBlendMaxFastSweep
 471 ldloc.2  
 472 ldc.r8 30 
 481 callvirt 0x0600007e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBlendLapThreshFastFine
 486 ldloc.2  
 487 ldc.r8 1 
 496 callvirt 0x06000080 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefBlendMaxFastFine
 501 ldloc.2  
 502 ldc.i4.2  
 503 callvirt 0x06000082 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.set_DefComputeMode
 508 ldloc.2  
 509 ldloc.0  
 510 ldloc.1  
 511 call 0x2b000004 
 516 ldloc.2  
 517 ret  
```

## Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._GetBandBlendParams 0x06000084
```
   0 ldarg.0  
   1 ldarg.2  
   2 call 0x06000025 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.GetDefaultParam
   7 stloc.0  
   8 ldc.i4.6  
   9 newarr 0x01000039 
  14 stloc.s 6 
  16 ldloc.s 6 
  18 ldc.i4.0  
  19 ldloc.0  
  20 callvirt 0x06000019 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.get_DefFistLow
  25 stelem.r8  
  26 ldloc.s 6 
  28 ldc.i4.1  
  29 ldloc.0  
  30 callvirt 0x0600001b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.get_DefSecondHigh
  35 stelem.r8  
  36 ldloc.s 6 
  38 ldc.i4.2  
  39 ldloc.0  
  40 callvirt 0x0600001d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.get_DefSecondLow
  45 stelem.r8  
  46 ldloc.s 6 
  48 ldc.i4.3  
  49 ldloc.0  
  50 callvirt 0x0600001f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.get_DefThirdHigh
  55 stelem.r8  
  56 ldloc.s 6 
  58 ldc.i4.4  
  59 ldloc.0  
  60 callvirt 0x06000021 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.get_DefThirdLow
  65 stelem.r8  
  66 ldloc.s 6 
  68 ldc.i4.5  
  69 ldloc.0  
  70 callvirt 0x06000023 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.get_DefFourthHigh
  75 stelem.r8  
  76 ldloc.s 6 
  78 stloc.1  
  79 ldc.r8 0 
  88 stloc.2  
  89 ldc.r8 0 
  98 stloc.3  
  99 ldc.r8 0 
 108 stloc.s 4 
 110 ldc.r8 0 
 119 stloc.s 5 
 121 ldarg.1  
 122 ldloc.1  
 123 ldloca.s 2 
 125 ldloca.s 3 
 127 ldloca.s 4 
 129 ldloca.s 5 
 131 call 0x06000085 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._CalcBandBlend
 136 ldc.i4.4  
 137 newarr 0x01000039 
 142 stloc.s 7 
 144 ldloc.s 7 
 146 ldc.i4.0  
 147 ldloc.2  
 148 stelem.r8  
 149 ldloc.s 7 
 151 ldc.i4.1  
 152 ldloc.3  
 153 stelem.r8  
 154 ldloc.s 7 
 156 ldc.i4.2  
 157 ldloc.s 4 
 159 stelem.r8  
 160 ldloc.s 7 
 162 ldc.i4.3  
 163 ldloc.s 5 
 165 stelem.r8  
 166 ldloc.s 7 
 168 ret  
```

## Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam._CalcBandBlend 0x06000085
```
   0 ldarg.1  
   1 ldc.i4.0  
   2 ldelem.r8  
   3 stloc.0  
   4 ldarg.1  
   5 ldc.i4.1  
   6 ldelem.r8  
   7 stloc.1  
   8 ldarg.1  
   9 ldc.i4.2  
  10 ldelem.r8  
  11 stloc.2  
  12 ldarg.1  
  13 ldc.i4.3  
  14 ldelem.r8  
  15 stloc.3  
  16 ldarg.1  
  17 ldc.i4.4  
  18 ldelem.r8  
  19 stloc.s 4 
  21 ldarg.1  
  22 ldc.i4.5  
  23 ldelem.r8  
  24 stloc.s 5 
  26 ldarg.0  
  27 ldloc.0  
  28 blt.un.s 43 
  30 ldarg.2  
  31 ldc.r8 1 
  40 stind.r8  
  41 br.s 69 
  43 ldarg.0  
  44 ldloc.1  
  45 blt.un.s 58 
  47 ldarg.2  
  48 ldarg.0  
  49 ldloc.1  
  50 sub  
  51 ldloc.0  
  52 ldloc.1  
  53 sub  
  54 div  
  55 stind.r8  
  56 br.s 69 
  58 ldarg.2  
  59 ldc.r8 0 
  68 stind.r8  
  69 ldarg.0  
  70 ldloc.1  
  71 blt.un.s 89 
  73 ldarg.3  
  74 ldc.r8 1 
  83 ldarg.2  
  84 ldind.r8  
  85 sub  
  86 stind.r8  
  87 br.s 132 
  89 ldarg.0  
  90 ldloc.2  
  91 blt.un.s 106 
  93 ldarg.3  
  94 ldc.r8 1 
 103 stind.r8  
 104 br.s 132 
 106 ldarg.0  
 107 ldloc.3  
 108 blt.un.s 121 
 110 ldarg.3  
 111 ldarg.0  
 112 ldloc.3  
 113 sub  
 114 ldloc.2  
 115 ldloc.3  
 116 sub  
 117 div  
 118 stind.r8  
 119 br.s 132 
 121 ldarg.3  
 122 ldc.r8 0 
 131 stind.r8  
 132 ldarg.0  
 133 ldloc.2  
 134 blt.un.s 150 
 136 ldarg.s 4 
 138 ldc.r8 0 
 147 stind.r8  
 148 br.s 222 
 150 ldarg.0  
 151 ldloc.3  
 152 blt.un.s 171 
 154 ldarg.s 4 
 156 ldc.r8 1 
 165 ldarg.3  
 166 ldind.r8  
 167 sub  
 168 stind.r8  
 169 br.s 222 
 171 ldarg.0  
 172 ldloc.s 4 
 174 blt.un.s 190 
 176 ldarg.s 4 
 178 ldc.r8 1 
 187 stind.r8  
 188 br.s 222 
 190 ldarg.0  
 191 ldloc.s 5 
 193 blt.un.s 210 
 195 ldarg.s 4 
 197 ldarg.0  
 198 ldloc.s 5 
 200 sub  
 201 ldloc.s 4 
 203 ldloc.s 5 
 205 sub  
 206 div  
 207 stind.r8  
 208 br.s 222 
 210 ldarg.s 4 
 212 ldc.r8 0 
 221 stind.r8  
 222 ldarg.0  
 223 ldloc.s 5 
 225 bge.un.s 240 
 227 ldarg.s 5 
 229 ldc.r8 1 
 238 stind.r8  
 239 ret  
 240 ldarg.0  
 241 ldloc.s 4 
 243 bgt.un.s 262 
 245 ldarg.s 5 
 247 ldc.r8 1 
 256 ldarg.s 4 
 258 ldind.r8  
 259 sub  
 260 stind.r8  
 261 ret  
 262 ldarg.s 5 
 264 ldc.r8 0 
 273 stind.r8  
 274 ret  
```

## Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam..ctor 0x06000086
```
   0 ldarg.0  
   1 call 0x0a000014 ..ctor
   6 ret  
```

## Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.GetDefaultParam 0x06000093
```
   0 ldloca.s 0 
   2 initobj 0x02000012 
   8 ldarg.0  
   9 ldarg.1  
  10 call 0x06000083 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.GetDefaultParam
  15 stloc.1  
  16 ldloca.s 0 
  18 ldloc.1  
  19 callvirt 0x06000029 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefWidth
  24 stfld 0x040000b1 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nWidth
  29 ldloca.s 0 
  31 ldloc.1  
  32 callvirt 0x0600002b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefHeight
  37 stfld 0x040000b2 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nHeight
  42 ldloca.s 0 
  44 ldloc.1  
  45 callvirt 0x0600002d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefChannel
  50 stfld 0x040000b6 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nChannel
  55 ldloca.s 0 
  57 ldloc.1  
  58 callvirt 0x0600002f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefColorBits
  63 stfld 0x040000b7 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nColorBits
  68 ldloca.s 0 
  70 ldloc.1  
  71 callvirt 0x06000031 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefMaxImgNum
  76 stfld 0x040000b8 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nMaxImgNum
  81 ldloca.s 0 
  83 ldloc.1  
  84 callvirt 0x06000033 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefCacheImgNum
  89 stfld 0x040000b9 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nCacheImgNum
  94 ldloca.s 0 
  96 ldloc.1  
  97 callvirt 0x06000035 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefArbitraryZOrder
 102 stfld 0x040000ba Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.bArbitraryZOrder
 107 ldloca.s 0 
 109 ldloc.1  
 110 callvirt 0x06000037 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefEFIBlendMode
 115 stfld 0x040000bb Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.bEFIBlendMode
 120 ldloca.s 0 
 122 ldloc.1  
 123 callvirt 0x06000039 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefLapPyrLevel
 128 stfld 0x040000bc Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nLapPyrLevel
 133 ldloca.s 0 
 135 ldloc.1  
 136 callvirt 0x0600003b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefDepthMapPyrLevel
 141 stfld 0x040000bd Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nDepthMapPyrLevel
 146 ldloca.s 0 
 148 ldloc.1  
 149 callvirt 0x0600003d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefGaussFilterSizeFastEFI
 154 stfld 0x040000be Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nGaussFilterSizeFastEFI
 159 ldloca.s 0 
 161 ldloc.1  
 162 callvirt 0x0600003f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefGaussFilterSize3D
 167 stfld 0x040000bf Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nGaussFilterSize3D
 172 ldloca.s 0 
 174 ldc.i4.4  
 175 newarr 0x01000039 
 180 stloc.2  
 181 ldloc.2  
 182 ldc.i4.0  
 183 ldloc.1  
 184 callvirt 0x06000041 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlendFastEFI0
 189 stelem.r8  
 190 ldloc.2  
 191 ldc.i4.1  
 192 ldloc.1  
 193 callvirt 0x06000043 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlendFastEFI1
 198 stelem.r8  
 199 ldloc.2  
 200 ldc.i4.2  
 201 ldloc.1  
 202 callvirt 0x06000045 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlendFastEFI2
 207 stelem.r8  
 208 ldloc.2  
 209 ldc.i4.3  
 210 ldloc.1  
 211 callvirt 0x06000047 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlendFastEFI3
 216 stelem.r8  
 217 ldloc.2  
 218 stfld 0x040000c0 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBandBlendFastEFI
 223 ldloca.s 0 
 225 ldc.i4.4  
 226 newarr 0x01000039 
 231 stloc.3  
 232 ldloc.3  
 233 ldc.i4.0  
 234 ldloc.1  
 235 callvirt 0x06000049 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlend3D0
 240 stelem.r8  
 241 ldloc.3  
 242 ldc.i4.1  
 243 ldloc.1  
 244 callvirt 0x0600004b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlend3D1
 249 stelem.r8  
 250 ldloc.3  
 251 ldc.i4.2  
 252 ldloc.1  
 253 callvirt 0x0600004d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlend3D2
 258 stelem.r8  
 259 ldloc.3  
 260 ldc.i4.3  
 261 ldloc.1  
 262 callvirt 0x0600004f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBandBlend3D3
 267 stelem.r8  
 268 ldloc.3  
 269 stfld 0x040000c1 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBandBlend3D
 274 ldloca.s 0 
 276 ldloc.1  
 277 callvirt 0x06000051 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefDataNum
 282 stfld 0x040000c2 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nDataNum
 287 ldloca.s 0 
 289 ldloc.1  
 290 callvirt 0x06000053 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefDataInterpNum
 295 stfld 0x040000c3 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nDataInterpNum
 300 ldloca.s 0 
 302 ldloc.1  
 303 callvirt 0x06000055 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefLapThresh3D
 308 stfld 0x040000c4 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dLapThresh3D
 313 ldloca.s 0 
 315 ldloc.1  
 316 callvirt 0x06000057 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefLapThreshEFI
 321 stfld 0x040000c5 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dLapThreshEFI
 326 ldloca.s 0 
 328 ldloc.1  
 329 callvirt 0x06000059 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefDownSizeRateTexless
 334 stfld 0x040000c6 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dDownSizeRateTexless
 339 ldloca.s 0 
 341 ldloc.1  
 342 callvirt 0x0600005b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefGaussFilterSizeTexless
 347 stfld 0x040000c7 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nGaussFilterSizeTexless
 352 ldloca.s 0 
 354 ldloc.1  
 355 callvirt 0x0600005d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefItrMax
 360 stfld 0x040000c8 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nItrMax
 365 ldloca.s 0 
 367 ldloc.1  
 368 callvirt 0x0600005f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefSadThresh
 373 stfld 0x040000c9 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dSadThresh
 378 ldloca.s 0 
 380 ldloc.1  
 381 callvirt 0x06000061 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefGuideImgChannel
 386 stfld 0x040000ca Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nGuideImgChannel
 391 ldloca.s 0 
 393 ldloc.1  
 394 callvirt 0x06000063 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefDownSizeRateGuided
 399 stfld 0x040000cb Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dDownSizeRateGuided
 404 ldloca.s 0 
 406 ldloc.1  
 407 callvirt 0x06000065 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefWMRadius3D
 412 stfld 0x040000cc Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nWMRadius3D
 417 ldloca.s 0 
 419 ldloc.1  
 420 callvirt 0x06000067 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefWMEps3D
 425 stfld 0x040000cd Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dWMEps3D
 430 ldloca.s 0 
 432 ldloc.1  
 433 callvirt 0x06000069 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefMedFliterSize3D
 438 stfld 0x040000ce Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nMedFliterSize3D
 443 ldloca.s 0 
 445 ldloc.1  
 446 callvirt 0x0600006b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBilateralRadius3D
 451 stfld 0x040000cf Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nBilateralRadius3D
 456 ldloca.s 0 
 458 ldloc.1  
 459 callvirt 0x0600006d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBilateralEps3D
 464 stfld 0x040000d0 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBilateralEps3D
 469 ldloca.s 0 
 471 ldloc.1  
 472 callvirt 0x0600006f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefWMRadiusEFI
 477 stfld 0x040000d1 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nWMRadiusEFI
 482 ldloca.s 0 
 484 ldloc.1  
 485 callvirt 0x06000071 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefWMEpsEFI
 490 stfld 0x040000d2 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dWMEpsEFI
 495 ldloca.s 0 
 497 ldloc.1  
 498 callvirt 0x06000073 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefMedFliterSizeEFI
 503 stfld 0x040000d3 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nMedFliterSizeEFI
 508 ldloca.s 0 
 510 ldloc.1  
 511 callvirt 0x06000075 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBilateralRadiusEFI
 516 stfld 0x040000d4 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.nBilateralRadiusEFI
 521 ldloca.s 0 
 523 ldloc.1  
 524 callvirt 0x06000077 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBilateralEpsEFI
 529 stfld 0x040000d5 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBilateralEpsEFI
 534 ldloca.s 0 
 536 ldloc.1  
 537 callvirt 0x06000079 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBlendLapThreshFastSweep
 542 stfld 0x040000d6 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBlendLapThreshFastSweep
 547 ldloca.s 0 
 549 ldloc.1  
 550 callvirt 0x0600007b Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBlendMaxFastSweep
 555 stfld 0x040000d7 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBlendMaxFastSweep
 560 ldloca.s 0 
 562 ldloc.1  
 563 callvirt 0x0600007d Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBlendLapThreshFastFine
 568 stfld 0x040000d8 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBlendLapThreshFastFine
 573 ldloca.s 0 
 575 ldloc.1  
 576 callvirt 0x0600007f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefBlendMaxFastFine
 581 stfld 0x040000d9 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.dBlendMaxFastFine
 586 ldloca.s 0 
 588 ldloc.1  
 589 callvirt 0x06000081 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIDefaultParam.get_DefComputeMode
 594 stfld 0x040000da Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIParam.computeMode
 599 ldloc.0  
 600 ret  
```

