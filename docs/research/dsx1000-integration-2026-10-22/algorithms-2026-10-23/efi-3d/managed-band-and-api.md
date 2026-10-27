## Managed band threshold defaults and P/Invoke signatures
Source: `ev_379de98b66523eeb7bc419eabf4a809f30e787c4bec9fb63d97cafd76c9bb6b2`

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_GetMemorySize 0x06000009
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 3, 'parameter_types': ['valuetype:72', 'valuetype:80', 'valuetype:68&'], 'parse_status': 'decoded', 'raw_length': 11, 'raw_sha256': 'f39687337f5359eb5d071ce771abe86259f0ba4c739b45f602f5b00580c11a57', 'resolved_sha256': 'b84f6be5c383d46505068353ca09252a2623f69a8519f4ecc5b444cb1960f9a5', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 call 0x06000011 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_GetMemorySize
   8 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_CalcCacheImageNum 0x0600000a
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 5, 'parameter_types': ['valuetype:72', 'valuetype:80', 'i8', 'i8', 'i4&'], 'parse_status': 'decoded', 'raw_length': 12, 'raw_sha256': '200411932ff2f0aa0f26b7ff504f1bcb08ec3f930d20dc440db046554af385f5', 'resolved_sha256': '7713fa82d044bb4ad76a68016c536070e511b97ad815a79870dfcc8f775848cc', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 ldarg.3  
   4 ldarg.s 4 
   6 call 0x06000012 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_CalcCacheImageNum
  11 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_CreateContext 0x0600000b
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 8, 'parameter_types': ['valuetype:72', 'valuetype:80', 'native-int', 'i8', 'native-int', 'i8', 'native-int&', 'native-int&'], 'parse_status': 'decoded', 'raw_length': 16, 'raw_sha256': 'e6d9bf62c012b6f66ed5ce86841ac31d038bc405d83b8ff64749a7a3cbdaad6c', 'resolved_sha256': '1134766056a4078105db54ee1a0b62eff8974f002ec99c0286f3303fed93006a', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 ldarg.3  
   4 ldarg.s 4 
   6 ldarg.s 5 
   8 ldarg.s 6 
  10 ldarg.s 7 
  12 call 0x06000013 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_CreateContext
  17 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_ReleaseContext 0x0600000c
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 2, 'parameter_types': ['native-int', 'native-int'], 'parse_status': 'decoded', 'raw_length': 5, 'raw_sha256': '6674fcf283f7478835feb66119c947f579af066e11c9bdf837672b35f01120e2', 'resolved_sha256': '6674fcf283f7478835feb66119c947f579af066e11c9bdf837672b35f01120e2', 'return_type': 'void'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 call 0x06000014 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_ReleaseContext
   7 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_InputImage 0x0600000d
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 5, 'parameter_types': ['native-int', 'native-int', 'native-int', 'r4', 'native-int'], 'parse_status': 'decoded', 'raw_length': 9, 'raw_sha256': 'b473ab5ef1da0e19c81803a0c6983153f6016b4cd6716aeb04f8c2acaeecc7df', 'resolved_sha256': '7d4d1f69580f55abfaa5a366327368246a3a1939f16b718d39429f8ab82c168f', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 ldarg.3  
   4 ldarg.s 4 
   6 call 0x06000015 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_InputImage
  11 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_Construct 0x0600000e
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 6, 'parameter_types': ['native-int', 'native-int', 'native-int', 'r4[]', 'class:8', 'native-int'], 'parse_status': 'decoded', 'raw_length': 12, 'raw_sha256': '6670175aa8a0943fa81fb618f81a35190507808e4b93f5552d9b74873f3b961f', 'resolved_sha256': '7367bb3f8e90f6316f3a75ccc36e2e85a01e4be44f4aeb0b4edfc296ea86881e', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 ldarg.3  
   4 ldarg.s 4 
   6 ldarg.s 5 
   8 call 0x06000016 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_Construct
  13 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_InputImageForConstruct 0x0600000f
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 4, 'parameter_types': ['native-int', 'native-int', 'i4', 'native-int[]'], 'parse_status': 'decoded', 'raw_length': 9, 'raw_sha256': '7c053af4246217d6867253fcb4708b2ee70fee653e8aff080534af595179b3d0', 'resolved_sha256': '3e9efef26ee513bd60b1f2df629dee1b01498046da192fa63deb816334e732ff', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 ldarg.3  
   4 call 0x06000017 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_InputImageForConstruct
   9 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMV.EFIMV_EndConstruct 0x06000010
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 4, 'parameter_types': ['native-int', 'native-int', 'native-int', 'r4[]'], 'parse_status': 'decoded', 'raw_length': 9, 'raw_sha256': 'e01c4b2e1ec89c9b45024be80fd4e050a05b38f12f98d920c0a6b32cf7b3b6b5', 'resolved_sha256': '049e189dc5df6f97ed3c090123ab228af9b4593764d2dca7d3de3b722fb14543', 'return_type': 'valuetype:16'}
```
   0 ldarg.0  
   1 ldarg.1  
   2 ldarg.2  
   3 ldarg.3  
   4 call 0x06000018 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_EndConstruct
   9 ret  
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_GetMemorySize 0x06000011
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 3, 'parameter_types': ['valuetype:72', 'valuetype:80', 'valuetype:68&'], 'parse_status': 'decoded', 'raw_length': 10, 'raw_sha256': '79e5e11676d00fdf7c868224b0af99fd9d9d1527280fb3f00fc8e1a4a43bd8f1', 'resolved_sha256': '7478ee7114e4b41f7f741e54a2d713f7eaca2241da53983895f18941269e0b78', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_CalcCacheImageNum 0x06000012
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 5, 'parameter_types': ['valuetype:72', 'valuetype:80', 'i8', 'i8', 'i4&'], 'parse_status': 'decoded', 'raw_length': 11, 'raw_sha256': '7c81dad352747849b12826d18cb04329f66c8bdab876c83a174f2f7d4533b8ba', 'resolved_sha256': '8e076abc6c1932c4a0231a7c63e44ce8cc480b411b51bfae0c9be48ee176fa42', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_CreateContext 0x06000013
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 8, 'parameter_types': ['valuetype:72', 'valuetype:80', 'native-int', 'i8', 'native-int', 'i8', 'native-int&', 'native-int&'], 'parse_status': 'decoded', 'raw_length': 15, 'raw_sha256': 'b8be878906aa16f28587d73265916daa5d68e2909791b740ea8ef3ca77f1b5f6', 'resolved_sha256': '18bd5c2761f09ce467a7d5333b6fcd81b65cf0c5f3352cda085bba5fc7c7e62c', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_ReleaseContext 0x06000014
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 2, 'parameter_types': ['native-int', 'native-int'], 'parse_status': 'decoded', 'raw_length': 5, 'raw_sha256': '6674fcf283f7478835feb66119c947f579af066e11c9bdf837672b35f01120e2', 'resolved_sha256': '6674fcf283f7478835feb66119c947f579af066e11c9bdf837672b35f01120e2', 'return_type': 'void'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_InputImage 0x06000015
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 5, 'parameter_types': ['native-int', 'native-int', 'native-int', 'r4', 'native-int'], 'parse_status': 'decoded', 'raw_length': 8, 'raw_sha256': '83e84728a8b434373c2cd00caef6f062b0355cd2c2f97faac9aa8ce4e64ae794', 'resolved_sha256': '83e84728a8b434373c2cd00caef6f062b0355cd2c2f97faac9aa8ce4e64ae794', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_Construct 0x06000016
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 6, 'parameter_types': ['native-int', 'native-int', 'native-int', 'r4[]', 'class:8', 'native-int'], 'parse_status': 'decoded', 'raw_length': 11, 'raw_sha256': '8677c5914b9fb61854b48180c2b6b1ab9d45da5f3ad9650ddb9629dd8b87f4d5', 'resolved_sha256': '6af691413074a81506de36ffd44a0ccdcb06d4bfe4f76dc9a55e299bb13364fa', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_InputImageForConstruct 0x06000017
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 4, 'parameter_types': ['native-int', 'native-int', 'i4', 'native-int[]'], 'parse_status': 'decoded', 'raw_length': 8, 'raw_sha256': 'dcb265497a06bc8f2573bd05a5136157076b3ea70a4c0fb526fe0074c195fb22', 'resolved_sha256': 'dcb265497a06bc8f2573bd05a5136157076b3ea70a4c0fb526fe0074c195fb22', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIMVInvoke.EFIMV_EndConstruct 0x06000018
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 4, 'parameter_types': ['native-int', 'native-int', 'native-int', 'r4[]'], 'parse_status': 'decoded', 'raw_length': 8, 'raw_sha256': '1f94d461aac59065d9a1e15948611b9d92b4ce27ae23d566c1a745210e49b8c3', 'resolved_sha256': '1f94d461aac59065d9a1e15948611b9d92b4ce27ae23d566c1a745210e49b8c3', 'return_type': 'i4'}
```
```

### Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.GetDefaultParam 0x06000025
{'calling_convention': 'default', 'field_type': None, 'generic_parameter_count': None, 'issue': None, 'kind': 'method', 'parameter_count': 2, 'parameter_types': ['string', 'valuetype:52'], 'parse_status': 'decoded', 'raw_length': 7, 'raw_sha256': 'f78cf97f0203f923501eaae7b15c104fa15221627aa1a62db630feab607bef3d', 'resolved_sha256': '84442482eda7d57682b6ff2a8c70f238827dc1a8703e0c25fe00609488a3dfe1', 'return_type': 'class:48'}
```
   0 ldarg.0  
   1 ldsfld 0x0400005f Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam._BAND_BLEND_PARAM_PATH
   6 ldsfld 0x04000060 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam._BAND_BLEND_FILE_NAME
  11 call 0x0a000044 .Concat
  16 stloc.0  
  17 ldsfld 0x0a000045 .Empty
  22 stloc.1  
  23 newobj 0x06000026 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam..ctor
  28 stloc.2  
  29 ldarg.1  
  30 brtrue.s 130 
  32 ldsfld 0x04000061 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam._BAND_BLEND_SECTION_3D
  37 stloc.1  
  38 ldloc.2  
  39 ldc.r8 70 
  48 callvirt 0x0600001a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefFistLow
  53 ldloc.2  
  54 ldc.r8 35 
  63 callvirt 0x0600001c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefSecondHigh
  68 ldloc.2  
  69 ldc.r8 20 
  78 callvirt 0x0600001e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefSecondLow
  83 ldloc.2  
  84 ldc.r8 15 
  93 callvirt 0x06000020 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefThirdHigh
  98 ldloc.2  
  99 ldc.r8 10 
 108 callvirt 0x06000022 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefThirdLow
 113 ldloc.2  
 114 ldc.r8 0 
 123 callvirt 0x06000024 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefFourthHigh
 128 br.s 226 
 130 ldsfld 0x04000062 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam._BAND_BLEND_SECTION_EFI
 135 stloc.1  
 136 ldloc.2  
 137 ldc.r8 35 
 146 callvirt 0x0600001a Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefFistLow
 151 ldloc.2  
 152 ldc.r8 20 
 161 callvirt 0x0600001c Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefSecondHigh
 166 ldloc.2  
 167 ldc.r8 15 
 176 callvirt 0x0600001e Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefSecondLow
 181 ldloc.2  
 182 ldc.r8 10 
 191 callvirt 0x06000020 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefThirdHigh
 196 ldloc.2  
 197 ldc.r8 5 
 206 callvirt 0x06000022 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefThirdLow
 211 ldloc.2  
 212 ldc.r8 3 
 221 callvirt 0x06000024 Olympus.LI.Triumph.Image.ExtendFocusImageSGSharp.EFIBandBlendParam.set_DefFourthHigh
 226 ldloc.2  
 227 ldloc.0  
 228 ldloc.1  
 229 call 0x2b000003 
 234 ldloc.2  
 235 ret  
```
