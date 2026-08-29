# 离线分割引擎包格式

首选项中的离线引擎管理器只负责导入、完整性检查和 CPU 诊断，不会注册新的分割工具。引擎目录或 ZIP 根目录需要包含 `engine.json`（也接受 `manifest.json`）。

```json
{
  "kind": "fdm.offline_segmentation_engine",
  "schema_version": 1,
  "engine_id": "sam3",
  "display_name": "SAM3 CPU 离线包",
  "version": "1.0.0",
  "device": "cpu",
  "python": "runtime/python.exe",
  "diagnostic": ["@scripts/diagnose.py"],
  "resources": [
    {
      "path": "models/model.bin",
      "sha256": "填写 64 位小写 SHA-256",
      "required": true
    }
  ]
}
```

- `engine_id` 目前只接受 `sam3` 或 `micro_sam`。
- `python`、以 `@` 开头的诊断参数及资源路径必须位于包目录内。
- 每个已列出的资源都必须提供 SHA-256；导入和每次诊断前都会重新核验。
- 诊断强制设置 CPU 与离线环境变量，仅在用户点击“运行 CPU 诊断”时执行包内命令。
- “关联现有目录”不会复制文件；“导入 ZIP”会将验证后的内容复制到应用设置目录。
- “移除配置”只修改首选项草稿；“删除包文件”仅对软件托管包可用，并会在二次确认后立即释放磁盘空间。
