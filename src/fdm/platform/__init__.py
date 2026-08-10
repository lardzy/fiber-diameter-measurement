"""Platform-specific adapters used by optional desktop integrations.

Modules in this package must remain safe to import on non-native platforms.
Concrete Windows APIs are therefore created lazily and accept injectable
adapters for deterministic tests.
"""
