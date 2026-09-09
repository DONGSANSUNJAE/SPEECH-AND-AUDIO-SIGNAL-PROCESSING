# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: package desktop_app.py as a standalone AudioLab.app.

Only the pywebview launcher is frozen. Streamlit/librosa/etc. stay unfrozen
and keep running from .venv via subprocess, exactly as in dev mode — this
avoids the headaches of freezing the whole Streamlit stack. Freezing the
launcher alone is what matters: it gives the app its own Info.plist, so
macOS can attribute microphone permission to "AudioLab" instead of to the
shared system Python framework.
"""

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AudioLab',
)

app = BUNDLE(
    coll,
    name='AudioLab.app',
    icon=None,
    bundle_identifier='kr.ac.yonsei.audiolab',
    info_plist={
        'NSMicrophoneUsageDescription': '말한 소리를 녹음해 파형과 스펙트로그램을 분석합니다.',
        'CFBundleDisplayName': '음성 분석',
        'CFBundleName': '음성 분석',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '13.0',
    },
)
