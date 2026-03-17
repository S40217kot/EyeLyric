# EyeLyric – Gaze Synthesizer Prototype

## What is this?
視線だけで音程を連続的に操作できる視線シンセサイザーのプロトタイプです。

## How it works
- MediaPipe FaceMeshで視線推定
- 視線X → Pitch
- 視線Y → Volume
- ノイズ低減処理あり

## Status
- 視線テルミンとして演奏可能
- 今後：音色・Unity・高精度視線推定へ拡張予定
