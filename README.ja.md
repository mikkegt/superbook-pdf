# superbook-pdf

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests: 38 passed](https://img.shields.io/badge/Tests-38%20passed-brightgreen.svg)](tests/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20(tested)-000000.svg?logo=apple&logoColor=white)]()

書籍PDF処理ツール: 傾き補正、色調整、AI超解像、OCR。

[DN_SuperBook_PDF_Converter](https://github.com/mikkegt/DN_SuperBook_PDF_Converter)（C# / .NET 6.0 / Windows専用）のPython書き換え版。**macOS（Apple Silicon）** で開発・テスト済み。**Linux** ・ **Windows** でも動作する見込みです（未検証）。

### superbook-pdf の特徴

- **Pure Python** — Cコンパイラや Cython のビルド不要。`uv sync` だけで準備完了
- **簡単インストール** — Homebrew/apt/choco でシステムツール、uv で Python 依存関係。手動ファイル配置不要
- **PDF レイアウト設定** — 1ページ/見開き、左綴じ/右綴じを CLI で指定。教科書にも漫画にも対応
- **38 ユニットテスト** — 信頼性の高いテスト済みコードベース

## 機能

- **PDF→画像抽出**（ImageMagick、300dpi）
- **エッジトリミング**（0.5%のスキャン端除去）
- **AI超解像**（RealESRGAN 2倍、CUDA/MPS/CPU対応）
- **傾き補正**（ImageMagickで角度検出 + OpenCVで回転）
- **色補正**（線形スケーリング、Smoothstepホワイトニング、ゴースト抑制）
- **テキスト領域検出**（Otsu二値化 + 輪郭解析）
- **IQRクロップ領域**（Tukeyフェンスによる外れ値除去）
- **グラデーション紙色パディング**（コーナーサンプリングによるバイリニア背景）
- **ページ番号OCR**（Tesseract）
- **日本語OCR**（YomiToku、オプション）
- **PDFレイアウト設定**（1ページ/見開き、左綴じ/右綴じ）

## 必要環境

### システムツール

**macOS**（Homebrew）:

```bash
brew install imagemagick ghostscript exiftool qpdf pdfcpu tesseract tesseract-lang
```

**Ubuntu / Debian**:

```bash
sudo apt install imagemagick ghostscript libimage-exiftool-perl qpdf tesseract-ocr tesseract-ocr-jpn
# pdfcpu: https://github.com/pdfcpu/pdfcpu/releases からダウンロード
```

**Windows**（Chocolatey）:

```powershell
choco install imagemagick ghostscript exiftool qpdf tesseract
# pdfcpu: https://github.com/pdfcpu/pdfcpu/releases からダウンロード
```

### Python

Python 3.10以上 + [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

### AI超解像（オプション）

```bash
pip install torch realesrgan basicsr
```

GPU加速: CUDA（NVIDIA、Linux/Windows）、MPS（Apple Silicon）、またはCPUフォールバック。

### 日本語OCR（オプション）

```bash
pip install yomitoku
```

## 使い方

```bash
# 基本的な使い方（AI超解像なし）
superbook input.pdf output.pdf --skip-upscale

# AI超解像あり
superbook input.pdf output.pdf

# デバッグ画像あり、ページ数制限
superbook input.pdf output.pdf --max-pages 10 --save-debug -v

# OCRあり
superbook input.pdf output.pdf --ocr

# 漫画（右綴じ、見開き）
superbook input.pdf output.pdf --binding right --layout spread

# 教科書（左綴じ、1ページ表示）
superbook input.pdf output.pdf --binding left --layout single
```

### CLIオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--margin N` | `7` | テキスト領域周囲のマージン（%） |
| `--max-pages N` | 全ページ | 処理ページ数を制限 |
| `--skip-upscale` | オフ | RealESRGAN超解像をスキップ |
| `--ocr` | オフ | ページ番号OCR + YomiTokuを有効化 |
| `--save-debug` | オフ | 中間画像を保存 |
| `--layout [single\|spread]` | `single` | ページレイアウト: 1ページ / 見開き |
| `--binding [left\|right]` | `left` | 綴じ方向: 左綴じ（L2R）/ 右綴じ（R2L） |
| `-v, --verbose` | オフ | デバッグログ出力 |

### レイアウト設定ガイド

| 用途 | `--layout` | `--binding` |
|------|-----------|-------------|
| 教科書（横書き） | `single` | `left` |
| 教科書（見開き） | `spread` | `left` |
| 漫画（縦書き・右から左） | `spread` | `right` |
| 漫画（1ページ） | `single` | `right` |

## 処理パイプライン

```
ステップ1: PDF → 画像（ImageMagick、300dpi）
ステップ2: エッジトリミング（0.5%マージン除去）
ステップ3: AI超解像（RealESRGAN 2倍）
ステップ4: ページ処理
  4-1: ページリスト初期化（奇数/偶数分離）
  4-2: 傾き補正 + 色統計
  4-3: グローバル色パラメータ算出（MAD外れ値除去）
  4-4: 色補正 + テキスト領域検出
  4-5: ページ番号OCR（Tesseract、オプション）
  4-6: クロップ領域決定（IQR外れ値検出）
  4-7: 最終出力（リサイズ + グラデーションパディング）
ステップ5: PDF再構築（ImageMagick → ExifTool → pdfcpu）
ステップ6: 日本語OCR（YomiToku、オプション）
```

## 開発

```bash
# dev依存関係をインストール
uv sync --group dev

# テスト実行
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## ライセンス

AGPL-3.0-only（元プロジェクトに準拠）

## クレジット

[DN_SuperBook_PDF_Converter](https://github.com/mikkegt/DN_SuperBook_PDF_Converter)（mikkegt）をベースにしています。
