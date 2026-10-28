# Recruit Assignment for Physical AI Research Intern

## 1. 概要

フィジカル AI の研究開発では、モデルや学習コードを自分で一から書くことよりも、
**既存の実装が理論どおりに動いているかを読み解き、得られた結果が本当に正しいかを確かめる**ことが重要になる場面が多くあります。

本課題では、2 次元平面上を点が移動するシンプルな環境を題材に、学習済みのモデル・エージェントを評価していただきます。
新しいモデルを実装することは求めていません。与えられたコードと学習結果を調べ、何が起きているのかを根拠とともに説明してください。

## 2. 課題

課題は 2 つあります。それぞれのディレクトリの README に詳細を記載しています。

| 課題 | ディレクトリ | 内容 |
| --- | --- | --- |
| 課題 1 | [`task1_imitation/`](task1_imitation/README.md) | 模倣学習 (CNN / Diffusion Transformer) で学習したモデルの評価 |
| 課題 2 | [`task2_rl/`](task2_rl/README.md) | 強化学習 (PPO) で学習したエージェントの評価 |

課題 1 と課題 2 は独立しているため、どちらから取り組んでも構いません。

## 3. 課題の進め方

### 3.1. 環境

- Python 3.10 以上
- パッケージ管理には [uv](https://docs.astral.sh/uv/) を使用します
- 必要なライブラリ (`numpy`, `torch`, `matplotlib`) は `pyproject.toml` に記載しています
- その他、必要に応じてライブラリを追加していただいて構いません
- GPU は不要です。すべて CPU で実行できます

### 3.2. セットアップ

```bash
$ git clone https://github.com/hacarus/physical-ai-recruit-assignment.git
$ cd physical-ai-recruit-assignment
$ uv sync
```

動作確認として、以下を実行してください。

```bash
$ cd task1_imitation
$ uv run python rollout_example.py --ckpt checkpoints/cnn_straight.pt --layout straight
```

## 4. 提出方法と注意点

### 提出方法

- 提出形式は自由です (Markdown, PDF, Jupyter Notebook, リポジトリの URL 等)。**英語で書いても構いません**
- 以下を含めて提出してください
  - 各課題の回答
  - 評価・調査のために作成したコードと図

### 注意点

- すべての課題を満たせなかった場合でも、そのことを明記したうえで提出してください
- 与えられたコードに問題があると考えた場合は、その旨と根拠を記述してください
- AI ツール (ChatGPT, Claude, GitHub Copilot 等) の使用は自由です。ただし、**使用した AI ツールはすべて明記してください**
  - 提出された内容については面談で質問します。AI ツールを使った部分も含め、ご自身で説明できるようにしてください
