# jmt-bench-code

## 概要

`jmt-bench-code` は、[Mt-Bench](https://github.com/EleutherAI/mt-bench) スイートを使用して言語モデルを評価するためのベンチマークツールです。このツールは [Weights & Biases (W&B)](https://wandb.ai/) と統合されており、実験のトラッキングを行います。また、高速な推論を実現するために [vLLM](https://github.com/vllm-project/vllm) を活用し、評価モデルとして `gpt-4o-mini` を使用して対象言語モデルのパフォーマンスを評価します。

## 目次

- [前提条件](#前提条件)
- [インストール](#インストール)
- [設定](#設定)
  - [config.yaml](#configyaml)
- [認証](#認証)
  - [Weights & Biases (W&B) ログイン](#weights--biases-wb-login)
  - [Hugging Face ログイン](#hugging-face-login)
- [モデル回答の生成と評価の実行](#モデル回答の生成と評価の実行)
- [ログとモニタリング](#ログとモニタリング)
- [必要なパッケージ](#必要なパッケージ)
- [トラブルシューティング](#トラブルシューティング)
- [貢献方法](#貢献方法)
- [ライセンス](#ライセンス)
- [謝辞](#謝辞)
- [クイックリファレンス](#クイックリファレンス)

---

## 前提条件

`jmt-bench-code` をセットアップする前に、以下の項目が準備されていることを確認してください：

- **Python 3.8+**: システムにPythonがインストールされていること。 [こちら](https://www.python.org/downloads/) からダウンロード可能です。
- **CUDA (オプション)**: GPUによるモデル推論を利用する場合、CUDAがインストールされ適切に設定されていること。
- **Hugging Face アカウント**: Hugging Faceにホストされているモデルにアクセスするために必要です。
- **Weights & Biases (W&B) アカウント**: 実験のトラッキングとログのために必要です。

---

## インストール

1. **リポジトリのクローン**

   ```bash
   git clone https://github.com/your-username/jmt-bench-code.git
   cd jmt-bench-code
