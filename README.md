# PyTorch Lightning 画像分類サンプル

## Environment

- Python: 3.6
- PyTorch: 1.5
- PyTorch-Lightning: 0.9
- OpenCV 4.2.0

## Training

データセットを次のように用意します。この場合、`dog` と `cat` の 2 つがクラスとなります。

```
{folder}/dog/xxx.png
{folder}/dog/xxy.png
{folder}/dog/xxz.png

{folder}/cat/123.png
{folder}/cat/nsdf3.png
{folder}/cat/asd932_.png
```

`train.py` の `env` の内容を書き換えます。`dataset_root` に先程の `{folder}` の部分のパスを指定してください。
また、`num_class` には先程のクラスの数を指定してください。

次のコマンドで学習を開始します。

```sh
$ python3 train.py
```

コマンドを実行すると、ルートディレクトリに `dataset-sample.png` というファイルが作成されています。
ここでモデルに入力されるデータを確認することができます。意図したどおりに画像が変形されているかを確認してください。
もし意図した通りになっていない場合は `transform` の内容を変更してみてください。

注意：データセットの数が少ない場合は余ったマスが黒になります。

### その他設定

**env**

`base_model`: 使用する学習済みモデルです。`model.py` の `MODELS` に使用可能なモデルの一覧があるので、そこから選んでください。
`batch_size`: 一度に学習を行うデータの数です。GPU のメモリサイズに合わせて調整してください。

**transform**

`Resize`: アスペクト比を保ったまま画像サイズを変更します。縦横の小さい方を指定したサイズに合わせます。
`RandomCrop`: 正方形の画像にランダムにクロップします。指定したサイズでクロップします。

### 学習済みモデルについて

現在 `VGG16`, `ResNet`, `EfficientNet` の3つを用意しています。一般的にモデルの規模は `VGG16`, `ResNet`, `EfficientNet` の順番で大きくなっています。
ただし、データセットが小さい場合モデルの規模が大きくなると、過学習が発生するなど学習が失敗する可能性も生じます。なので、それぞれを試してから使用するモデルを選ぶのが良いです。

## Inference

推測させたい画像を `inference.py` の `IMAGE_PATH` で指定します。
そして次のコマンドで画像の分類を行います。

```sh
$ python3 inference.py
```

## Issue

```
CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)` (createCublasHandle at /opt/conda/conda-bld/pytorch_1591914838379/work/aten/src/ATen/cuda/CublasHandlePool.cpp:8)
TypeError: 'NoneType' object is not iterable
```

`num_class` が間違っている可能性があります。