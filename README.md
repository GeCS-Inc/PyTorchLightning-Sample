# PyTorch Lightning 画像分類サンプル

## Environment

- Python: 3.6
- PyTorch: 1.5
- PyTorch-Lightning: 0.9

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

### その他設定

**env**

`batch_size`: 一度に学習を行うデータの数です。GPU のメモリサイズに合わせて調整してください。

**transform**

`Resize`: アスペクト比を保ったまま画像サイズを変更します。縦横の小さい方を指定したサイズに合わせます。
`RandomCrop`: 正方形の画像にランダムにクロップします。指定したサイズでクロップします。

## Inference

推測させたい画像を `inference.py` の `IMAGE_PATH` で指定します。
そして次のコマンドで画像の分類を行います。

```sh
$ python3 inference.py
```
