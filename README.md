# topographic-ica-cifar10
CIFAR10データセットを使用した再構成型 Topographic ICAによる特徴表現の学習の可視化です。  
chainerに局所結合層（重み共有なしの畳込み層）が実装されていないようなので、全結合層で代用しました。

## 実行方法
`python train.py`  
はじめて実行した時はデータセットをダウンロードするため、しばらく時間がかかります。

## 実行結果
局所受容野層の重みを可視化したもの (200エポック)  
![](https://github.com/s059ff/topographic-ica-cifar10/blob/master/sample/kernel.png)  
