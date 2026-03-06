# Npind(numpy-parallel-indexer)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange)](https://semver.org/)

**Npind** は、[NumPy](https://numpy.org/ja/) のインデックス操作を高速化するライブラリです。
複雑なインデックス操作は Numpy を用いた処理のボトルネックとなりえます。
Npind が開発された理由は、このボトルネックを緩和し Numpy の能力をより引き出すためです。

Npind が高速である理由は、[Numba](https://numba.pydata.org/) により並列処理を行っているためです。
Npind は、Numba の高速な for ループをラップし、Numpy 互換の簡潔な API を提供します。
Numba は、Numba で未だサポートされていない Numpy メソッドを実装します(特に```numpy.take``` の ```out``` 引数！)。

## 1. Npind の特徴

- **NumPy 互換**: NumPy の簡潔な API を引き継ぐ。
- **最小限の依存関係**: Pythonコードのみで構成される。依存ライブラリは Numpy と Numba のみ。
- **低オーバーヘッド**: インプレース演算をサポートし、動的メモリ確保を最小限抑える。
- **インデキシングに特化**: インデックス操作に注力することで、パフォーマンスを追求する。

## 2. インストール

インストール:

```bash
git clone https://github.com/ShibanGon/npind.git
cd npind
pip install -e ".[dev]"
``` 

アンインストール

```bash
pip uninstall npind
```

## 3. 使い方

``` python
import numpy as np
import npind as npi

a = np.random.rand(10000, 10000)
indices = [0, 500, 999]

result = npi.take(a, indices, axis=1)
```
### 3.1. ベンチマーク結果

npind は、特に大規模データにおいて標準の NumPy や素の Numba 実装を上回るパフォーマンスを発揮します 。詳細な計測条件は [benchmark.py](./benchmarks/benchmark_take.py) を参照してください 。

<img src="./assets/benchmark_result.png" width="1000pt">

## 4. 開発ロードマップと実装状況

現在、以下の関数の並列実装および `out` 引数による最適化を順次進めています。

[Indexing routines](https://numpy.org/doc/stable/reference/routines.indexing.html)

| メソッド名      | 開発ステータス |
| :-------------- | -------------: |
| take            | `@njit` 対応中 |
| take_along_axis |                |
| choose          |                |
| compress        |                |
| select          |                |
| place           |                |
| put             |                |
| putmask         |                |

[Sort, search, and count](https://numpy.org/doc/stable/reference/routines.sort.html)


| メソッド名    | 開発ステータス |
| :------------ | -------------: |
| argsort       |                |
| partition     |                |
| argpartition  |                |
| argwhere      |                |
| nonzero       |                |
| flatnonzero   |                |
| where         |                |
| searchsorted  |                |
| extract       |                |
| count_nonzero |                |

[Statistics](https://numpy.org/doc/stable/reference/routines.statistics.html)

| メソッド名 | 開発ステータス |
| :--------- | -------------: |
| digitize   |                |

[Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html)
| メソッド名                    | 開発ステータス |
| :---------------------------- | -------------: |
| numpy.ndarray.\_\_getitem\_\_ |                |


### 4.1. 現在の制限事項 (Current Limitations)

- **Numba 内での直接呼び出し**: 現時点では `npi.take` を他の `@njit` 関数の中から直接呼び出すことはできません。将来のアップデートで `numba.extending.overload` への対応を予定しています。
- **データ型**: 現在は数値型の配列を主にサポートしています。オブジェクト型の配列には対応していません。

## 5. ライセンス
MIT License