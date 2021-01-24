# <p align="center">DataCon:beers:</p>

```shell
                        _   .-')
                        ( '.( OO )_
                        ,--.      .-'),-----.  .-'),-----. ,--.   ,--.)
                        |  |.-') ( OO'  .-.  '( OO'  .-.  '|   `.'   |
                        |  | OO )/   |  | |  |/   |  | |  ||         |
                        |  |`-' |\_) |  |\|  |\_) |  |\|  ||  |'.'|  |
                        (|  '---.'  \ |  | |  |  \ |  | |  ||  |   |  |
                        |      |    `'  '-'  '   `'  '-'  '|  |   |  |
                        `------'      `-----'      `-----' `--'   `--'
```
> [DataCon2019大数据安全分析大赛](https://www.butian.net/datacon)方向二（恶意代码检测）冠军方案:rose::rose:，详细思路分享见[知乎](https://zhuanlan.zhihu.com/p/64252076)，[DataCon2020大数据安全分析大赛](https://datacon.qianxin.com/#integral)方向五（恶意代码分析）季军方案，详细思路分享见[知乎](https://zhuanlan.zhihu.com/p/185715807)，由于比赛时间仓促代码写得比较混乱，还请各位读者多多见谅！

### DataCon2019综合积分榜排名（部分）

![](https://github.com/yhangf/DataCon/blob/master/DataCon2019/useful/rank.png)

### 源码

#### stage1

- [[deep_learning_model.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/deep_learning_model.ipynb)]
- [[call_pid_tfidf_stacking.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/call_pid_tfidf_stacking.ipynb)]
- [[exinfos.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/exinfos.ipynb)]
- [[explore.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/explore.ipynb)]
- [[feature_engineering.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/feature_engineering.ipynb)]
- [[new_feature_engineering.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/new_feature_engineering.ipynb)]
- [[out_of_fold.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/out_of_fold.ipynb)]
- [[ret_value_stacking.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/ret_value_stacking.ipynb)]
- [[stacking.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/stacking.ipynb)]
- [[test.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage1/test.ipynb)]

#### stage2

- [[feature_engineering.ipynb](https://nbviewer.jupyter.org/github/yhangf/DataCon/blob/master/DataCon2019/code/stage2/feature_engineering.ipynb)]
- [[for_cluster_kmeans.py](https://github.com/yhangf/DataCon/blob/master/DataCon2019/code/stage2/for_cluster_kmeans.py)]
- [[get_call_name_tfidf_features.py](https://github.com/yhangf/DataCon/blob/master/DataCon2019/code/stage2/get_call_name_tfidf_features.py)]
- [[plot_comparison.py](https://github.com/yhangf/DataCon/blob/master/DataCon2019/code/stage2/plot_comparison.py)]
- [[yield_call_name_api_name_exinfos_tsne.py](https://github.com/yhangf/DataCon/blob/master/DataCon2019/code/stage2/yield_call_name_api_name_exinfos_tsne.py)]
- [[DBSCAN.py](https://github.com/yhangf/DataCon/blob/master/DataCon2019/code/stage2/DBSCAN.py)]

### DataCon2020综合积分榜排名（部分）

![](https://github.com/yhangf/DataCon/blob/master/DataCon2020/PPT/picture/2020rank.png)

### 源码

- [[get_id.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/get_id.py)]: 获取测试集的文件名
- [[get_raw_test_data.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/get_raw_test_data.py)]: 获取测试集的原始字符串
- [[get_raw_train_data.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/get_raw_train_data.py)]: 获取训练集的原始字符串
- [[test_train_model.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/test_train_model.py)]: 测试训练的模型
- [[yield_end_result.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/yield_end_result.py)]: 生成最终提交的结果
- [[yield_features.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/yield_features.py)]: 由原始字符串生成特征矩阵
- [[yield_train_model.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/yield_train_model.py)]: 生成训练模型
- [[plot.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/plot.py)]: 绘图模块
- [[t_sne.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/t_sne.py)]: 降维可视化模块
- [[lgb_cv.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/lgb_cv.py)]: LightGBM模型+交叉验证
- [[xgb_bagging.py](https://github.com/yhangf/DataCon/blob/master/DataCon2020/codes/xgb_bagging.py)]: XGBoost模型+Bagging
