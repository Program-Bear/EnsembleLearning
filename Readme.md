姓名：魏钧宇
学号：2015011263

提交目录下的包括一个报告和四个文件夹：
data：最初给的训练集和测试集
dic：通过特征提取获得字典
src：源代码
result：最终的预测结果

运行方法：
在命令行中执行

python3 src/ensemble.py --help
你可以看到运行所需的所有必须选项和可选选项
必须选项包括：
字典路径：dic/TreeDic.txt（对应使用1163个特征） 或者 dic/dic.txt（对应使用1w+特征）
训练集：data/exp2.train.csv
测试集：data/exp2.validation_review.csv
输出路径：result/result.csv（最终的预测结果就在Result文件夹下的result.csv文件中）

可选的选项及其默认设置请参见 --help中的说明，默认情况对应于在Kaggle上的最好提交结果（使用全特征集的50轮的Bagging决策树）

