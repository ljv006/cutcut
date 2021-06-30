<!--
 * @Author: 龙嘉伟
 * @Date: 2021-06-30 15:35:33
 * @LastEditors: 龙嘉伟
 * @LastEditTime: 2021-06-30 15:41:06
 * @Description: 
-->
# cutcut通用分词工具
### 在开源数据上使用albert进行实体识别，切分句子，得到对应的单词序列。
## 更新说明
### 2021-06-30
- 基本完成分词功能，后期需要增加自定义词典及自定义词添加。
- 将模型打包成wheel格式，使用pip进行安装。

## 使用说明
  1. 调用get_wheel.sh生成安装文件，在dist目录下；
  2. 使用pip install XXX.whl文件；
  3. 在python中使用import cutcut引入分词包；
  4. 使用cutcut.lcut进行分词。