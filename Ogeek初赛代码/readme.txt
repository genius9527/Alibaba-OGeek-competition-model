I. 队伍介绍：
队伍名称：别人家的弟弟
队长简介：祎凡 江西财经大学研一
联系方式：手机（同微信）：15279175419        邮箱：15279175419@163.com

II. 解题思路：
基于lightgbm，以ctr统计特征、文本相似度为主要特征，加以数据清洗、平滑处理、插值处理、加权处理后做出的模型

III. 代码实现逻辑：
1. 数据清洗：去prefix，title里的特殊字符，去掉首尾空格，把%2C替换成空格
2. prefix_sum,prefix_count,prefix_ctr,title_sum,title_count,title_ctr,tag_sum,tag_count,tag_ctr,prefix_tag	_sum,prefix_title_sum,title_tag_sum,prefix_tagcount,prefix_titlecount,title_tagcount,prefix_title_ctr,prefix_tag_ctr,title_tag_ctr
prefix,title,tag出现的次数，被点击的次数，点击率，以及他们之间两两的组合特征
（思路来源小幸运开源的baseline https://zhuanlan.zhihu.com/p/46482521）
3. prefix_title_tag_sum: prefix,title,tag 被点击的次数
4. prefix_title_tag_count: prefix,title,tag 出现的次数
5. prefix_title_tag_ctr = (prefix_title_tag_sum+0.1)/(prefix_title_tag_count+1) 
prefix, title, tag点击率，平滑处理
6. 缺值填充: prefix_title_ctr, prefix_tag_ctr, title_tag_ctr用均值填充缺失值, 用title_tag_ctr 填充缺失的prefix_title_tag_ctr
7. mean_pro: query_prediction：各个推荐值的平均概率
8. prefix_len,title_len: prefix和title的长度，并且根据对应tag的平均长度做了权重处理
9. max_query_len: query_prediction里面概率最大的长度
10. title_code: 根据不同长度被点击的概率，对title_len进行编码处理，
11. in_query_big,click8,count8,ctr8: title是否在query_prediction里面且概率大于10%，根据prefix_len,title_len,tag的组合特征
（思路来源论坛wszdc的开源 https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.11409386.0.0.38281d07B2cBaO&raceId=231688&postsId=34595）
12. prediction_1-9,similarity1-9,equal_rate-1-9,query_prediction里面所有的选择对应的概率，与prefix_len为权重的相似度，以title为相似度
（思路来源GrinAndBear https://github.com/GrinAndBear/OGeek/blob/master/create_feature.py）
13. Levenshtein distance ratio 平滑处理
14. query_match: title与query_prediction的jaccard 相似度，并以query_prediction的概率作为权重相加得到
15. title_query_max: title与query_prediction的最大jaccard相似度
16. title_query_mean: title与query_prediction的jaccard相似度的均值
17. title_query_rank: query_prediction中与title最大相似度的词条的排名

IV. 预估训练+预测时间
第一次测试：4:15AM-5:05AM, 50分钟 配置 i5 16GB RAM, Windows10 by aimhighfly
第二次测试：8:15AM-8:50AM，35分钟 配置i7-8700 16GB RAM,Windows10 by 祎凡
第三次测试：8:46AM-9:41AM, 55分钟  配置Intel(R) Xeon(R) CPU E5-2680 32GB RAM, ubuntu by pong呐
预计训练+预测时间为45分钟左右

V. 调用方法
sh run.sh train.txt  vali.txt  test.txt  

run.sh:
#!/bin/sh
echo "start.."
python main.py  $1  $2 $3
echo "end."
 
  

