I. ������ܣ�
�������ƣ����˼ҵĵܵ�
�ӳ���飺�t�� �����ƾ���ѧ��һ
��ϵ��ʽ���ֻ���ͬ΢�ţ���15279175419        ���䣺15279175419@163.com

II. ����˼·��
����lightgbm����ctrͳ���������ı����ƶ�Ϊ��Ҫ����������������ϴ��ƽ��������ֵ������Ȩ�����������ģ��

III. ����ʵ���߼���
1. ������ϴ��ȥprefix��title��������ַ���ȥ����β�ո񣬰�%2C�滻�ɿո�
2. prefix_sum,prefix_count,prefix_ctr,title_sum,title_count,title_ctr,tag_sum,tag_count,tag_ctr,prefix_tag	_sum,prefix_title_sum,title_tag_sum,prefix_tagcount,prefix_titlecount,title_tagcount,prefix_title_ctr,prefix_tag_ctr,title_tag_ctr
prefix,title,tag���ֵĴ�����������Ĵ���������ʣ��Լ�����֮���������������
��˼·��ԴС���˿�Դ��baseline https://zhuanlan.zhihu.com/p/46482521��
3. prefix_title_tag_sum: prefix,title,tag ������Ĵ���
4. prefix_title_tag_count: prefix,title,tag ���ֵĴ���
5. prefix_title_tag_ctr = (prefix_title_tag_sum+0.1)/(prefix_title_tag_count+1) 
prefix, title, tag����ʣ�ƽ������
6. ȱֵ���: prefix_title_ctr, prefix_tag_ctr, title_tag_ctr�þ�ֵ���ȱʧֵ, ��title_tag_ctr ���ȱʧ��prefix_title_tag_ctr
7. mean_pro: query_prediction�������Ƽ�ֵ��ƽ������
8. prefix_len,title_len: prefix��title�ĳ��ȣ����Ҹ��ݶ�Ӧtag��ƽ����������Ȩ�ش���
9. max_query_len: query_prediction����������ĳ���
10. title_code: ���ݲ�ͬ���ȱ�����ĸ��ʣ���title_len���б��봦��
11. in_query_big,click8,count8,ctr8: title�Ƿ���query_prediction�����Ҹ��ʴ���10%������prefix_len,title_len,tag���������
��˼·��Դ��̳wszdc�Ŀ�Դ https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.11409386.0.0.38281d07B2cBaO&raceId=231688&postsId=34595��
12. prediction_1-9,similarity1-9,equal_rate-1-9,query_prediction�������е�ѡ���Ӧ�ĸ��ʣ���prefix_lenΪȨ�ص����ƶȣ���titleΪ���ƶ�
��˼·��ԴGrinAndBear https://github.com/GrinAndBear/OGeek/blob/master/create_feature.py��
13. Levenshtein distance ratio ƽ������
14. query_match: title��query_prediction��jaccard ���ƶȣ�����query_prediction�ĸ�����ΪȨ����ӵõ�
15. title_query_max: title��query_prediction�����jaccard���ƶ�
16. title_query_mean: title��query_prediction��jaccard���ƶȵľ�ֵ
17. title_query_rank: query_prediction����title������ƶȵĴ���������

IV. Ԥ��ѵ��+Ԥ��ʱ��
��һ�β��ԣ�4:15AM-5:05AM, 50���� ���� i5 16GB RAM, Windows10 by aimhighfly
�ڶ��β��ԣ�8:15AM-8:50AM��35���� ����i7-8700 16GB RAM,Windows10 by �t��
�����β��ԣ�8:46AM-9:41AM, 55����  ����Intel(R) Xeon(R) CPU E5-2680 32GB RAM, ubuntu by pong��
Ԥ��ѵ��+Ԥ��ʱ��Ϊ45��������

V. ���÷���
sh run.sh train.txt  vali.txt  test.txt  

run.sh:
#!/bin/sh
echo "start.."
python main.py  $1  $2 $3
echo "end."
 
  

