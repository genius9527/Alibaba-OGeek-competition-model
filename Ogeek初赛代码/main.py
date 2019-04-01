# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Levenshtein
import difflib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import re
import matplotlib.pylab as plt
import json

import sys
print(os.listdir())
print(sys.argv[0])
print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])

train_data = pd.read_table(sys.argv[1],
                           names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8',
                           quoting=3).astype(str)
val_data = pd.read_table(sys.argv[2],
                         names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None,
                         encoding='utf-8').astype(str)
test_data = pd.read_table(sys.argv[3],
                          names=['prefix', 'query_prediction', 'title', 'tag'], header=None, encoding='utf-8').astype(str)

train_data = train_data[train_data['query_prediction'] != 'nan']

train_data = train_data[train_data['label'] != '音乐']
test_data['label'] = -1

print(1)


def char_cleaner(char):
    new_char = re.sub('%2C', ' ', char)
    pattern = re.compile("[^\s\.a-zA-Z0-9\u4E00-\u9FA5]")
    new_char = re.sub(pattern, "", new_char)
    new_char = new_char.lower()
    return new_char if new_char else char


train_data['prefix'] = train_data['prefix'].apply(lambda x: char_cleaner(str(x).strip()))
train_data['title'] = train_data['title'].apply(lambda x: char_cleaner(str(x).strip()))
train_data['tag'] = train_data['tag'].apply(lambda x: char_cleaner(str(x).strip()))

val_data['prefix'] = val_data['prefix'].apply(lambda x: char_cleaner(str(x).strip()))
val_data['title'] = val_data['title'].apply(lambda x: char_cleaner(str(x).strip()))
val_data['tag'] = val_data['tag'].apply(lambda x: char_cleaner(str(x).strip()))

test_data['prefix'] = test_data['prefix'].apply(lambda x: char_cleaner(str(x).strip()))
test_data['title'] = test_data['title'].apply(lambda x: char_cleaner(str(x).strip()))
test_data['tag'] = test_data['tag'].apply(lambda x: char_cleaner(str(x).strip()))

train_data['label'] = train_data['label'].apply(lambda x: int(x))
val_data['label'] = val_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))


print(2)

items = ['prefix', 'title', 'tag']
'''
小幸运dalao的baseline里的
组合点击率
'''

for item in items:
    temp = train_data.groupby(item, as_index=False)['label'].agg({item + '_click': 'sum', item + '_count': 'count'})
    temp[item + '_ctr'] = temp[item + '_click'] / (temp[item + '_count'])
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')
    val_data = pd.merge(val_data, temp, on=item, how='left')

for i in range(len(items)):
    for j in range(i + 1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg(
            {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
        temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (temp['_'.join(item_g) + '_count'] + 3)
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')
        val_data = pd.merge(val_data, temp, on=item_g, how='left')
        test_data['_'.join(item_g) + '_ctr'].fillna(test_data['_'.join(item_g) + '_ctr'].mean() * 0.68)
        val_data['_'.join(item_g) + '_ctr'].fillna(val_data['_'.join(item_g) + '_ctr'].mean() * 0.68)

temp = train_data.groupby(items, as_index=False)['label'].agg(
    {'_'.join(items) + '_click': 'sum', '_'.join(items) + '_count': 'count'})
temp['_'.join(items) + '_ctr'] = (temp['_'.join(items) + '_click'] + 0.1) / (temp['_'.join(items) + '_count'] + 1)
# temp['_'.join(items)+'_ctr2'] = (temp['_'.join(items)+'_count'])/(temp['_'.join(items)+'_click']+0.1)
train_data = pd.merge(train_data, temp, on=items, how='left')
# train_data['prefix_title_tag_ctr'] cleaning
# train_data['prefix_title_tag_ctr'] = train_data.apply(lambda x: x['prefix_title_tag_ctr'] if x['prefix_title_tag_count'] > 0
#                                                       else -1,axis =1)
val_data = pd.merge(val_data, temp, on=items, how='left')
test_data = pd.merge(test_data, temp, on=items, how='left')

# fill NA
test_data['prefix_title_tag_ctr'] = test_data.apply(
    lambda x: x['prefix_title_tag_ctr'] if x['prefix_title_tag_count'] > 0
    else x['title_tag_ctr'], axis=1)

val_data['prefix_title_tag_ctr'] = val_data.apply(lambda x: x['prefix_title_tag_ctr'] if x['prefix_title_tag_count'] > 0
else x['title_tag_ctr'], axis=1)

pattern1 = r'0\.\d+'
pat1 = re.compile(pattern1)


def find_pro(x):
    pro = re.findall(pat1, x)
    if pro == []:
        return np.nan
    else:
        return np.mean(np.array(pro).astype(float))


train_data['mean_pro'] = train_data.query_prediction.apply(find_pro)  # 提取query_prediction里所有的概率数值然后计算平均值
val_data['mean_pro'] = val_data.query_prediction.apply(find_pro)
test_data['mean_pro'] = test_data.query_prediction.apply(find_pro)


def return_len(x):
    try:
        return len(x)
    except:
        return len(str(x))


train_data['title_len'] = train_data.title.apply(return_len)
val_data['title_len'] = val_data.title.apply(return_len)
test_data['title_len'] = test_data.title.apply(return_len)

train_data['prefix_len'] = train_data.prefix.apply(return_len)
val_data['prefix_len'] = val_data.prefix.apply(return_len)
test_data['prefix_len'] = test_data.prefix.apply(return_len)


def max_len(a):
    a = eval(a)
    test_a = list(a)
    test_b = list(a.values())
    if test_b == []:
        return 0
    weizhi = test_b.index(max(test_b))
    return len(test_a[weizhi])


train_data['max_query_len'] = train_data.query_prediction.apply(max_len)  # 最大概率对应的句子的长度
val_data['max_query_len'] = val_data.query_prediction.apply(max_len)
test_data['max_query_len'] = test_data.query_prediction.apply(max_len)

print(3)
'''
把title_len进行编码
实际模型中这个特征的得分好像很低
'''

a = []  # 0-10
b = []  # 10-25
c = []  # 25-35
d = []  # 35-45
e = []  # 45-
f = []  # 只出现一次
for i in train_data.title_len.unique():
    try:
        if (train_data["label"][train_data["title_len"] == i].value_counts(normalize=True)[1] * 100) < 10:
            a.append(i)

        if 10 < (train_data["label"][train_data["title_len"] == i].value_counts(normalize=True)[1] * 100) < 25:
            b.append(i)

        if 25 < (train_data["label"][train_data["title_len"] == i].value_counts(normalize=True)[1] * 100) < 35:
            c.append(i)

        if 35 < (train_data["label"][train_data["title_len"] == i].value_counts(normalize=True)[1] * 100) < 45:
            d.append(i)

        if 45 < (train_data["label"][train_data["title_len"] == i].value_counts(normalize=True)[1] * 100) < 100:
            e.append(i)

    except:
        f.append(i)


def title_len_code(i):
    if i in a:
        return 10
    if i in b:
        return 20
    if i in c:
        return 30
    if i in d:
        return 40
    if i in e:
        return 50
    if i in f:
        return 0


train_data['title_code'] = train_data.title_len.apply(title_len_code)
val_data['title_code'] = val_data.title_len.apply(title_len_code)
test_data['title_code'] = test_data.title_len.apply(title_len_code)

'''
忘了看的哪篇开源的了
计算title是否在query_prediction里面且对应的概率大于10%
'''


def bbb(a, b):
    if a == 'nan':
        return np.nan
    c = {}
    for i, j in eval(a).items():
        c[i.lower()] = j

    if b.lower() in c:
        if float(c[b.lower()]) > 0.1:
            return 1
        else:
            return 0
    else:
        return 0


train_data['in_query_big'] = train_data.apply(lambda x: bbb(x['query_prediction'], x['title']), axis=1)
val_data['in_query_big'] = val_data.apply(lambda x: bbb(x['query_prediction'], x['title']), axis=1)
test_data['in_query_big'] = test_data.apply(lambda x: bbb(x['query_prediction'], x['title']), axis=1)
'''
计算prefix_len,title_len并且用对应的tag作为权重
'''
train_data['prefix_len'] = train_data['prefix'].apply(return_len)
val_data['prefix_len'] = val_data['prefix'].apply(return_len)
test_data['prefix_len'] = test_data['prefix'].apply(return_len)

test_b = {}
for i in train_data.tag.unique():
    test_b[i] = train_data[(train_data['tag'] == i)].title_len.median()


def sim(a):
    return a[0] / a[1]


test_b_pre = {}
for i in train_data.tag.unique():
    test_b_pre[i] = train_data[(train_data['tag'] == i)].prefix_len.median()


def get_title_len(a):
    return a[1] / test_b[a[0]]


def get_prefix_len(a):
    return a[1] / test_b_pre[a[0]]


train_data['title_len'] = train_data[['tag', 'title_len']].apply(get_title_len, axis=1)
test_data['title_len'] = test_data[['tag', 'title_len']].apply(get_title_len, axis=1)
val_data['title_len'] = val_data[['tag', 'title_len']].apply(get_title_len, axis=1)

train_data['prefix_len'] = train_data[['tag', 'prefix_len']].apply(get_prefix_len, axis=1)
test_data['prefix_len'] = test_data[['tag', 'prefix_len']].apply(get_prefix_len, axis=1)
val_data['prefix_len'] = val_data[['tag', 'prefix_len']].apply(get_prefix_len, axis=1)

train_data['sim'] = train_data[['prefix_len', 'title_len', 'tag']].apply(sim, axis=1)
val_data['sim'] = val_data[['prefix_len', 'title_len', 'tag']].apply(sim, axis=1)
test_data['sim'] = test_data[['prefix_len', 'title_len', 'tag']].apply(sim, axis=1)
'''
计算query_prediction里面有几种选择 我感觉预剪枝要把这个剪掉
'''
train_data['query_prediction_len'] = train_data['query_prediction'].apply(lambda x: str(x).count(',') + 1)
val_data['query_prediction_len'] = val_data['query_prediction'].apply(lambda x: str(x).count(',') + 1)
test_data['query_prediction_len'] = test_data['query_prediction'].apply(lambda x: str(x).count(',') + 1)
'''
我觉得很有意思的是prefix_len和title_len和tag做组合特征 但是我prefix_len和title_len做过处理了 所以好像不太好？（有时间研究一下）
'''
temp8 = train_data.groupby(['prefix_len', 'title_len', 'tag'], as_index=False)['label'].agg(
    {'click8': 'sum', 'count8': 'count', 'ctr8': 'mean'})
train_data = pd.merge(train_data, temp8, on=['prefix_len', 'title_len', 'tag'], how='left')
val_data = pd.merge(val_data, temp8, on=['prefix_len', 'title_len', 'tag'], how='left')
test_data = pd.merge(test_data, temp8, on=['prefix_len', 'title_len', 'tag'], how='left')

print('Add Levenshtein distance')
from Levenshtein import distance, hamming, median

train_data['levenshtein'] = train_data.apply(lambda x: distance(x['prefix'], x['title']), axis=1)
val_data['levenshtein'] = val_data.apply(lambda x: distance(x['prefix'], x['title']), axis=1)
test_data['levenshtein'] = val_data.apply(lambda x: distance(x['prefix'], x['title']), axis=1)

print('Add Levenshtein distance ratio')
train_data['levenshtein_rate'] = train_data.apply(
    lambda x: x['levenshtein'] / (max(len(x['prefix']), len(x['title'])) + 1), axis=1)
val_data['levenshtein_rate'] = val_data.apply(lambda x: x['levenshtein'] / (max(len(x['prefix']), len(x['title'])) + 1),
                                              axis=1)
test_data['levenshtein_rate'] = test_data.apply(
    lambda x: x['levenshtein'] / (max(len(x['prefix']), len(x['title'])) + 1), axis=1)

print('Add jaccard_similarity')


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


train_data['jaccard_similarity'] = train_data.apply(lambda x: jaccard_similarity(x['prefix'], x['title']), axis=1)
val_data['jaccard_similarity'] = val_data.apply(lambda x: jaccard_similarity(x['prefix'], x['title']), axis=1)
test_data['jaccard_similarity'] = test_data.apply(lambda x: jaccard_similarity(x['prefix'], x['title']), axis=1)

print('Add query_prediction_match_ratio')


def match(df):
    score = 0.0
    d = json.loads(str(df['query_prediction']))
    for k, v in d.items():
        score += float(v) * jaccard_similarity(k, df['title'])
    #     print(score)
    return score if d else jaccard_similarity(df['prefix'], df['title'])


train_data['query_match'] = train_data.apply(lambda x: match(x), axis=1)
val_data['query_match'] = val_data.apply(lambda x: match(x), axis=1)
test_data['query_match'] = test_data.apply(lambda x: match(x), axis=1)

print('Add title_query_max')

def title_query_max(df):
    d = json.loads(str(df['query_prediction']))
    return max([jaccard_similarity(k, df['title']) for k in d.keys()]) if d else 0


train_data['title_query_max'] = train_data.apply(lambda x: title_query_max(x), axis=1)
val_data['title_query_max'] = val_data.apply(lambda x: title_query_max(x), axis=1)
test_data['title_query_max'] = test_data.apply(lambda x: title_query_max(x), axis=1)

print('Add title_query_mean')
import json


def title_query_mean(df):
    d = json.loads(str(df['query_prediction']))
    return np.mean([jaccard_similarity(k, df['title']) for k in d.keys()]) if d else 0


train_data['title_query_mean'] = train_data.apply(lambda x: title_query_mean(x), axis=1)
val_data['title_query_mean'] = val_data.apply(lambda x: title_query_mean(x), axis=1)
test_data['title_query_mean'] = test_data.apply(lambda x: title_query_mean(x), axis=1)

print('Add title_query_rank')
import json


def rank(df):
    d = json.loads(str(df['query_prediction']))
    rank = 1 if d else 0
    sim = 0
    cp = 0.0
    for k, v in d.items():
        if jaccard_similarity(k, df['title']) > sim:
            sim = jaccard_similarity(k, df['title'])
            cp = float(v)
    for v in d.values():
        if float(v) > cp:
            rank += 1
    return rank


train_data['query_match'] = train_data.apply(lambda x: rank(x), axis=1)
val_data['query_match'] = val_data.apply(lambda x: rank(x), axis=1)
test_data['query_match'] = test_data.apply(lambda x: rank(x), axis=1)

train_data['flag'] = 0
val_data['flag'] = -1
test_data['flag'] = -2

data = pd.concat([train_data, val_data, test_data], ignore_index=True)

'''
第一个人开源的代码 就是计算title 和 prefix 和query_prediction里的各个选择的相似度 感觉不如词义相似度有用？
'''


def extract_prob(pred):
    pred = eval(pred)
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_prob_lst = []
    for i in range(10):
        if len(pred) < i + 2:
            pred_prob_lst.append(0)
        else:
            pred_prob_lst.append(pred[i][1])
    return pred_prob_lst


def extract_similarity(lst):
    pred = eval(lst[1])
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    len_prefix = lst[0]
    similarity = []
    for i in range(10):
        if len(pred) < i + 2:
            similarity.append(0)
        else:
            similarity.append(len_prefix / float(len(pred[i][0])))
    return similarity


def levenshtein_similarity(str1, str2):
    return Levenshtein.ratio(str1, str2)


print(4)

def get_equal_rate(lst):
    pred = eval(lst[1])
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    equal_rate = []
    for i in range(10):
        if len(pred) < i + 2:
            equal_rate.append(0)
        else:
            equal_rate.append(levenshtein_similarity(lst[0], pred[i][0]))
    return equal_rate


data['pred_prob_lst'] = data['query_prediction'].apply(extract_prob)
data['similarity'] = data[['prefix_len', 'query_prediction']].apply(extract_similarity, axis=1)
data['equal_rate'] = data[['title', 'query_prediction']].apply(get_equal_rate, axis=1)


def add_pred_similarity_feat(data):
    for i in range(10):
        data['prediction' + str(i)] = data.pred_prob_lst.apply(lambda x: float(x[i]))
        data['similarity' + str(i)] = data.similarity.apply(lambda x: float(x[i]))
        data['equal_rate' + str(i)] = data.equal_rate.apply(lambda x: float(x[i]))
    return data


data = add_pred_similarity_feat(data)
data = data.drop(['prefix', 'title', 'tag', 'query_prediction', 'pred_prob_lst', 'similarity', 'equal_rate'], axis=1)
train_data = data[data.flag == 0]
val_data = data[data.flag == -1]
test_data = data[data.flag == -2]

train_data_ = pd.concat([train_data, val_data])
# test_data = val_data

X = np.array(train_data_.drop(['label', 'flag'], axis=1))
y = np.array(train_data_['label'])
X_test_ = np.array(test_data.drop(['label', 'flag'], axis=1))
print('================================')
print(X.shape)
print(y.shape)
print('================================')

xx_logloss = []
xx_submit = []
N = 5
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'reg_alpha': 0.041545473,  # 我不记得昨天有没有加这两个参数 但是我有一次加了这两个参数效果是提升了的
    'reg_lambda': 0.0535294,
    'bagging_freq': 5,
    'verbose': 1
}

for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
    print(f1_score(y_test, np.where(gbm.predict(X_test, num_iteration=gbm.best_iteration) > 0.41, 1, 0)))
    xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
    xx_submit.append(gbm.predict(X_test_, num_iteration=gbm.best_iteration))

print('train_logloss:', np.mean(xx_logloss))
plt.figure(figsize=(12, 6))
lgb.plot_importance(gbm, max_num_features=30)
plt.title("Featurertances")
plt.show()
score_b = list(gbm.feature_importance())
s = 0
for i in xx_submit:
    s = s + i

test_data['test_label'] = list(s / N)
test_data['test_label'] = (test_data['test_label'] > 0.41) * 1
test_data.test_label.value_counts()
test_data['test_label'].to_csv('result.csv', index=False)