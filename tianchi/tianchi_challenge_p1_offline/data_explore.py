import pandas as pd
import numpy as np
"""
U——用户集合
I——商品全集
P——商品子集，P ⊆ I
D——用户对商品全集的行为数据集合
Goal:利用D来构造U中用户对P中商品的推荐模型
"""
item = pd.read_csv("fresh_comp_offline/tianchi_fresh_comp_train_item.csv")
#print(item.head())
user = pd.read_csv("fresh_comp_offline/tianchi_fresh_comp_train_user.csv")
#print(user.head())
"""
P:
Item:(620918, 3)
Contain 3 features:
item_id:商品标识( 抽样&字段脱敏) numeric type
item_geohash:商品位置的空间标识，可以为空(由经纬度通过保密的算法生成) text type
item_category:商品分类标识(字段脱敏) numeric type
-------------------
D:
User:(23291027, 6)
Contain 6 features:
user_id:用户标识(抽样&字段脱敏) numeric type
item_id:商品标识(字段脱敏) numeric type
behavior_type:1/2/3/4(浏览、收藏、加购物车、购买) numeric type
user_geohash:用户位置的空间标识，可以为空(由经纬度通过保密的算法生成) text type
item_category:商品分类标识(字段脱敏) numeric type
time:行为时间(hours) times type
"""
#print(item.shape)
"""
P;                     start/min               end/max           count          unique
item_id                   958                 404562430          620918         422858
item_category              2                    14071            620918         1054
注： 在数据中我们发现在样本数据中存在一个现象，每个物品的id仅会归类到一个category中；
    在这里不存在一个item_id归属于多个category的现象，而对于一个item_id，可能;
    存在多item_geohash属性，这一现象标志了该物品可能存在不同的商品位置空间。而这些；
    存在item_geohash属性的物品总数为 620918 － 422858 个。
"""
"""
D;                     start/min               end/max           count          unique
user_id                  492                  142442955          23291027       20000
item_id                  37                   404562488          23291027       4758484
item_category            2                      14080            23291027       9557
time
---------------
                         max              count         unique
missing item_id          58               --
missing item_category    9                --
missing user_id          -                --
"""
item = item.sort(['item_id','item_category'])
user = user.sort(['user_id','item_id','item_category'])
item_id = item['item_id']
item_id_min = np.min(item_id)
item_id_max = np.max(item_id)
print(item_id_min,item_id_max)
item_category = item['item_category']
item_category_min = np.min(item_category)
item_category_max = np.max(item_category)
print(item_category_min,item_category_max)
user_id = user['user_id']
user_id_min = np.min(user_id)
user_id_max = np.max(user_id)
print(user_id_min,user_id_max)
Uitem_id = user['item_id']
Uitem_id_min = np.min(Uitem_id)
Uitem_id_max = np.max(Uitem_id)
print(Uitem_id_min,Uitem_id_max)
Uitem_category = user['item_category']
Uitem_category_min = np.min(Uitem_category)
Uitem_category_max = np.max(Uitem_category)
print(Uitem_category_min,Uitem_category_max)
#print(item.describe())
#print(user.describe())
item_unique = pd.Series.unique(item_id)
print(len(item_unique))
Icategory = pd.Series.unique(item_category)
print(len(Icategory))
user_unique = pd.Series.unique(user_id)
print(len(user_unique))
Uitem_unqiue = pd.Series.unique(Uitem_id)
print(len(Uitem_unqiue))
Ucategory = pd.Series.unique(Uitem_category)
print(len(Ucategory))
# reorginze the dataframe of items
df_item_unique = pd.DataFrame({'item_id':item_unique})
#df_item_unique = df_item_unique.sort()
df_item_unique.to_csv("unique_item.csv",header=False, index=False)
print(df_item_unique.head())

item = item.drop('item_geohash', axis=1)
test = item
#test = pd.merge(df_item_unique, item, how='right', on = ['item_id'] )
print(test.shape)
#item_id = test['item_id']
#item_unique = pd.Series.unique(item_id)
#print(len(item_unique))
test = test.drop_duplicates(['item_id'])
print(test.shape)
test.to_csv("test.csv",index = False)
