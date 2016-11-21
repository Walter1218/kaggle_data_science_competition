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
D:
Item:(620918, 3)
Contain 3 features:
item_id:商品标识( 抽样&字段脱敏) numeric type
item_geohash:商品位置的空间标识，可以为空(由经纬度通过保密的算法生成) text type
item_category:商品分类标识(字段脱敏) numeric type
-------------------
P:
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
D;                     start/min               end/max              count
item_id                   958                 404562430
item_category              2                    14071
"""

"""
P;                     start/min               end/max              count
user_id                  492                  142442955
item_id                  37                   404562488
item_category             2                     14080
time
"""
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
