import pandas as pd
import numpy as np
"""
base model for tianchi_fresh_comp_challenge_offline_version.
build simple item-user matrix first
"""
#user = pd.read_csv("fresh_comp_offline/tianchi_fresh_comp_train_user.csv")
user_base = pd.read_csv("fresh_comp_offline/tianchi_fresh_comp_train_user_filtered.csv")
item = pd.read_csv("fresh_comp_offline/tianchi_fresh_comp_train_item.csv")
"""
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
#user 20000
#user.shape = (2084859, 6)
#sort
new_user_base = user_base.sort(['user_id','time','item_id'])
new_item_base = item.sort(['item_id'])
# saving the dataframe
new_user_base.to_csv("fresh_comp_offline/user_sorted.csv")
new_item_base.to_csv("fresh_comp_offline/item_sorted.csv")
# now we have the clean item and user dataframe
# we can build the simple base model from those two dataset
print(new_user_base.shape)
print(new_item_base.shape)
new_item_category = new_item_base['item_category']
Icategory = pd.Series.unique(new_item_category)
print(len(Icategory))

# build base model without behavior_type,user_geohash,time and item_ geohash
# 20000(unique user) * 1054(unique item category)
umatrix = np.zeros((20000,1054))
user_list = new_user_base['user_id']
#used for query
Iuser = pd.Series.unique(user_list)

# only 19972 user can used at this moment
print(len(Iuser))
# use 492 for query test
item_list = new_user_base.query('user_id == 492')['item_category']
Iitem = pd.Series.unique(item_list)
print(Iitem)
print(Icategory)