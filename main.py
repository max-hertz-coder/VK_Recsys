import turicreate as tc


actions = tc.SFrame.read_csv('data/train_interactions.csv')

users = tc.SFrame.read_csv('data/users_meta.csv')
items = tc.SFrame.read_csv('data/items_meta.csv')
test = tc.SFrame.read_csv('data/test_pairs.csv')
actions['weight'] = actions['like'] - actions['dislike']
actions = actions['user_id', 'item_id', 'weight']

model = tc.recommneder.ranking_factorization_recommender.create(actions, 'user_id', 'item_id', target='weight', user_info=users, item_info=items)

scores = model.predict(test)
test['predict'] = scores
test.export.csv('sub.csv')