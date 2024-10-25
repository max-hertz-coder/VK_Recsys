from scipy.sparse import csr_matrix
import polars as pl
import implicit

train = pl.scan_parquet("../data/train_interactions.parquet")

train = train.filter((pl.col("like") + pl.col("dislike")) >= 1)
train = train.with_columns(weight=pl.col("like") - pl.col("dislike"))
train = train.select("user_id", "item_id", "weight")


items_meta = pl.read_parquet("../data/items_meta.parquet")
users_meta = pl.read_parquet("../data/users_meta.parquet")
n_items = items_meta["item_id"].max() + 1
n_users = users_meta["user_id"].max() + 1


train = csr_matrix((train["weight"],
                    (train["user_id"].to_numpy(),
                     train["item_id"].to_numpy())),
                   shape=(n_users, n_items))


model = implicit.als.AlternatingLeastSquares(factors=50,
                                             iterations=10,
                                             regularization=1,
                                             alpha=100,
                                             calculate_training_loss=True)
model.fit(train)


test_pairs = pl.read_csv('../data/test_pairs.csv')


als_predict = (model.user_factors[test_pairs['user_id']] *
               model.item_factors[test_pairs['item_id']]).sum(axis=1)