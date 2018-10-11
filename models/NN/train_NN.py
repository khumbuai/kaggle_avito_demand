from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(4)

import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
from tqdm import tqdm
from keras.layers import Input, PReLU, BatchNormalization, SpatialDropout1D, Dropout, GlobalAveragePooling1D, CuDNNGRU, \
    Bidirectional, Dense, Embedding
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
import keras.backend as K
import time
import gc
import pickle
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import TruncatedSVD
from keras.layers import Concatenate, Flatten
from sklearn import metrics
from keras.initializers import he_uniform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from utils import root_mean_squared_error
from global_variables import *


MODEL_PATH = 'models/topmodels/NN/LSTM_CONV1/'
EMBEDDING = 'word_embeddings/w2v/model2.w2v'
num_words = 100000
max_len_desc = 130
max_len_title = 20




def prense(x, units):
    x = Dense(units)(x)
    x = PReLU()(x)
    return x


print('Loading data...', end='')
train = pd.read_csv(TRAIN_CSV, index_col='item_id',parse_dates=['activation_date'])#.sample(frac=0.1, random_state=23)
test = pd.read_csv(TEST_CSV, index_col='item_id', parse_dates=['activation_date'])#.sample(frac=0.1, random_state=23)
print('Done')
print(train.shape)
print(test.shape)

# reorganize data handle N/A
labels = train[['deal_probability']].copy()
# train.drop(['deal_probability'],axis = 1,inplace = True)
train_indices = train.index
test_indices = test.index
df = pd.concat([train, test])
del train, test

text_cols = ['title', 'description']

# PROCESS TEXT: RAW
print("Text to seq process...")
from keras.preprocessing.text import Tokenizer

for col in text_cols:
    df[col] = df[col].astype(str).fillna('nan')
    df[col] = df[col].str.lower()

df['text'] = df.apply(lambda row: ' '.join([row[col] for col in text_cols]), axis=1)
print("   Fitting tokenizer...")
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df['text'].values.tolist())
df.drop(['text'], axis=1, inplace=True)

df['param_3'] = df['param_3'].astype(str).fillna('nan')
params_vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),
    max_features=10000, analyzer='char'
)


print("   Transforming text to seq...")
for col in text_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.lower()
    df[col] = tokenizer.texts_to_sequences(df[col])




df["wday"] = df['activation_date'].dt.weekday
#df["week"] = df['activation_date'].dt.week
#df["month"] = df['activation_date'].dt.day
df.drop(['activation_date','image'],axis = 1,inplace = True)
df['image_top_1'].fillna(-1, inplace=True)
df['image_top_1'] = df['image_top_1'].astype(np.int)
df['city'] = df['city'] + '_' + df['region']


#SCALE price variable
df['price'] = np.log1p(df['price'])
#df['item_seq_number'] = np.log1p(df['item_seq_number'])

print('loading duo')
duo_cols = ['r_title_wds','r_title_cap','r_title_spa','n_description_uni_wds','r_description_wds','r_description_dig',
 'r_description_cap','r_description_spa','r_description_pun','r_title_description','r_title_params']
train_duo = pd.read_csv('models/word_char/all/train_word_char.csv', index_col='item_id', usecols=duo_cols + ['item_id'])
test_duo = pd.read_csv('models/word_char/all/test_word_char.csv', index_col='item_id', usecols=duo_cols + ['item_id'])
#duo_cols = train_duo.columns.values.tolist()
df = df.join(pd.concat([train_duo,test_duo]))
del train_duo, test_duo

print('loading region')
region_cat_cols = ['reg_time_zone']
region_num_cols = ['reg_dense','rural','reg_population','reg_urban']
df_region = pd.read_csv('models/region/wiki/df_region_wiki.csv', index_col=0)
df = df.join(df_region)
del df_region

print('loading ll')
ll_cat_cols = []#['lat_lon_hdbscan_cluster_05_03','lat_lon_hdbscan_cluster_10_03','lat_lon_hdbscan_cluster_20_03']
ll_num_cols = ['latitude','longitude']
ll_cols = ll_cat_cols + ll_num_cols
train_ll = pd.read_csv('models/long_lat_clusters/train_ll.csv', usecols=ll_cols + ['item_id'], index_col='item_id')
test_ll = pd.read_csv('models/long_lat_clusters/test_ll.csv', usecols=ll_cols + ['item_id'], index_col='item_id')
df_ll = pd.concat([train_ll,test_ll])
df = df.join(df_ll)
del train_ll, test_ll

print('loading img_deal_prob')
train_img = pd.read_csv('models/image/DenseNet121/train_img_deal_prob.csv', index_col=0)
test_img = pd.read_csv('models/image/DenseNet121//test_img_deal_prob.csv', index_col=0)
df = df.join(pd.concat([train_img,test_img]))
del train_img, test_img


print('loading ridge1')
train_ridge1 = pd.read_csv('models/topmodels/RIDGE/1/train_ridge1.csv', index_col='item_id')
test_ridge1 = pd.read_csv('models/topmodels/RIDGE/1/test_ridge1.csv',index_col='item_id')
df_ridge1 = pd.concat([train_ridge1,test_ridge1])
df_ridge1.columns.values[0] = 'ridge_1'
df = df.join(df_ridge1)
del train_ridge1, test_ridge1, df_ridge1

print('loading ridge2')
train_ridge2 = pd.read_csv('models/topmodels/RIDGE/2/train_ridge2.csv', index_col='item_id')
test_ridge2 = pd.read_csv('models/topmodels/RIDGE/2/test_ridge2.csv',index_col='item_id')
df_ridge2 = pd.concat([train_ridge2,test_ridge2])
df_ridge2.columns.values[0] = 'ridge_2'
df = df.join(df_ridge2)
del train_ridge2, test_ridge2, df_ridge2

print('loading ridge price')
train_ridge_price = pd.read_csv('models/topmodels/RIDGE/3/train_ridge_price1.csv', index_col='item_id')
test_ridge_price = pd.read_csv('models/topmodels/RIDGE/3/test_ridge_price1.csv',index_col='item_id')
df = df.join(pd.concat([train_ridge_price,test_ridge_price]))
del train_ridge_price, test_ridge_price,

print('loading densenet201_conf')
train_dense = pd.read_csv('models/image/DenseNet201/train_densenet201_conf.csv',index_col='item_id')
test_dense = pd.read_csv('models/image/DenseNet201/test_densenet201_conf.csv',index_col='item_id')
df = df.join(pd.concat([train_dense,test_dense]))
del train_dense, test_dense

print('loading ava')
train_ava = pd.read_csv('models/image/Inception_ResNetv2/AVA/train_incep_ava_mean.csv',index_col='item_id')
test_ava = pd.read_csv('models/image/Inception_ResNetv2/AVA/test_incep_ava_mean.csv',index_col='item_id')
df = df.join(pd.concat([train_ava,test_ava]))
del train_ava, test_ava

print('loading periods')
period_cols = ['n_user_items', 'avg_times_up_user', 'avg_days_up_user','avg_init_time','avg_wait_days','avg_dupe_count']
train_period = pd.read_csv('models/periods/train_period.csv', index_col='item_id', usecols=['item_id'] + period_cols)
test_period = pd.read_csv('models/periods/test_period.csv', index_col='item_id', usecols=['item_id'] + period_cols)
df_period = pd.concat([train_period, test_period])
df = df.join(df_period)
del train_period, test_period, df_period

gc.collect()





print('Handling categorical variables...')

cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'user_type', 'image_top_1', 'wday',
            'n_user_items']  #
cat_cols += region_cat_cols + ['user_id']
cat_cols += ['param_1', 'param_2']
# to_drop = ['user_id']

# df2 = df.loc[train_indices].copy()
encoders = [{} for cat in cat_cols]

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

for i, cat in enumerate(cat_cols):
    df[cat] = df[cat].astype(str).fillna('nan')
    print('grouping {}'.format(cat))
    gp = df.loc[train_indices, cat].astype(str).value_counts().divide(len(train_indices))
    gp = gp.reset_index()
    gp['np'] = gp.apply(lambda x: x['index'] if x[cat] > 0.00001 else 'other_' + cat, axis=1)
    df[cat] = df[cat].astype(str).map(gp.set_index('index')['np'])
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(df.loc[train_indices, cat].astype(str).unique())}
    df[cat] = df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders]

with open(MODEL_PATH + 'encoders.p', 'wb') as f:
    pickle.dump(encoders, f)

df['param_3'] = df['param_3'].astype(str).fillna('nan')
print('vectorizing params')
params_vectorizer.fit(df.loc[train_indices, 'param_3'].values)


text_cols = ['description', 'title','param_3']
to_ignore = ['deal_probability'] # ,'user_id'
non_num_cols = text_cols + cat_cols + to_ignore
num_cols = [c for c in df.columns.values if not c in non_num_cols]

print(non_num_cols)
print(num_cols)


print('scaling num_cols')
for col in num_cols:
    print('scaling {}'.format(col))
    col_mean = df[col].mean()
    df[col].fillna(col_mean, inplace=True)
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    del scaler

from gensim.models import word2vec

model = word2vec.Word2Vec.load(EMBEDDING)
embed_size = model.vector_size

word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= num_words: continue
    try:
        embedding_vector = model[word]
    except KeyError:
        embedding_vector = None
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del model
gc.collect()


def get_input_features(df):
    X = {'desc':pad_sequences(df['description'], maxlen=max_len_desc),
         'title':pad_sequences(df['title'], maxlen=max_len_title)}
    X['sparse_params']= params_vectorizer.transform(df['param_3'].values)
    X['numerical'] = np.array(df[num_cols])
    for cat in cat_cols:
        X[cat] = np.array(df[cat])

    return X


from keras.regularizers import l2
from keras.layers import Layer, Conv1D, GlobalMaxPool1D, GlobalAveragePooling1D, CuDNNLSTM
import keras


class LayerNorm1D(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=keras.initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True, )

        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def build_model():
    sparse_params = Input(shape=[X_train['sparse_params'].shape[1]], dtype='float32', sparse=True, name='sparse_params')

    categorical_inputs = []
    for cat in cat_cols:
        categorical_inputs.append(Input(shape=[1], name=cat))

    categorical_embeddings = []
    for i, cat in enumerate(cat_cols):
        categorical_embeddings.append(
            Embedding(embed_sizes[i], 10, embeddings_regularizer=l2(0.00001))(categorical_inputs[i]))

    categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
    categorical_logits = prense(categorical_logits, 256)
    categorical_logits = prense(categorical_logits, 128)

    numerical_inputs = Input(shape=[len(num_cols)], name='numerical')

    numerical_logits = numerical_inputs
    numerical_logits = BatchNormalization()(numerical_logits)
    numerical_logits = prense(numerical_logits, 256)
    numerical_logits = prense(numerical_logits, 128)

    params_logits = prense(sparse_params, 64)
    params_logits = prense(params_logits, 32)

    desc_inp = Input(shape=[max_len_desc], name='desc')
    title_inp = Input(shape=[max_len_title], name='title')
    embedding = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)  # nb_words
    emb_desc = embedding(desc_inp)
    emb_title = embedding(title_inp)
    emb_text = Concatenate(axis=1)([emb_desc,emb_title])

    text_logits = SpatialDropout1D(0.2)(emb_text)
    text_logits = Bidirectional(CuDNNLSTM(128, return_sequences=True))(text_logits)
    text_logits = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(text_logits)
    avg_pool = GlobalAveragePooling1D()(text_logits)
    max_pool = GlobalMaxPool1D()(text_logits)
    text_logits = Concatenate()([avg_pool, max_pool])
    x = Dropout(0.2)(text_logits)
    x = Concatenate()([categorical_logits, text_logits])
    x = BatchNormalization()(x)
    x = Concatenate()([x, params_logits, numerical_logits])
    x = Dense(512, kernel_initializer=he_uniform(seed=0))(x)
    x = PReLU()(x)
    x = Dense(256, kernel_initializer=he_uniform(seed=0))(x)
    x = PReLU()(x)
    x = Dense(128, kernel_initializer=he_uniform(seed=0))(x)
    x = PReLU()(x)
    x = LayerNorm1D()(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[desc_inp] + [title_inp] + [sparse_params] + categorical_inputs + [numerical_inputs],
                  outputs=out)

    model.compile(optimizer=Adam(lr=0.0005, clipnorm=0.5), loss='mean_squared_error',
                  metrics=[root_mean_squared_error])
    return model


from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences

N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, random_state=24)
fold_gen = kf.split(train_indices)
FOLDS = 10
preds = []
ys = []
len_runs = []
# for k in [0,1,2]:
#    train_index_ids, valid_index_ids = fold_gen.__next__()
for fold_id in range(FOLDS):
    train_index_ids, valid_index_ids = fold_gen.__next__()
    train_indices2 = train_indices[train_index_ids]
    valid_indices = train_indices[valid_index_ids]

    dtrain = df.loc[train_indices2]
    dvalid = df.loc[valid_indices]

    y_train = labels.loc[train_indices2]['deal_probability']
    y_valid = labels.loc[valid_indices]['deal_probability']
    X_train = get_input_features(dtrain)
    del dtrain
    X_valid = get_input_features(dvalid)
    del dvalid

    model = build_model()

    check_point = ModelCheckpoint(MODEL_PATH + 'model{}.hdf5'.format(fold_id), monitor='val_loss', mode='min', save_best_only=True,
                                  verbose=1)
    early_stop = EarlyStopping(patience=2)
    history = model.fit(X_train, y_train, batch_size=512, epochs=15, validation_data=(X_valid, y_valid),
                        verbose=1, callbacks=[check_point, early_stop])
    len_runs.append(len(history.history['loss']))
    preds.append(model.predict(X_valid, verbose=True))
    ys.append(y_valid)
    # rsme = np.sqrt(metrics.mean_squared_error(y_valid, model.predict(X_valid, verbose=True)))
    # print('RMSE:%s' % rsme)

ys2 = np.concatenate(ys)
preds2 = np.concatenate(preds)
rsme = np.sqrt(metrics.mean_squared_error(ys2, preds2))
avg_len_run = np.mean(len_runs)
print('RMSE:%s' % rsme)
print(avg_len_run)

from sklearn.model_selection import KFold

N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, random_state=24)
fold_gen = kf.split(train_indices)
FOLDS = 10
oof_inds = []
# for k in [0,1,2]:
#    train_index_ids, valid_index_ids = fold_gen.__next__()
test_preds = []
#model = build_model()
X_test = get_input_features(df.loc[test_indices])
for fold_id in range(FOLDS):
    train_index_ids, valid_index_ids = fold_gen.__next__()
    oof_inds.append(train_indices[valid_index_ids])

    model.load_weights(MODEL_PATH + 'model{}.hdf5'.format(fold_id))
    test_preds.append(model.predict(X_test,verbose=True))


oof_inds = np.concatenate(oof_inds)
oof_predictions = pd.DataFrame(preds2, index=oof_inds)
oof_predictions.columns = ['oof_deal_probability']
oof_predictions.to_csv(MODEL_PATH + 'oof_predictions.csv',index=True)

from utils import bag_by_average

pred = bag_by_average(test_preds)
subm = pd.read_csv(SAMPLE_SUBMISSION)
subm['deal_probability'] = pred
subm.to_csv(MODEL_PATH + 'submission.csv', index=False)