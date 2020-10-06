import pandas as pd
import pyspark
import string
from langdetect import detect
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pyspark.sql.types import StringType, StructField, FloatType, LongType, StructType, DateType, IntegerType
from collaborative_filtering import *
import pickle

# - Hom many observations to use when estimating model?
n_obs = 1000


def clean_text(text):
    """
    :param text: String input
    :return Cleaned text
    """
    text = str(text).lower()
    text = re.sub('\[.*?\]', "", text)
    text = re.sub('https?://\S+|www\.\S+', "", text)
    text = re.sub('<.*?>+', "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub('\n', "", text)
    text = re.sub('\w*\d\w*', "", text)
    text = re.sub("[^a-zA-Z]", " ", text)

    # - set stop words
    stop_words = set(stopwords.words('english'))

    # - split into words
    words = text.split()

    cleaned_words = []
    init_lemma = PorterStemmer()

    for word in words:
        if word not in stop_words:
            word = init_lemma.stem(word)
            cleaned_words.append(word)

    return set(cleaned_words)


def _detect_lang(text):
    try:
        return detect(text)
    except Exception:
        return "en"


def data_prep(ratings_p, books_p, users_p, sep_used=None):
    """
    The function returns data necessary for modelling stage.

    :param ratings_p: File path to ratings data
    :param books_p: File path to books data
    :param users_p: File path to users data
    :param sep_used: Separator used in the data
    :return: Data - book_ratings, books, users, merged data
    """

    # - set separator
    if sep_used is None:
        sep_used = [";", ";", ";"]

    # - upload data
    ratings_data = pd.read_csv(ratings_p, sep=sep_used[0],
                               encoding='latin-1', low_memory=False,
                               header=0, escapechar='\\')
    books_data = pd.read_csv(books_p, sep=sep_used[1],
                             encoding='latin-1', low_memory=False,
                             header=0, escapechar='\\')
    users_data = pd.read_csv(users_p, sep=sep_used[2],
                             encoding='latin-1', low_memory=False,
                             header=0, escapechar='\\')

    # - get lowercase of string columns
    books_data = books_data.applymap(lambda s: s.lower() if type(s) == str else s)
    users_data = users_data.applymap(lambda s: s.lower() if type(s) == str else s)

    # - drop columns
    books_data.drop(["Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1, inplace=True)

    # - add new ISBN numeric Index
    books_data["ISBN_n"] = range(len(books_data))

    # - clean text
    books_data["Book-Title-cl"] = books_data["Book-Title"].apply(lambda x: clean_text(x))

    # - detect language
    # books_data["lang"] = books_data["Book-Title"].apply(lambda x: _detect_lang(x))

    # - create location
    location = users_data.Location.str.split(', ', n=2, expand=True)
    users_data['city'] = location[0]
    users_data['state'] = location[1]
    users_data['country'] = location[2]
    users_data.drop('Location', axis=1, inplace=True)

    # - merge data
    merged_data = pd.merge(users_data, ratings_data, on="User-ID")
    merged_data = pd.merge(books_data, merged_data, on="ISBN")

    return pd.DataFrame(ratings_data), pd.DataFrame(books_data), pd.DataFrame(users_data), merged_data


if __name__ == "__main__":

    # - PySpark session
    spark = pyspark.sql.SparkSession.builder.appName("DataSentics Project - Book Recommender").getOrCreate()

    # - Auxiliary functions
    def equivalent_type(f):
        if f == 'datetime64[ns]':
            return DateType()
        elif f == 'int64':
            return LongType()
        elif f == 'int32':
            return IntegerType()
        elif f == 'float64':
            return FloatType()
        else:
            return StringType()


    def define_structure(i_string, format_type):
        try:
            typo = equivalent_type(format_type)
        except:
            typo = StringType()
        return StructField(i_string, typo)


    # - Given pandas DataFrame, it will return a spark's DataFrame.
    def pandas_to_spark(pandas_df):

        columns = list(pandas_df.columns)
        types = list(pandas_df.dtypes)
        struct_list = []

        for column, typo in zip(columns, types):
            struct_list.append(define_structure(column, typo))
        p_schema = StructType(struct_list)

        return spark.createDataFrame(pandas_df, p_schema)


    # - read data
    book_ratings, books, users, master_data = data_prep(
        ratings_p="C:/Users/Peter/Desktop/ds_projects/book_rating/0 data/BX-Book-Ratings.csv",
        books_p="C:/Users/Peter/Desktop/ds_projects/book_rating/0 data/BX-Books.csv",
        users_p="C:/Users/Peter/Desktop/ds_projects/book_rating/0 data/BX-Users.csv")

    # - copy DataFrames to PySpark (unnecessary - it wzs supposed to be used for spark ALS Model)
    s_master_data = pandas_to_spark(master_data[:n_obs])

    # - estimate models (narrow down number of observations due to running time)
    model_svd, rmse_svd, best_params_svd = svd_model(master_data[:n_obs])

    # - save the model to disk
    svd_filename = "C:/Users/Peter/Desktop/ds_projects/book_rating/recommendation-app/model/SVD_Model.pkl"

    with open(svd_filename, 'wb') as file:
        pickle.dump(model_svd, file)

    # - save data to disk
    books.to_csv("C:/Users/Peter/Desktop/ds_projects/book_rating/recommendation-app/data/books.csv", index=False)
    users.to_csv("C:/Users/Peter/Desktop/ds_projects/book_rating/recommendation-app/data/users.csv", index=False)
    master_data.to_csv("C:/Users/Peter/Desktop/ds_projects/book_rating/recommendation-app/data/master_data.csv",
                       index=False)
