from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


def svd_model(df, i_metric=None):
    """
    The function returns best fitted SVD model. Parameters are selected with GridSearchCV using cross - validation.

    :param df: master data created in the main.py.
    :param i_metric: Metric used to select best model, rmse [default].
    :return: Estimated model, rmse of best model and parameters of selected best model.
    """

    if i_metric is None:
        i_metric = ["rmse"]

    df = df.rename(columns={"Book-Rating": "rating"})

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df[["User-ID", "ISBN", "rating"]], reader)

    # - define grid for cross validation
    param_grid = {"n_epochs": [5, 10],
                  "lr_all": [0.002, 0.005],
                  "reg_all": [0.4, 0.6]}

    # - run cross validation
    model_set = GridSearchCV(SVD, param_grid, measures=i_metric, cv=50)
    model_set.fit(data)

    # - get best rmse score
    best_rmse = model_set.best_score[i_metric[0]]

    # - get best params
    best_params = model_set.best_params[i_metric[0]]

    # We can now use the algorithm that yields the best rmse:
    best_model = model_set.best_estimator[i_metric[0]]
    best_model.fit(data.build_full_trainset())

    return best_model, best_rmse, best_params


def als_model(df):
    # - split data into test train
    df = df.withColumnRenamed("Book-Rating", "rating")
    df = df[["User-ID", "ISBN_n", "rating"]]
    (train, test) = df.randomSplit([0.8, 0.2], seed=2020)

    # Create ALS model
    als = ALS(
        userCol="User-ID",
        itemCol="ISBN_n",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop"
    )

    # - Add param_grid
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 50, 100, 150]) \
        .addGrid(als.regParam, [.01, .05, .1, .15]) \
        .build()

    # - Define evaluator as RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction")

    # - Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=10)

    # - Fit cross validator on train
    model = cv.fit(train)

    # - Extract best model from above
    best_model = model.bestModel

    # - View the predictions
    test_predictions = best_model.transform(test)
    rmse = evaluator.evaluate(test_predictions)

    return best_model, rmse, model.bestModel
