from xgboost import XGBRegressor


def train_model(X_train, y_train, sample_weight=None):

    model = XGBRegressor(
        objective="reg:squarederror",

        n_estimators=750,
        learning_rate=0.03,
        max_depth=5,

        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,

        reg_alpha=0.1,
        reg_lambda=1.0,

        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight
    )

    return model