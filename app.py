from train_model import SimpleRegressor

def main():
    data_path = "data/bandwidth_prediction_dataset.csv"
    reg = SimpleRegressor()
    df = reg.load_data(data_path)
    X_train, X_test, y_train, y_test = reg.preprocess(df, target_column='actual_bandwidth')
    reg.train(X_train, y_train)
    predictions = reg.evaluate(X_test, y_test)
    #reg.plot_predictions(y_test, predictions)
    reg.save_model("models/bandwidth_model.pkl")

if __name__ == '__main__':
    main()
