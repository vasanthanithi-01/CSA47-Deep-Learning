from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=200, n_features=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for activation in ['relu', 'logistic', 'tanh']:
    model = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation=activation,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(activation, "Accuracy:",
          round(accuracy_score(y_test, pred), 4))
