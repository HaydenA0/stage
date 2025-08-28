import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv("flight_data.csv")
    return data


if __name__ == "__main__":
    data = load_data()
    print(data.head(4))
