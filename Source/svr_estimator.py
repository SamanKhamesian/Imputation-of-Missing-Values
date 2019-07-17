import numpy as np
from sklearn.svm import SVR

from config import NAN, SVRParam


class SVREstimator:
    def __init__(self, data):
        self.data = data
        self.complete_rows, self.incomplete_rows = self.__extract_rows()

    # Extract complete and incomplete rows
    def __extract_rows(self):
        rows, columns = len(self.data), len(self.data[0])
        complete_rows, incomplete_rows = [], []
        for i in range(rows):
            for j in range(columns):
                if self.data[i][j] == NAN:
                    incomplete_rows.append(i)
                    break
                complete_rows.append(i)

        return np.array(complete_rows), np.array(incomplete_rows)

    # Estimate the missing values
    def estimate_missing_value(self):
        estimated_data = np.zeros(len(self.incomplete_rows))
        complete_data = np.array([self.data[x] for x in self.complete_rows])
        incomplete_data = np.array([self.data[x] for x in self.incomplete_rows])

        for column, value in enumerate(incomplete_data.transpose()):
            ind_rows = np.where(value == NAN)[0]
            if len(ind_rows) > 0:
                x_train = np.delete(complete_data.transpose(), column, 0).transpose()
                y_train = np.array(complete_data[:, column])

                model = SVR(gamma='scale', C=SVRParam.C, epsilon=SVRParam.EP)
                model.fit(x_train, y_train)

                x_test = []
                x_test_temp = np.delete(incomplete_data.transpose(), column, 0).transpose()
                for i in ind_rows:
                    x_test.append(x_test_temp[i])

                predicted = model.predict(np.array(x_test))

                for i, v in enumerate(ind_rows):
                    estimated_data[v] = predicted[i]

        return estimated_data
