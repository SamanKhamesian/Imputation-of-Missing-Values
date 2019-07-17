import numpy as np
from skfuzzy import cmeans

from config import NAN, FCMParam


class FCMeansEstimator:
    def __init__(self, c, m, data):
        self.c = c
        self.m = m
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
    def estimate_missing_values(self):
        estimated_data = []
        complete_data = np.array([self.data[x] for x in self.complete_rows])
        centers, _, _, _, _, _, _ = cmeans(data=complete_data.transpose(), c=self.c, m=self.m, error=FCMParam.ERROR,
                                           maxiter=FCMParam.MAX_ITR, init=None)

        # Calculate distance between two points based on euclidean distance
        def calculate_distance(data_1, data_2):
            return np.linalg.norm(data_1 - data_2)

        # Calculate the membership value for given point
        def calculate_membership(dist_matrix, distance, m):
            numerator = np.power(distance, -2 / (1 - m))
            denominator = np.array([np.power(x, -2 / (1 - m)) for x in dist_matrix]).sum()
            return numerator / denominator

        for i in self.incomplete_rows:
            estimated = 0
            dist, membership_value = [], []
            miss_ind = np.where(self.data[i] == NAN)[0][0]

            for center in centers:
                dist.append(calculate_distance(data_1=np.delete(np.array(center), miss_ind),
                                               data_2=np.delete(np.array(self.data[i]), miss_ind)))

            for d in dist:
                membership_value.append(calculate_membership(dist, d, self.m))

            for k in range(self.c):
                estimated += centers[k][miss_ind] * membership_value[k]

            estimated_data.append(estimated)

        return np.array(estimated_data)
