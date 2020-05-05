class TimeSeriesSplit:

    def __init__(self, n_splits=5, test_size=24, missing_date=24):
        self.n_splits = n_splits
        self.test_size = test_size
        self.missing_date = missing_date

    def split(self, data):
        n = self.n_splits
        while n > 0:

            end_train = -7 * n * self.test_size
            start_test = end_train + self.missing_date
            end_test = start_test + self.test_size

            train = data[:end_train]
            test = data[start_test:end_test]
            yield train, test
            n -= 1