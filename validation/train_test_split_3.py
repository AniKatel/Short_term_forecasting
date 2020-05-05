class TimeSeriesSplit:

    def __init__(self, n_splits=5, test_size=24, missing_date=24):
        self.n_splits = n_splits
        self.test_size = test_size
        self.missing_date = missing_date

    def split(self, data):
        n = 0
        pointer = 1
        weekends = [5, 6]
        weekdays = [0, 1, 2, 3, 4]
        predict_weekday = (data.index[-1].weekday() + 2) % 7

        while n < self.n_splits:
            test_weekday = data.index[-pointer].weekday()
            if (predict_weekday in weekends and test_weekday in weekends
                    or predict_weekday in weekdays and test_weekday in weekdays):
                end_train = -(pointer - 1 + self.test_size + self.missing_date)
                start_test = end_train + self.missing_date
                end_test = start_test + self.test_size

                train = data[:end_train]
                if pointer != 1:
                    test = data[start_test:end_test]
                else:
                    test = data[start_test:]
                yield train, test
                n += 1
            pointer += 24
