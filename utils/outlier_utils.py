class Outliers:
    def __init__(self):
        # ad count > 10000
        self.train_userid_outliers = [839368]
        self.test_userid_outliers = [3548147, 3522917, 3206914, 3093561, 3834944, 3648518]
        self.outlier_age = 6
        self.outlier_gender = 1


if __name__ == '__main__':
    outliers = Outliers()
    print(outliers.train_userid_outliers)
    print(outliers.test_userid_outliers)
