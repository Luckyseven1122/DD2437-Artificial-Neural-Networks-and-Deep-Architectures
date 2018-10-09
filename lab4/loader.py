from io_tools import get_training_data, get_testing_data
import autoencoder as _encoder
from autoencoder import network
import imp
import sys

class loader:
    def pad(self, x, prompt):
    	return "\n*- " + x.ljust(60, "-") + "._\n" + (">" if prompt else "")

    def check_yes_no(self, x):
    	while x.lower().strip() != "y" and x.lower().strip() != "n":
    		x = input(self.pad("Run again?\nPlease make your selection: Y / N ", True))
    	return True if x.lower().strip() == "y" else False

    def __init__(self):
        self.train_X, self.train_Y = get_training_data()
        self.test_X, self.test_Y = get_testing_data()

    def run(self):
        while (True):
            if self.check_yes_no(''):
                try:
                    imp.reload(_encoder)
                    from autoencoder import network
                    network(self.train_X.copy(),
                            self.train_Y.copy(),
                            self.test_X.copy(),
                            self.test_Y.copy()).run()
                except Exception as e:
                    print()
                    print(e)
            else:
                break

if __name__ == '__main__':
    l = loader()
    l.run()
