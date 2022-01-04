
class MakeDataset:
    def __init__(self, crypto_currencies: list = ['BTC', 'ETH', 'SOL', 'ADA', 'SHIB', 'LTC']):
        self.crypto_currencies = crypto_currencies

    def update_datasets(self):
