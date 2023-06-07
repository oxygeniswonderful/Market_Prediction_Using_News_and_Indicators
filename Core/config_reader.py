import yaml

class config_reader:

    def __init__(self, path):
        self.path = path
    def read_config(self, key):
        with open(self.path, 'r') as config:
            data = yaml.load(config, Loader=yaml.FullLoader)
        return(data[key])