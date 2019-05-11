import csv
import pandas as pd

from model.diamond import Diamond


class DiamondCsvReader:
    @staticmethod
    def read_csv(filename):
        diamond_list = []

        with open(filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                diamond = Diamond(row['carat'], row['cut'], row['color'], row['clarity'],
                                  row['depth'], row['table'], row['x'], row['y'], row['z'], row['price'])
                diamond_list.append(diamond)

        return diamond_list

    @staticmethod
    def get_data_frame(filename):
        diamond_list = DiamondCsvReader.read_csv(filename)

        dictionary = {
            'carat': [],
            'cut': [],
            'color': [],
            'clarity': [],
            'x': [],
            'y': [],
            'z': [],
            'depth': [],
            'table': [],
            'price': []
        }

        columns = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z', 'depth', 'table', 'price']

        for diamond in diamond_list:
            dictionary['carat'].append(diamond.carat)
            dictionary['cut'].append(diamond.cut)
            dictionary['color'].append(diamond.color)
            dictionary['clarity'].append(diamond.clarity)
            dictionary['x'].append(diamond.x)
            dictionary['y'].append(diamond.y)
            dictionary['z'].append(diamond.z)
            dictionary['depth'].append(diamond.depth)
            dictionary['table'].append(diamond.table)
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)

    @staticmethod
    def get_nominal_data_frame(filename):
        diamond_list = DiamondCsvReader.read_csv(filename)

        dictionary = {
            'carat': [],
            'nominals': [],
            'x': [],
            'y': [],
            'z': [],
            'depth': [],
            'table': [],
            'price': []
        }

        columns = ['carat', 'nominals', 'x', 'y', 'z', 'depth', 'table', 'price']

        for diamond in diamond_list:
            dictionary['carat'].append(diamond.carat)
            dictionary['nominals'].append(diamond.cut*diamond.color*diamond.clarity)
            dictionary['x'].append(diamond.x)
            dictionary['y'].append(diamond.y)
            dictionary['z'].append(diamond.z)
            dictionary['depth'].append(diamond.depth)
            dictionary['table'].append(diamond.table)
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)

    @staticmethod
    def get_volume_data_frame(filename):
        diamond_list = DiamondCsvReader.read_csv(filename)

        dictionary = {
            'carat': [],
            'cut': [],
            'color': [],
            'clarity': [],
            'volume': [],
            'depth': [],
            'table': [],
            'price': []
        }

        columns = ['carat', 'cut', 'color', 'clarity', 'volume', 'depth', 'table', 'price']

        for diamond in diamond_list:
            dictionary['carat'].append(diamond.carat)
            dictionary['cut'].append(diamond.cut)
            dictionary['color'].append(diamond.color)
            dictionary['clarity'].append(diamond.clarity)
            dictionary['volume'].append(diamond.x*diamond.y*diamond.z*diamond.carat)
            dictionary['depth'].append(diamond.depth)
            dictionary['table'].append(diamond.table)
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)

    @staticmethod
    def get_two_vars_data_frame(filename):
        diamond_list = DiamondCsvReader.read_csv(filename)

        dictionary = {
            'carat': [],
            'volume': [],
            'price': []
        }

        columns = ['carat', 'volume', 'price']

        for diamond in diamond_list:
            dictionary['carat'].append(diamond.carat)
            dictionary['volume'].append(diamond.x*diamond.y*diamond.z*diamond.carat)
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)

    @staticmethod
    def get_four_vars_data_frame(filename):
        diamond_list = DiamondCsvReader.read_csv(filename)

        dictionary = {
            'carat': [],
            'x': [],
            'y': [],
            'z': [],
            'price': []
        }

        columns = ['carat', 'x', 'y', 'z', 'price']

        for diamond in diamond_list:
            dictionary['carat'].append(diamond.carat)
            dictionary['x'].append(diamond.x)
            dictionary['y'].append(diamond.y)
            dictionary['z'].append(diamond.z)
            dictionary['price'].append(diamond.price)

        return pd.DataFrame(dictionary, columns=columns)
