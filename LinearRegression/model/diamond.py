class Diamond:
    carat = 0
    cut = 0
    color = 0
    clarity = 0
    depth = 0
    table = 0
    x = 0
    y = 0
    z = 0
    price = 0

    @staticmethod
    def define_cut_value(cut):
        switch_cut = {
            "Fair": 1,
            "Good": 2,
            "Very Good": 3,
            "Premium": 4,
            "Ideal": 5
        }

        return switch_cut.get(cut, 0)

    @staticmethod
    def define_color_value(color):
        switch_color = {
            "J": 1,
            "I": 2,
            "H": 3,
            "G": 4,
            "F": 5,
            "E": 6,
            "D": 7
        }

        return switch_color.get(color, 0)

    @staticmethod
    def define_clarity_value(clarity):
        switch_clarity = {
            "I1": 1,
            "SI2": 2,
            "SI1": 3,
            "VS2": 4,
            "VS1": 5,
            "VVS2": 6,
            "VVS1": 7,
            "IF": 8
        }

        return switch_clarity.get(clarity, 0)

    def __init__(self, carat, cut, color, clarity, depth, table, x, y, z, price):
        self.carat = float(carat)
        self.cut = self.define_cut_value(cut)
        self.color = self.define_color_value(color)
        self.clarity = self.define_clarity_value(clarity)
        self.depth = float(depth)
        self.table = float(table)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.price = float(price)

    def __repr__(self):
        return "carat: {0} cut: {1} color: {2} clarity: {3} depth: {4} table: {5} x: {6} y: {7} z: {8} price: {9}"\
                .format(self.carat, self.cut, self.color,
                        self.clarity, self.depth, self.table,
                        self.x, self.y, self.z, self.price)
