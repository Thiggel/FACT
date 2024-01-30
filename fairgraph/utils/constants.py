from enum import Enum


class Datasets(str, Enum):
    NBA = "NBA"
    POKEC_N = "POKEC_N"
    POKEC_Z = "POKEC_Z"
    SYNTHETIC = "SYNTHETIC"


pareto_fronts = {
    "Graphair": {
        Datasets.NBA: [
            [1.49, 67.5],
            [1.515, 68.0],
            [1.75, 68.2],
            [2.07, 68.4],
            [2.56, 69.36],
        ],
        Datasets.POKEC_Z: [
            [0.5, 66.2],
            [1.2, 66.6],
            [1.45, 67.2],
            [2.1, 68.17],
        ],
        Datasets.POKEC_N: [
            [0.75, 65.25],
            [0.8, 66.05],
            [0.9, 66.7],
            [1.4, 67.2],
            [2.02, 67.43],
        ],
    },
    "FairDrop": {
        Datasets.NBA: [
            [3.45, 66.01],
            [3.66, 69.01],
        ],
        Datasets.POKEC_Z: [
            [1.5, 63],
            [5.78, 67.8],
        ],
        Datasets.POKEC_N: [
            [2.48, 66.8],
            [4.05, 67.42],
        ]
    },
    "NIFTY": {
        Datasets.NBA: [
            [2.82, 66],
            [3.1, 67.55],
            [3.3, 69.95],
        ],
        Datasets.POKEC_Z: [
            [1.58, 64.95],
            [4.4, 67.1],
        ],
        Datasets.POKEC_N: [
            [6.6, 64.5],
            [6.9, 65.48],
        ],
    },
    "FairAug": {
        Datasets.NBA: [
            [1.78, 64.6],
            [3.7, 65.6],
            [4.99, 66.38],
        ],
        Datasets.POKEC_Z: [
            [3.63, 67.75],
            [4.4, 68.6],
            [5.28, 69.17],
        ],
        Datasets.POKEC_N: [
            [3.7, 67.1],
            [4.85, 67.75],
            [5.1, 68.61],
        ],
    },
}


colors = {
    "Graphair": "cornflowerblue",
    "FairDrop": "orange",
    "NIFTY": "green",
    "FairAug": "red",
}