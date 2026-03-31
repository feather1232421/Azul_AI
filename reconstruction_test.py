from logic import AzulGame
from config import *
import numpy as np
import json
from pydantic import BaseModel
from typing import List


class PlaceTokenAreaData(BaseModel):
    empty: bool
    color: int


class PlayerBoardData(BaseModel):
    score: int
    manualAreas: List[List[PlaceTokenAreaData]]
    coloredAreas: List[List[PlaceTokenAreaData]]
    loseAreas: List[PlaceTokenAreaData]


class TableData(BaseModel):
    factories: List[List[PlaceTokenAreaData]]
    center: List[PlaceTokenAreaData]
    me: PlayerBoardData
    opponents: List[PlayerBoardData]


if __name__ == "__main__":
    js = '{"factories":[[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":false,"color":1},{"empty":false,"color":2},{"empty":false,"color":2},{"empty":false,"color":1}],[{"empty":false,"color":5},{"empty":false,"color":2},{"empty":false,"color":2},{"empty":false,"color":2}],[{"empty":false,"color":4},{"empty":false,"color":5},{"empty":false,"color":3},{"empty":false,"color":2}],[{"empty":false,"color":4},{"empty":false,"color":2},{"empty":false,"color":2},{"empty":false,"color":2}]],"center":[{"empty":false,"color":0},{"empty":false,"color":4},{"empty":false,"color":1},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],"me":{"score":0,"manualAreas":[[{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"coloredAreas":[[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"loseAreas":[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]},"opponents":[{"score":0,"manualAreas":[[{"empty":false,"color":3}],[{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"coloredAreas":[[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"loseAreas":[{"empty":false,"color":3},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]}]}'

    data = json.loads(js)
    state = TableData(**data)
    # print(state)
    game = AzulGame.from_table_data(state)

    game.display_all_info()

