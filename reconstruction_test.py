from logic import AzulGame
from config import *
import numpy as np
import json
from pydantic import BaseModel
from typing import List


class TokenNumberData(BaseModel):
    color: int
    number: int


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
    remainTokens: List[TokenNumberData]
    loseTokens: List[TokenNumberData]


if __name__ == "__main__":
    js = '{"factories":[[{"empty":false,"color":5},{"empty":false,"color":4},{"empty":false,"color":4},{"empty":false,"color":2}],[{"empty":false,"color":3},{"empty":false,"color":1},{"empty":false,"color":3},{"empty":false,"color":5}],[{"empty":false,"color":4},{"empty":false,"color":3},{"empty":false,"color":4},{"empty":false,"color":2}],[{"empty":false,"color":2},{"empty":false,"color":2},{"empty":false,"color":5},{"empty":false,"color":3}],[{"empty":false,"color":3},{"empty":false,"color":5},{"empty":false,"color":2},{"empty":false,"color":2}]],"center":[{"empty":false,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],"me":{"score":-3,"manualAreas":[[{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"coloredAreas":[[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":false,"color":4},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"loseAreas":[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]},"opponents":[{"score":2,"manualAreas":[[{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":false,"color":1},{"empty":false,"color":1},{"empty":false,"color":1},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"coloredAreas":[[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":false,"color":4},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":false,"color":2},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":false,"color":2},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}],[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]],"loseAreas":[{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0},{"empty":true,"color":0}]}],"remainTokens":[{"color":1,"number":14},{"color":2,"number":6},{"color":3,"number":11},{"color":4,"number":14},{"color":5,"number":15}],"loseTokens":[{"color":3,"number":4},{"color":5,"number":1},{"color":2,"number":6},{"color":1,"number":2}]}'
    data = json.loads(js)
    state = TableData(**data)
    # print(state)
    game = AzulGame.from_table_data(state)

    game.display_all_info()
    print(game.public_board.bag)
    print(game.public_board.discard_pile)
