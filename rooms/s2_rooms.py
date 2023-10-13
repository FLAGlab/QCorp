#%pip install matplotlib pandas --q

from board import Coordinates, generate_grid_world_board, Grid, generate_labyrinth, generate_taxi, TaxiGrid, GridBlockedPaths
from math import ceil
from matplotlib.pyplot import Rectangle, subplots, rcParams
from q_learning import QLearning, QLearningTaxi
from pandas import DataFrame, set_option

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 150)

rcParams['figure.figsize'] = [12, 8]
rcParams['figure.dpi'] = 100


def plot_scenario(scenario: QLearning):
    _, axes = subplots()
    axes.set_aspect('equal')
    axes.set_xlim(0, scenario.environment.dimensions.x)
    axes.set_ylim(scenario.environment.dimensions.y, 0)
    for x in range(scenario.environment.dimensions.x):
        for y in range(scenario.environment.dimensions.y):
            coordinates = Coordinates(x, y)
            if coordinates in scenario.environment.board:
                value = ceil((scenario.get_value(coordinates)) * 100) / 100
                color = 'white'
                qvalue = scenario.get_policy_qvalue(coordinates)
                if isinstance(scenario, QLearningTaxi):
                    if coordinates == scenario.environment.objective:
                        color = "green"
                    elif coordinates == scenario.environment.passenger:
                        color = "blue"
                    elif coordinates in scenario.environment.objectives:
                        color = "orange"
                else:
                    if coordinates == scenario.environment.objective:
                        color = "green"
                    elif coordinates == scenario.environment.start:
                        color = "blue"
                    elif value < 0:
                        color = "red"
                if qvalue != None:
                    qvalue = ceil(qvalue * 100) / 100
                    action = scenario.get_policy(coordinates)
                    action = "OUT" if action == None else action.name
                else:
                    qvalue = 0
                    action = "None"
                axes.add_patch(Rectangle((x, y), 1, 1, facecolor=color))
                if isinstance(scenario, QLearningTaxi):
                    text = "Con:{}\nQ:{:.2f}\nSin:{}\nQ:{:.2f}".format(scenario.get_policy_passenger(
                        coordinates).name, scenario.get_policy_qvalue_passenger(coordinates), action, qvalue)
                else:
                    text = "{}\nQ:{:.2f}\n{:.2f}".format(action, qvalue, value)
                axes.text(x + 0.5, y + 0.5, text, ha='center', va='center')
            else:
                axes.add_patch(Rectangle((x, y), 1, 1, facecolor='gray'))
    for (first, second) in scenario.environment.blocked_paths:
        if first.x == second.x:
            x = [first.x, first.x + 1]
            y = [second.y, second.y] if first.y < second.y else [first.y, first.y]
            axes.plot(x, y, color="black")
        elif first.y == second.y:
            y = [first.y, second.y + 1]
            x = [second.x, second.x] if first.x < second.x else [first.x, first.x]
            axes.plot(x, y, color="black")


def generate_q_table(scenario: QLearning) -> DataFrame:
    q_table = {"Estado": [], }
    for action in scenario.environment.get_actions():
        q_table[action.name] = []
    q_table["Objetivo"] = []
    for state in sorted(scenario.Q, key=lambda x: (x.x, x.y)):
        if len(scenario.Q[state]) < 2:
            continue
        q_table["Objetivo"].append(
            "Sí" if state == scenario.environment.objective else "No")
        q_table["Estado"].append(f"({state.x}, {state.y})")
        for action in scenario.Q[state]:
            q_table[action.name].append(scenario.Q[state][action])
    return DataFrame(q_table)


random_seed = 5657656
