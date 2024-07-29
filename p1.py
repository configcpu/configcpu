# a Program to Implement Tic-Tac-Toe game using Python

import os
import time

board = ["", "", "", "", "", "", "", "", ""]
player = 1

win = 1
draw = -1
running = 0

Game = running
mark = "X"

def DrawBoard():
    print("%c | %c | %c" , (board[0], board[1], board[2]))
    print("-------------")
    print("%c | %c | %c", (board[3], board[4], board[5]))
    print("-------------")
    print("%c | %c | %c" , (board[6], board[7], board[8]))

def CheckPosition(x):
    if board[x - 1] == "":
        return True
    else:
        return False

def CheckWin():
    global Game
    if board[0] == board[1] and board[1] == board[2] and board[0] != "":
        Game = win
    elif board[3] == board[4] and board[4] == board[5] and board[3] != "":
        Game = win
    elif board[6] == board[7] and board[7] == board[8] and board[6] != "":
        Game = win
    elif board[0] == board[3] and board[3] == board[6] and board[0] != "":
        Game = win
    elif board[1] == board[4] and board[4] == board[7] and board[1] != "":
        Game = win
    elif board[2] == board[5] and board[5] == board[8] and board[2] != "":
        Game = win
    elif (
        board[0] != ""
        and board[1] != ""
        and board[2] != ""
        and board[3] != ""
        and board[4] != ""
        and board[5] != ""
        and board[6] != ""
        and board[7] != ""
        and board[8] != ""
    ):
        Game = draw
    else:
        Game = running

print("Tic - Tac - Toe Game")
print("Player 1 [x] --- Player 2 [o] \n")
print()
print()
print("Please wait")
time.sleep(3)

while Game == running:
    os.system("clear")  # Use os.system("cls") on Windows
    DrawBoard()
    if player % 2 != 0:
        print("Player 1's chance")
        mark = "X"
    else:
        print("Player 2's chance")
        mark = "O"
    choice = int(input("Enter position between [1-9]: "))
    if CheckPosition(choice):
        board[choice - 1] = mark
        player += 1
        CheckWin()

os.system("clear")  # Use os.system("cls") on Windows
DrawBoard()
if Game == draw:
    print("Game Draw")
elif Game == win:
    player -= 1
    if player % 2 == 0:
        print("Player 1 won")
    else:
        print("Player 2 won")

