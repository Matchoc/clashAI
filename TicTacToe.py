import sys
import os
os.environ["path"] = os.path.dirname(sys.executable) + ";" + os.environ["path"]
import glob
import win32gui
import win32ui
import win32con
import win32api
import datetime
import dateutil.relativedelta
import operator
import random
import numpy
import json
import scipy.ndimage
import multiprocessing
import matplotlib.pyplot as plt
import cv2
from PIL import Image

PRINT_LEVEL=4
DATA_FOLDER = "data"
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		sys.stdout.buffer.write((str(msg) + "\n").encode('UTF-8'))
		
class ScopedTimer:
	totals = {}
	def __init__(self, name, level=3):
		self.starttime = datetime.datetime.now()
		self.name = name
		self.level = level
		
	def __del__(self):
		delta = datetime.datetime.now() - self.starttime
		if self.name not in ScopedTimer.totals:
			ScopedTimer.totals[self.name] = datetime.timedelta(0)
		ScopedTimer.totals[self.name] += delta
		myprint("{name} : {delta} / {total}".format(name=self.name, delta=str(delta), total=str(ScopedTimer.totals[self.name])), self.level)
		#myprint(str(self.name) + " : " + str(delta),self.level)

EMPTY = 0
X = 1
O = 2
class TicTacToe:
	def __init__(self, size=3):
		self.size = size
		self.board = numpy.zeros(size * size, numpy.int8)
		self.board = self.board.reshape(size,size)
		
	def __str__(self):
		line = ""
		for y in range(self.size):
			line += "\t"
			for x in range(self.size):
				if self.board[x][y] == EMPTY:
					line += " "
				if self.board[x][y] == X:
					line += "X"
				if self.board[x][y] == O:
					line += "O"
				if x < self.size - 1:
					line += "\t|\t"
			line += "\n"
			if y < self.size - 1:
				line += "-" * (self.size * 25)
				line += "\n"
		return line
		
	def __repr__(self):
		return "".join(map(str,self.board.reshape(self.size * self.size).tolist()))
		
	def play_x(self, x, y):
		self.board[x][y] = X
		if self.winner():
			return True
		else:
			return False
		
	def play_o(self, x, y):
		self.board[x][y] = O
		if self.winner():
			return True
		else:
			return False
		
	def play(self, owner, x, y):
		self.board[x][y] = owner
		if self.winner():
			return True
		else:
			return False
		
	def is_valid_move(x,y):
		return self.board[x][y] == 0
		
	def get_valid_moves_list(self):
		valid_move_list = []
		for x in range(self.size):
			for y in range(self.size):
				if self.board[x][y] == EMPTY:
					valid_move_list.append( (x,y) )
		return valid_move_list
		
	def winner(self):
		full_line = False
		for a in range(len(self.board)):
			row = self.board[a]
			full_line |= (row[0] != EMPTY and row.tolist().count(row[0]) == len(row))
			col = self.board[:,a]
			full_line |= (col[0] != EMPTY and col.tolist().count(col[0]) == len(col))
		diag = numpy.diagonal(self.board)
		full_line |= (diag[0] != EMPTY and diag.tolist().count(diag[0]) == len(diag))
		diag = numpy.diagonal(self.board[::-1,:])
		full_line |= (diag[0] != EMPTY and diag.tolist().count(diag[0]) == len(diag))
		
		return full_line
	
	
def init_cost_action(Q, state, valid_actions):
	res = {action : 0 for action in valid_actions}
	Q[repr(state)] = res
	
def get_max(Q_row):
	Q_values = numpy.array(list(Q_row.values()))
	max_val = Q_values[Q_values.argmax()]
	return max_val
	
Discount_factor = 0.9
Learning_rate = 0.5
def back_propagate(Q, leaf_reward, moves):
	prev = None
	for state, action in reversed(moves):
		if prev == None:
			Q[state][action] = leaf_reward
			prev = state
			continue
		
		max = get_max(Q[prev])
		Q[state][action] = Q[state][action] + Learning_rate * (0 + Discount_factor * max - Q[state][action])
		prev = state
	
def find_best_action(Q_row):
	Q_values = numpy.array(list(Q_row.values()))
	max_val = Q_values[Q_values.argmax()]
	ideal_moves = []
	for entry in Q_row:
		if Q_row[entry] == max_val:
			ideal_moves.append(entry)
			
	return random.choice(ideal_moves)
	
def play_a_move(Q, cur_state, turn):
	if repr(cur_state) not in Q:
		init_cost_action(Q, cur_state, cur_state.get_valid_moves_list())
	action = find_best_action(Q[repr(cur_state)])
	return action, cur_state.play(turn, *action)
	
def play_a_game(Q, size):
	cur_state = TicTacToe(size)
	x_moves = []
	o_moves = []
	move = 0
	winner = False
	while winner == False and move < size * size:
		if repr(cur_state) not in Q:
			init_cost_action(Q, cur_state, cur_state.get_valid_moves_list())
		
		action = find_best_action(Q[repr(cur_state)])
		
		if move % 2 == 0:
			x_moves.append([repr(cur_state), action])
			winner = cur_state.play_x(*action)
		else:
			o_moves.append([repr(cur_state), action])
			winner = cur_state.play_o(*action)
		
		if not winner:
			move += 1
	
	is_null = False
	if move >= size * size:
		myprint("Game ended in NULL",3)
		winner_moves = x_moves
		loser_moves = o_moves
		is_null = True
	elif move % 2 == 0:
		myprint("X Won", 3)
		winner_moves = x_moves
		loser_moves = o_moves
	else:
		myprint("O Won", 3)
		winner_moves = o_moves
		loser_moves = x_moves
	myprint(str(cur_state),2)
	
	return winner_moves, loser_moves, is_null
	
if __name__ == '__main__':
	Q = {}
	board_size = 3
	for i in range(10000):
		winner_moves, loser_moves, is_null = play_a_game(Q, board_size)
		if is_null:
			back_propagate(Q, -10.0, winner_moves)
			back_propagate(Q, -10.0, loser_moves)
		else:
			back_propagate(Q, 100.0, winner_moves)
			back_propagate(Q, -100.0, loser_moves)
	
	final_game = TicTacToe(board_size)
	final_game.play_x(0,0)
	ai_move, won = play_a_move(Q, final_game, O)
	final_game.play_x(2,2)
	ai_move, won = play_a_move(Q, final_game, O)
	final_game.play_x(0,1)
	ai_move, won = play_a_move(Q, final_game, O)
	
	myprint(str(final_game), 10)
	
	
	
	
	
		
	
	