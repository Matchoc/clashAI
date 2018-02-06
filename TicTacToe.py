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
from sklearn.neural_network import MLPClassifier, MLPRegressor

PRINT_LEVEL=0
DATA_FOLDER = "data"
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		#sys.stdout.write((str(msg) + "\n").encode('UTF-8'))
		if sys.version_info[0] < 3:
			print((str(msg) + "\n").encode('UTF-8'))
		else:
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
		
	def X(self):
		return self.board.reshape(self.size * self.size)
		
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
		
	def is_valid_move(self, x,y):
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
		
		if type(Q) is MLPRegressor:
			possible_actions = Q.predict([cur_state.X()])
			possible_actions = [(x, possible_actions[x]) for x in range(len(possible_actions))]
			myprint(str(possible_actions),5)
			possible_actions = sorted(possible_actions, key=lambda x: x[1], reverse=True)
			myprint(str(possible_actions),5)
			#index = numpy.argmax(possible_actions)
			for val in possible_actions:
				action = (int(val[0]) % cur_state.size, int(val[0]) / cur_state.size)
				if cur_state.is_valid_move(*action):
					break
			
		else:
			if repr(cur_state) not in Q:
				init_cost_action(Q, cur_state, cur_state.get_valid_moves_list())
			action = find_best_action(Q[repr(cur_state)])
		
		if move % 2 == 0:
			x_moves.append([cur_state, action])
			winner = cur_state.play_x(*action)
		else:
			o_moves.append([cur_state, action])
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
	
def play_interactive(Q, final_game):
	won = None
	symbols = [X, O]
	players = ['Player', 'AI']
	#players = ['AI', 'Player']
	myprint(str(final_game), 10)
	move_count = 0
	while not won and move_count < 9:
		p = players[0]
		s = symbols[0]
		if p == 'Player':
			move = None
			while move is None:
				try:
					move = input('Your turn (0-8): ')  # Python 3
					x = int(move) % final_game.size
					y = int(move) / final_game.size
					
					print('playing ' + str((x, y)))
					if final_game.is_valid_move(x, y):
						if s == X:
							won = final_game.play_x(x, y)
						else:
							won = final_game.play_o(x, y)
						move_count += 1
					else:
						print('Invalid move ' + str(move))
						move = None
				except Exception as e:
					print(e)
					print('Invalid move ' + str(move))
					print(final_game.board)
					move = None
		else:
			ai_move, won = play_a_move(Q, final_game, s)
			move_count += 1
		
		myprint(str(final_game), 10)

		if won:
			myprint(str(p) + ' won the game !',5)
		elif move_count >= 9:
			myprint("This game ended in a NULL",5)

		del players[0]
		players += [p]
		del symbols[0]
		symbols += [s]
	
def train_using_Q_table(board_size):
	Q = {}
	for i in range(1000):
		winner_moves, loser_moves, is_null = play_a_game(Q, board_size)
		if is_null:
			back_propagate(Q, -10.0, repr(winner_moves))
			back_propagate(Q, -10.0, repr(loser_moves))
		else:
			back_propagate(Q, 100.0, repr(winner_moves))
			back_propagate(Q, -100.0, repr(loser_moves))
	
	final_game = TicTacToe(board_size)
	play_interactive(Q, final_game)
	
def train_using_MLP(board_size):
	X = [[0,0,0,0,0,0,0,0,0]]
	y = [[100,0,0,0,0,0,0,0,0]]
	MACHINE_ALL = MLPRegressor(solver='sgd', alpha=10.0, hidden_layer_sizes=(150, 29), random_state=1000, activation="relu", max_iter=4000, batch_size=1)
	MACHINE_ALL.partial_fit(X, y)
	
	winner_moves, loser_moves, is_null = play_a_game(MACHINE_ALL, board_size)
	
	
if __name__ == '__main__':
	board_size = 3
	
	train_using_MLP(board_size)
	
	#train_using_Q_table(board_size)
	