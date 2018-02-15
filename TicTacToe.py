import sys
import os
os.environ["path"] = os.path.dirname(sys.executable) + ";" + os.environ["path"]
import glob
import copy
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
import pickle
import scipy.ndimage
import multiprocessing
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier, MLPRegressor

PRINT_LEVEL=4
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
#REWARD = 100.0
#LOSS = -100.0
#NULL = -10.0
REWARD = 1.0
LOSS = -1.0
NULL = -0.1
class TicTacToe:
	def __init__(self, size=3, fromstr=None):
		self.size = size
		self.board = numpy.zeros(size * size, numpy.int8)
		if fromstr is not None:
			index = 0
			for move in list(fromstr):
				key = int(move)
				self.board[index] = key
				index += 1
		self.board = self.board.reshape(size,size)
		
	def __str__(self):
		line = ""
		for y in range(self.size):
			line += "\t"
			for x in range(self.size):
				if self.board[y][x] == EMPTY:
					line += " "
				if self.board[y][x] == X:
					line += "X"
				if self.board[y][x] == O:
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
		self.board[y][x] = X
		if self.winner():
			return True
		else:
			return False
		
	def play_o(self, x, y):
		self.board[y][x] = O
		if self.winner():
			return True
		else:
			return False
		
	def play(self, owner, x, y):
		self.board[y][x] = owner
		if self.winner():
			return True
		else:
			return False
		
	def is_valid_move(self, x,y):
		return self.board[y][x] == 0
		
	def get_valid_moves_list(self):
		valid_move_list = []
		for x in range(self.size):
			for y in range(self.size):
				if self.board[y][x] == EMPTY:
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
Epsilon = 1.0
EpsilonStep = 0.1
EpsilonParts = 11
def back_propagate(Q, leaf_reward, moves):
	prev = None
	for obj_state, action in reversed(moves):
		state = repr(obj_state)
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
	
def to_index(x, y, board_size):
	return x + y * board_size
	
def to_xy(move, board_size):
	x = int(move) % board_size
	y = int(int(move) / board_size)
	return x,y
	
def play_a_move(Q, cur_state, turn):
	if type(Q) is MLPRegressor:
		possible_actions = Q.predict([cur_state.X()])
		possible_actions = [(x, possible_actions[0][x]) for x in range(len(possible_actions[0]))]
		possible_actions = sorted(possible_actions, key=lambda x: x[1], reverse=True)
		myprint(repr(cur_state) + " : " + str(possible_actions))
		#index = numpy.argmax(possible_actions)
		for val in possible_actions:
			action = to_xy(val[0], cur_state.size)
			if cur_state.is_valid_move(*action):
				break
	else:
		if repr(cur_state) not in Q:
			init_cost_action(Q, cur_state, cur_state.get_valid_moves_list())
		action = find_best_action(Q[repr(cur_state)])
		
	return action, cur_state.play(turn, *action)
	
def play_a_game(Q, size, epsilon=0.0):
	cur_state = TicTacToe(size)
	x_moves = []
	o_moves = []
	move = 0
	winner = False
	while winner == False and move < size * size:
		
		if type(Q) is MLPRegressor:
			rnd = random.random()
			possible_actions = Q.predict([cur_state.X()])
			possible_actions = [(x, possible_actions[0][x]) for x in range(len(possible_actions[0]))]
			possible_actions = sorted(possible_actions, key=lambda x: x[1], reverse=True)
			myprint("Possible Actions (" + repr(cur_state) + ") : " + str(possible_actions))
			#index = numpy.argmax(possible_actions)
			for val in possible_actions:
				action = to_xy(val[0], cur_state.size)
				if rnd < epsilon:
					action = to_xy(random.choice(possible_actions)[0], cur_state.size)
					myprint("Choosing Randomly {} / {}".format(rnd, epsilon),2)
				if cur_state.is_valid_move(*action):
					break
			
		else:
			if repr(cur_state) not in Q:
				init_cost_action(Q, cur_state, cur_state.get_valid_moves_list())
			action = find_best_action(Q[repr(cur_state)])
		
		if move % 2 == 0:
			x_moves.append([copy.deepcopy(cur_state), action])
			winner = cur_state.play_x(*action)
		else:
			o_moves.append([copy.deepcopy(cur_state), action])
			winner = cur_state.play_o(*action)
		
		myprint(str(cur_state))
		
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
					x, y = to_xy(move, final_game.size)
					
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
	for i in range(10000):
		winner_moves, loser_moves, is_null = play_a_game(Q, board_size)
		if is_null:
			back_propagate(Q, NULL, winner_moves)
			back_propagate(Q, NULL, loser_moves)
		else:
			back_propagate(Q, REWARD, winner_moves)
			back_propagate(Q, LOSS, loser_moves)
	
	save_Q_table(Q)
	
	final_game = TicTacToe(board_size)
	play_interactive(Q, final_game)
	
def MLP_training(machine, moves, board_size, reward):
	X = []
	new_y = []
	next_state = None
	next_action = None
	next_adjusted_y = None
	for state, action in reversed(moves):
		X.append(state.X())
		index = to_index(*action, board_size)
		
		if next_state is None:
			max_Q = reward / Discount_factor
		else:
			#estimated_ynext = machine.predict([next_state.X()])
			max_Q = max(next_adjusted_y)
			
		estimated_y = machine.predict([state.X()])
		estimated_y[0][index] = estimated_y[0][index] + (Discount_factor * max_Q)
		new_y.append(estimated_y[0])
		next_state = state
		next_action = action
		next_adjusted_y = estimated_y[0]
	
	return X, new_y
	
def run_MLP_game(machine, board_size, epsilon):
	winner_moves, loser_moves, is_null = play_a_game(machine, board_size, epsilon)

	X, new_y = MLP_training(machine, winner_moves, board_size, NULL if is_null else REWARD)
	X2, new_y2 = MLP_training(machine, loser_moves, board_size, NULL if is_null else LOSS)
	
	X.extend(X2)
	new_y.extend(new_y2)
	
	myprint("partial_fit X : " + str(X))
	myprint("partial_fit y : " + str(new_y))
	machine.partial_fit(X, new_y)
	
def save_Q_table(Q):
	with open("q_table.save", 'wb') as f:
		pickle.dump(Q, f)
	
def load_Q_table():
	with open("q_table.save", 'rb') as f:
		Q = pickle.load(f)
	return Q
	
def save_machine(MACHINE_ALL):
	joblib.dump(MACHINE_ALL, 'machine.save')
		
def load_machine():
	return joblib.load('machine.save')
	
def train_machine():
	X = [[0,0,0,0,0,0,0,0,0]]
	y = [[0,0,0,0,0,0,0,0,0]]
	#MACHINE_ALL = MLPRegressor(solver='sgd', alpha=1.0, hidden_layer_sizes=(1500, 29), random_state=1000, activation="relu", max_iter=4000, batch_size=5, learning_rate="constant", learning_rate_init=0.001)
	MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0005, alpha=0.00001, hidden_layer_sizes=(350,185), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 19
	MACHINE_ALL.partial_fit(X, y)
	
	max_game = 25000
	actual_epsilon = Epsilon
	dec_every = int(max_game / EpsilonParts)
	for i in range(max_game):
		if i % 10 == 0:
			myprint("Game {} of {}".format(i, max_game),5)
		run_MLP_game(MACHINE_ALL, board_size, actual_epsilon)
		if i % dec_every == 0:
			actual_epsilon -= EpsilonStep
			if actual_epsilon < 0.0 :
				actual_epsilon = 0.0
			myprint("Epsilon now : " + str(actual_epsilon),2)
		
	save_machine(MACHINE_ALL)
		
	return MACHINE_ALL
	
def train_MLP_using_saved_Q_table(board_size):
	a = ScopedTimer("train_MLP_using_saved_Q_table",5)
	Q = load_Q_table()
	#myprint("Q : " + str(Q))
	X = []
	y = []
	
	for state_str in Q:
		state_obj = TicTacToe(board_size, state_str)
		X.append(state_obj.X())
		cur_y = []
		for y_coord in range(board_size):
			for x_coord in range(board_size):
				if (x_coord,y_coord) in Q[state_str]:
					cur_y.append(Q[state_str][(x_coord,y_coord)])
				else:
					cur_y.append(0.0) # maybe should append like -1000 ?
				
		y.append(cur_y)
		
	# fake more training data to help regression
	#for z in range(2):
	#	X += X
	#	y += y
	#myprint("X : " + str(X))
	#myprint("y : " + str(y))
	#MACHINE_ALL = MLPRegressor(solver='sgd',    alpha=0.0001, hidden_layer_sizes=(350,75), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # 67 loss # home 208
	#MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0005, alpha=0.00005, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 175
	#MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0001, alpha=0.00001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.001) # home 170
	#MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0001, alpha=0.00001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.003) # home 166
	#MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0001, alpha=0.00001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 162
	#MACHINE_ALL = MLPRegressor(solver='sgd', 			   alpha=0.0001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 151
	#MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0005, alpha=0.00001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 124
	#MACHINE_ALL = MLPRegressor(solver='sgd', alpha=0.00001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 120
	#MACHINE_ALL = MLPRegressor(solver='sgd', alpha=0.0001, hidden_layer_sizes=(350,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # 41 loss
	#MACHINE_ALL = MLPRegressor(solver='sgd', alpha=0.01, hidden_layer_sizes=(350,75), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # 89 loss
	#MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0005, alpha=0.00001, hidden_layer_sizes=(500,85), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # home 95
	MACHINE_ALL = MLPRegressor(solver='sgd', tol=0.0000001, alpha=0.00001, hidden_layer_sizes=(350,185), random_state=1000, activation="logistic", max_iter=4000, learning_rate="adaptive", learning_rate_init=0.002) # 3 loss # home 19
	 
	MACHINE_ALL.fit(X, y)
	myprint("loss : {}, n_iter : {}".format(MACHINE_ALL.loss_, MACHINE_ALL.n_iter_),5)
	
	del a
	final_game = TicTacToe(board_size)
	play_interactive(MACHINE_ALL, final_game)
		
	
def train_using_MLP(board_size):
	#MACHINE_ALL = load_machine()
	MACHINE_ALL = train_machine()
	
	final_game = TicTacToe(board_size)
	play_interactive(MACHINE_ALL, final_game)
	
if __name__ == '__main__':
	board_size = 3
	
	#b = '100011202'
	#a = TicTacToe(3, b)
	#myprint(str(a))
	
	train_MLP_using_saved_Q_table(board_size)
	
	#train_using_MLP(board_size)
	
	#train_using_Q_table(board_size)
	