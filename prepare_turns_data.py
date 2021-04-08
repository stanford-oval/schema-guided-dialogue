import json
from generation import utterance_generator
import csv
import os
import random
import argparse
from pathlib import Path

UTTERANCE_TEMPLATE_DIR = "generation/utterance_templates"

utterance_gen = utterance_generator.TemplateUtteranceGenerator(
        UTTERANCE_TEMPLATE_DIR)

DIRS = ['train', 'dev', 'test']
K_VALUES = [1, 3, 5, 7]

def build_sliding_windows(utterances_in_dialogue, k):
	"""
	Given a list of utterances and a value k for the number of turns, 
	produce sliding windows of size k. Truncate as needed for utterances
	at the very beginning.

	Params:
	utterances_in_dialogue (List of turns):
	turn is of the form {'speaker': 'USER', 'utterance': 'some text', ... }

	k (int): number of turns

	Return:
	windows:
	List of (string, string) where (string, string) consists of the 
	entirety of the k turns + the ground truth output (separated by <s> character), and the robot generated output

	for k=1, windows is of the form [('USER: Book a ticket. <s> AGENT: Please confirm booking', 
							'Please confirm that you would like to book a ticket'), ...]
	"""
	windows = []
	
	# Produce sliding windows of size less than k, at the beginning of the dialogue
	for i in range(k):
		if i >= len(utterances_in_dialogue) - 1:
			continue

		last_turn = utterances_in_dialogue[i]
		last_speaker = last_turn["speaker"]
		last_utterance = last_turn["utterance"]

		if last_speaker == 'USER':
			continue

		utterances_in_truncated_window = utterances_in_dialogue[:i]
		k_turns_as_string = ''
		for i, turn in enumerate(utterances_in_truncated_window):
			k_turns_as_string += turn["speaker"] + ': ' + turn["utterance"] + ' <s> '

		robot_utterance = utterance_gen.get_robot_utterance(last_turn)
		
		k_turns_as_string += 'SYSTEM: ' + robot_utterance

		if len(robot_utterance) == 0 or len(last_utterance) == 0:
			continue
			
		k_turns_as_string = k_turns_as_string.replace('\n', ' ')
		last_utterance = last_utterance.replace('\n', ' ')

		windows.append((k_turns_as_string, last_utterance))


	# Produce sliding windows of size = k
	for i in range(len(utterances_in_dialogue) - k):
		idx_of_last_utterance = i + k

		last_turn = utterances_in_dialogue[idx_of_last_utterance]
		last_speaker = last_turn["speaker"]
		last_utterance = last_turn["utterance"]

		if last_speaker == 'USER':
			continue

		utterances_in_window = utterances_in_dialogue[i:idx_of_last_utterance]
		k_turns_as_string = ''
		for i, turn in enumerate(utterances_in_window):
			k_turns_as_string += turn["speaker"] + ': ' + turn["utterance"] + ' <s> '

		robot_utterance = utterance_gen.get_robot_utterance(last_turn)
		k_turns_as_string += 'SYSTEM: ' + robot_utterance

		if len(robot_utterance) == 0 or len(last_utterance) == 0:
			continue

		k_turns_as_string = k_turns_as_string.replace('\n', ' ')
		last_utterance = last_utterance.replace('\n', ' ')

		windows.append((k_turns_as_string, last_utterance))

	return windows


def read_json(json_file, k):
	"""
	Read a single json file into the proper format for the k
	turns experiment.

	Params:
	json_file (string): name of the json file to be read
	k (int): number of preceding turns

	Return:
	windows_in_file: List of (string, string) of the form
	[('USER: Book a ticket. <s> AGENT: Please confirm booking', 
				'Please confirm that you would like to book a ticket'), ...]
	"""
	f = open(json_file)
	data = json.load(f)

	windows_in_file = []
	for item in data:
		turns = item['turns']

		utterances_in_dialogue = []
		for turn in turns:
			utterances_in_dialogue.append(turn)

		windows_in_dialogue = build_sliding_windows(utterances_in_dialogue, k)
		windows_in_file += windows_in_dialogue

	return windows_in_file

def read_json_fewshot(dialog_ids, k):
	"""
	Reads all json files specified by dialog_ids.

	Params:
	dialogue List of (string): List of dialogue ids to be read
	k (int): number of preceding turns

	Return:
	windows_in_file: List of (string, string) of the form
	[('USER: Book a ticket. # AGENT: Please confirm booking', 
				'Please confirm that you would like to book a ticket'), ...]
	"""
	json_dir = 'sgd_dataset_dir/train'
	windows_in_file = []

	for file in os.listdir(json_dir):
		if file.startswith('schema'):
			continue

		f = open(json_dir + '/' + file)
		data = json.load(f)

		for item in data:
			dialogue_id = item['dialogue_id']
			if dialogue_id in dialog_ids:

				turns = item['turns']

				utterances_in_dialogue = []
				for turn in turns:
					utterances_in_dialogue.append(turn)

				windows_in_dialogue = build_sliding_windows(utterances_in_dialogue, k)
				windows_in_file += windows_in_dialogue

	return windows_in_file

def main():
	parser = argparse.ArgumentParser(
		description='Run ...')
	parser.add_argument('--k', dest='k', choices=[0, 1, 3, 5, 7], type=int, default=1)
	parser.add_argument('--data_dir', dest='data_dir', choices=['train', 'eval', 'valid'], default='train')
	parser.add_argument('--write_dir', dest='write_dir')
	parser.add_argument('--shuffle', dest='shuffle', action='store_true')
	parser.add_argument('--create_fewshot_split', dest='create_fewshot_split', action='store_true')
	parser.add_argument('--num_fewshot_examples', dest='num_fewshot_examples', choices=[5, 10, 20, 40, 80], type=int)

	opt = parser.parse_args()

	if not opt.create_fewshot_split:
		full_data_dir = os.path.join('sgd_dataset_dir', opt.data_dir)
		print("Dataset dir:", full_data_dir)
		
		full_write_dir = os.path.join(opt.write_dir, 'turns_' + str(opt.k), opt.data_dir + '.tsv')

		# create the directory and the nested turns directory if it doesn't exist
		Path(os.path.join(opt.write_dir, 'turns_' + str(opt.k))).mkdir(parents=True, exist_ok=True)

		
		all_sliding_windows = []
		for filename in os.listdir(full_data_dir):
			if filename.startswith("schema"):
				continue

			path_to_file = os.path.join(full_data_dir, filename)
			windows_in_file = read_json(path_to_file, opt.k)
			all_sliding_windows += windows_in_file

	else:
		full_write_dir = os.path.join(opt.write_dir, str(opt.num_fewshot_examples) + '_shot', 'turns_' + str(opt.k) + '.tsv')

		# create the directory and the nested turns directory if it doesn't exist
		Path(os.path.join(opt.write_dir, str(opt.num_fewshot_examples) + '_shot')).mkdir(parents=True, exist_ok=True)

		fewshot_ids_file = os.path.join('generation', 'fewshot_splits', str(opt.num_fewshot_examples) + '_shot.txt')
		dialog_ids = set()
		with open(fewshot_ids_file) as f:
			for line in f:
				dialog_ids.add(line.strip())

		all_sliding_windows = read_json_fewshot(dialog_ids, opt.k)

	if opt.shuffle:
		random.shuffle(all_sliding_windows)	# randomize the order

	with open(full_write_dir, 'w+') as f:
		for row in all_sliding_windows:
			f.write(row[0].strip() + '\t' + row[1] + '\n')

if __name__ == "__main__":
    main()
