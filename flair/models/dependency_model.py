import warnings
import logging
from pathlib import Path

import sys
sys.path.append('...')

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch

from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import Result, store_embeddings
#-----------ner_dp-----------------
from .biaffine_dp import projection, bilinear_classifier, BiLSTM_1

from tabulate import tabulate
import numpy as np
import pdb
import copy
import time

import sys

from flair.parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TrilinearScorer
from flair.parser.utils.alg import eisner, crf
from flair.parser.utils.metric import Metric
from flair.parser.utils.fn import ispunct, istree, numericalize_arcs

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
								pad_sequence)

import torch.nn.functional as F
from torch_struct import DependencyCRF, NonProjectiveDependencyCRF
from .mst_decoder import MST_inference
def process_potential(log_potential):
	# (batch, sent_len+1, sent_len+1) or (batch, sent_len+1, sent_len+1, labels)
	
	# (batch, sent_len)
	root_score = log_potential[:,1:,0]
	# convert (dependency, head) to (head, dependency)
	# (batch, sent_len, sent_len)
	log_potential = log_potential.transpose(1,2)[:,1:,1:]
	batch, sent_len = log_potential.shape[:2]
	# Remove the <ROOT> and put the root probability in the diagonal part 
	log_potential[:,torch.arange(sent_len),torch.arange(sent_len)] = root_score
	return log_potential

def get_struct_predictions(dist):
	# (batch, sent_len, sent_len) | head, dep
	argmax_val = dist.argmax
	batch, sent_len, _ = argmax_val.shape
	res_val = torch.zeros([batch,sent_len+1,sent_len+1]).type_as(argmax_val)
	res_val[:,1:,1:] = argmax_val
	res_val = res_val.transpose(1,2)
	# set diagonal part to heads
	res_val[:,:,0] = res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)]
	res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)] = 0
	
	return res_val.argmax(-1)

def convert_score_back(marginals):
	# (batch, sent_len, sent_len) | head, dep
	batch = marginals.shape[0]
	sent_len = marginals.shape[1]
	res_val = torch.zeros([batch,sent_len+1,sent_len+1]+list(marginals.shape[3:])).type_as(marginals)
	res_val[:,1:,1:] = marginals
	res_val = res_val.transpose(1,2)
	# set diagonal part to heads
	res_val[:,:,0] = res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)]
	res_val[:,torch.arange(sent_len+1),torch.arange(sent_len+1)] = 0
	
	return res_val

def generate_tree(log_potential, mask, is_mst=False):
	# input: scores with shape of (batch, sent_len+1, sent_len+1)| (dependency, head) relations, including the <ROOT> token
	# is_mst: use non projective tree algorithm (MST), other wise Eisner's

	# Return: the distribution of CRF
	# ----------------------------------------------------------------------------------------
	log_potential = process_potential(log_potential)
	
	if is_mst:
		dist = NonProjectiveDependencyCRF(log_potential, lengths=mask.sum(-1))
	else:
		dist = DependencyCRF(log_potential, lengths=mask.sum(-1))
	return dist

def is_punctuation(word, pos, punct_set=None):
	if punct_set is None:
		return is_uni_punctuation(word)
	else:
		return pos in punct_set

import uuid
uid = uuid.uuid4().hex[:6]
  


log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
	return var.view(-1).detach().tolist()[0]


def argmax(vec):
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)


def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
	_, idx = torch.max(vecs, 1)
	return idx


def log_sum_exp_batch(vecs):
	maxi = torch.max(vecs, 1)[0]
	maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
	recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
	return maxi + recti_

def log_sum_exp_vb(vec, m_size):
	"""
	calculate log of exp sum

	args:
		vec (batch_size, vanishing_dim, hidden_dim) : input tensor
		m_size : hidden_dim
	return:
		batch_size, hidden_dim
	"""
	_, idx = torch.max(vec, 1)  # B * 1 * M
	max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

	return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
																												m_size)  # B * M

def pad_tensors(tensor_list):
	ml = max([x.shape[0] for x in tensor_list])
	shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
	template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
	lens_ = [x.shape[0] for x in tensor_list]
	for i, tensor in enumerate(tensor_list):
		template[i, : lens_[i]] = tensor

	return template, lens_


# Part of Codes are from https://github.com/yzhangcs/biaffine-parser
class SemanticDependencyParser(flair.nn.Model):
	def __init__(
		self,
		hidden_size: int,
		embeddings: TokenEmbeddings,
		tag_dictionary: Dictionary,
		tag_type: str,
		use_crf: bool = False,
		use_rnn: bool = False,
		train_initial_hidden_state: bool = False,
		punct: bool = False, # ignore all punct in default
		tree: bool = False, # keep the dpendency with tree structure
		n_mlp_arc = 500,
		n_mlp_rel = 100,
		mlp_dropout = .33,
		use_second_order = False,
		token_loss = False,
		n_mlp_sec = 150,
		init_std = 0.25,
		factorize = True,
		use_sib = True,
		use_gp = True,
		use_cop = False,
		iterations = 3,
		binary = True,
		is_mst = False,
		rnn_layers: int = 3,
		lstm_dropout: float = 0.33,
		dropout: float = 0.0,
		word_dropout: float = 0.33,
		locked_dropout: float = 0.5,
		pickle_module: str = "pickle",
		interpolation: float = 1.,
		factorize_interpolation: float = 0.025,
		config = None,
		use_decoder_timer = True,
		debug = False,
		target_languages = 1,
		word_map = None,
		char_map = None,
		relearn_embeddings = False,
		distill_arc: bool = False,
		distill_rel: bool = False,
		distill_crf: bool = False,
		distill_posterior: bool = False,
		distill_prob: bool = False,
		distill_factorize: bool = False,
		crf_attention: bool = False,
		temperature: float = 1,
		diagonal: bool = False,
		is_srl: bool = False,
		embedding_selector = False,
		):
		"""
		Initializes a SequenceTagger
		:param hidden_size: number of hidden states in RNN
		:param embeddings: word embeddings used in tagger
		:param tag_dictionary: dictionary of tags you want to predict
		:param tag_type: string identifier for tag type
		:param use_crf: if True use CRF decoder, else project directly to tag space
		:param use_rnn: if True use RNN layer, otherwise use word embeddings directly
		:param rnn_layers: number of RNN layers
		:param dropout: dropout probability
		:param word_dropout: word dropout probability
		:param locked_dropout: locked dropout probability
		:param distill_crf: CRF information distillation
		:param crf_attention: use CRF distillation weights
		:param biaf_attention: use bilinear attention for word-KD distillation
		"""

		super(SemanticDependencyParser, self).__init__()
		self.debug = False
		self.biaf_attention = False
		self.token_level_attention = False
		self.use_language_attention = False
		self.use_language_vector = False
		self.use_crf = use_crf
		self.use_decoder_timer = False
		self.sentence_level_loss = False
		self.train_initial_hidden_state = train_initial_hidden_state
		#add interpolation for target loss and distillation loss
		self.token_loss = token_loss

		self.interpolation = interpolation
		self.debug = debug
		self.use_rnn = use_rnn
		self.hidden_size = hidden_size

		self.rnn_layers: int = rnn_layers
		self.embeddings = embeddings
		self.config = config
		self.punct = punct 
		self.punct_list = ['``', "''", ':', ',', '.', 'PU', 'PUNCT']
		self.tree = tree
		self.is_mst = is_mst
		self.is_srl = is_srl
		# set the dictionaries
		self.tag_dictionary: Dictionary = tag_dictionary
		self.tag_type: str = tag_type
		self.tagset_size: int = len(tag_dictionary)

		self.word_map = word_map
		self.char_map = char_map
		
		# distillation part
		self.distill_arc = distill_arc
		self.distill_rel = distill_rel
		self.distill_crf = distill_crf
		self.distill_posterior = distill_posterior
		self.distill_prob = distill_prob
		self.distill_factorize = distill_factorize
		self.factorize_interpolation = factorize_interpolation
		self.temperature = temperature
		self.crf_attention = crf_attention
		self.diagonal = diagonal
		self.embedding_selector = embedding_selector

		# initialize the network architecture
		self.nlayers: int = rnn_layers
		self.hidden_word = None

		# dropouts
		self.use_dropout: float = dropout
		self.use_word_dropout: float = word_dropout
		self.use_locked_dropout: float = locked_dropout

		self.pickle_module = pickle_module

		if dropout > 0.0:
			self.dropout = torch.nn.Dropout(dropout)

		if word_dropout > 0.0:
			self.word_dropout = flair.nn.WordDropout(word_dropout)

		if locked_dropout > 0.0:
			self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

		rnn_input_dim: int = self.embeddings.embedding_length

		self.relearn_embeddings: bool = relearn_embeddings

		if self.embedding_selector:
			self.selector = Parameter(
						torch.randn(len(self.embeddings.embeddings),2),
						requires_grad=True,
					)

		if self.relearn_embeddings:
			self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

		self.bidirectional = True
		self.rnn_type = "LSTM"
		if not self.use_rnn:
			self.bidirectional = False
		# bidirectional LSTM on top of embedding layer
		num_directions = 1

		# hiddens
		self.n_mlp_arc = n_mlp_arc
		self.n_mlp_rel = n_mlp_rel
		self.mlp_dropout = mlp_dropout
		self.n_mlp_sec = n_mlp_sec
		self.init_std = init_std
		self.lstm_dropout = lstm_dropout
		self.factorize = factorize
		self.lexical_Dropout = nn.Dropout(p=word_dropout)
		if self.use_rnn:
			self.rnn = BiLSTM_1(input_size=rnn_input_dim,
							   hidden_size=hidden_size,
							   num_layers=self.nlayers,
							   dropout=self.lstm_dropout)

			mlp_input_hidden = hidden_size * 2
		else:
			mlp_input_hidden = rnn_input_dim

	
		self.mlp_rel_h = projection(emb_size=mlp_input_hidden, output_size=n_mlp_rel)
		self.mlp_rel_d = projection(emb_size=mlp_input_hidden, output_size=n_mlp_rel)

		self.rel_attn = bilinear_classifier(dropout=self.mlp_dropout, input_size_x=n_mlp_rel, 
											input_size_y=n_mlp_rel, output_size=self.tagset_size,
											bias_x=True, bias_y=True,
											)
		#-----------------------------------------------
		self.binary = binary
		# the Second Order Parts
		self.use_second_order=use_second_order
		self.iterations=iterations
		self.use_sib = use_sib
		self.use_cop = use_cop
		self.use_gp = use_gp
		if self.use_second_order:
			if use_sib:
				self.mlp_sib_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_sib_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.trilinear_sib = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
			if use_cop:
				self.mlp_cop_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_cop_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.trilinear_cop = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
			if use_gp:
				self.mlp_gp_h = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_gp_d = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.mlp_gp_hd = MLP(n_in=mlp_input_hidden,
								 n_hidden=n_mlp_sec,
								 dropout=mlp_dropout)
				self.trilinear_gp = TrilinearScorer(n_mlp_sec,n_mlp_sec,n_mlp_sec,init_std=init_std, rank = n_mlp_sec, factorize = factorize)
				

		self.rel_criterion = nn.CrossEntropyLoss()
		if self.factorize:
			self.arc_criterion = nn.CrossEntropyLoss()

		if self.binary:
			self.rel_criterion = nn.CrossEntropyLoss(reduction='none')
			if self.factorize:
				self.arc_criterion = nn.BCEWithLogitsLoss(reduction='none')
		if self.crf_attention:
			self.distill_criterion = nn.CrossEntropyLoss(reduction='none')
			self.distill_rel_criterion = nn.CrossEntropyLoss(reduction='none')
		self.to(flair.device)

		

	def _init_model_with_state_dict(state):
		use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
		use_word_dropout = (
			0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
		)
		use_locked_dropout = (
			0.0
			if not "use_locked_dropout" in state.keys()
			else state["use_locked_dropout"]
		)
		if 'biaf_attention' in state:
			biaf_attention = state['biaf_attention']
		else:
			biaf_attention = False
		if 'token_level_attention' in state:
			token_level_attention = state['token_level_attention']
		else:
			token_level_attention = False
		if 'teacher_hidden' in state:
			teacher_hidden = state['teacher_hidden']
		else:
			teacher_hidden = 256
		use_cnn=state["use_cnn"] if 'use_cnn' in state else False

		model = SemanticDependencyParser(
			hidden_size=state["hidden_size"],
			embeddings=state["embeddings"],
			tag_dictionary=state["tag_dictionary"],
			tag_type=state["tag_type"],
			use_crf=state["use_crf"],
			use_rnn=state["use_rnn"],
			train_initial_hidden_state=state["train_initial_hidden_state"],
			n_mlp_arc = state["n_mlp_arc"],
			n_mlp_rel = state["n_mlp_rel"],
			mlp_dropout = state["mlp_dropout"],
			token_loss = False if 'token_loss' not in state else state["token_loss"],
			use_second_order = state["use_second_order"],
			n_mlp_sec = state["n_mlp_sec"],
			init_std = state["init_std"],
			factorize = state["factorize"],
			use_sib = state["use_sib"],
			use_gp = state["use_gp"],
			use_cop = state["use_cop"],
			iterations = state["iterations"],
			is_mst = False if "is_mst" not in state else state["is_mst"],
			binary = state["binary"],
			rnn_layers=state["rnn_layers"],
			dropout=use_dropout,
			word_dropout=use_word_dropout,
			locked_dropout=use_locked_dropout,
			config=state['config'] if "config" in state else None,
			word_map=None if 'word_map' not in state else state['word_map'],
			char_map=None if 'char_map' not in state else state['char_map'],
			relearn_embeddings = True if 'relearn_embeddings' not in state else state['relearn_embeddings'],
			distill_arc = False if 'distill_arc' not in state else state['distill_arc'],
			distill_rel = False if 'distill_rel' not in state else state['distill_rel'],
			distill_crf = False if 'distill_crf' not in state else state['distill_crf'],
			distill_posterior = False if 'distill_posterior' not in state else state['distill_posterior'],
			distill_prob = False if 'distill_prob' not in state else state['distill_prob'],
			distill_factorize = False if 'distill_factorize' not in state else state['distill_factorize'],
			factorize_interpolation = False if 'factorize_interpolation' not in state else state['factorize_interpolation'],
			diagonal = False if 'diagonal' not in state else state['diagonal'],
			embedding_selector = False if "embedding_selector" not in state else state["embedding_selector"],
		)
		model.load_state_dict(state["state_dict"])
		return model

	def _get_state_dict(self):
		model_state = {
			"state_dict": self.state_dict(),
			"embeddings": self.embeddings,
			"hidden_size": self.hidden_size,
			"tag_dictionary":self.tag_dictionary,
			"tag_type":self.tag_type,
			"use_crf": self.use_crf,
			"use_rnn":self.use_rnn,
			"train_initial_hidden_state": self.train_initial_hidden_state,
			"n_mlp_arc": self.n_mlp_arc,
			"n_mlp_rel": self.n_mlp_rel,
			"mlp_dropout": self.mlp_dropout,
			"token_loss": self.token_loss,
			"use_second_order": self.use_second_order,
			"n_mlp_sec": self.n_mlp_sec,
			"init_std": self.init_std,
			"factorize": self.factorize,
			"use_sib": self.use_sib,
			"use_gp": self.use_gp,
			"use_cop": self.use_cop,
			"iterations": self.iterations,
			"is_mst": self.is_mst,
			"binary": self.binary,
			"rnn_layers": self.rnn_layers,
			"dropout": self.use_dropout,
			"word_dropout": self.use_word_dropout,
			"locked_dropout": self.use_locked_dropout,
			"config": self.config,
			"word_map": self.word_map,
			"char_map": self.char_map,
			"relearn_embeddings": self.relearn_embeddings,
			"distill_arc": self.distill_arc,
			"distill_rel": self.distill_rel,
			"distill_crf": self.distill_crf,
			"distill_posterior": self.distill_posterior,
			"distill_prob": self.distill_prob,
			"distill_factorize": self.distill_factorize,
			"factorize_interpolation": self.factorize_interpolation,
			"diagonal": self.diagonal,
			"embedding_selector": self.embedding_selector,
		}
		return model_state

	def forward(self, sentences: List[Sentence]):
		self.zero_grad()
		# pdb.set_trace()
		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
		self.lengths = lengths
		longest_token_sequence_in_batch: int = max(lengths)

		self.embeddings.embed(sentences)
		if self.embedding_selector:
			if self.training:
				selection=torch.nn.functional.gumbel_softmax(self.selector,hard=True)
				sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx][1] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
			else:
				selection=torch.argmax(self.selector,-1)
				sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx] for idx, x in enumerate(sorted(sentences.features.keys()))],-1)
		else:
			sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sentences.features],-1)
		if hasattr(self,'keep_embedding'):	
			sentence_tensor = [sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())]
			embedding_name = sorted(sentences.features.keys())[self.keep_embedding]
			if 'forward' in embedding_name or 'backward' in embedding_name:
				# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys()) if 'forward' in x or 'backward' in x],-1)
				for idx, x in enumerate(sorted(sentences.features.keys())):
					if 'forward' not in x and 'backward' not in x:
						sentence_tensor[idx].fill_(0)
			else:
				for idx, x in enumerate(sorted(sentences.features.keys())):
					if x != embedding_name:
						sentence_tensor[idx].fill_(0)
			sentence_tensor = torch.cat(sentence_tensor,-1)

		sentence_tensor = self.lexical_Dropout(sentence_tensor)
		
		


		# pdb.set_trace()
		if self.use_rnn:
			x = pack_padded_sequence(sentence_tensor, lengths, True, False)
			x, _ = self.rnn(x)
			sentence_tensor, _ = pad_packed_sequence(x, True, total_length=sentence_tensor.shape[1])

		# get the mask and lengths of given batch
		mask=self.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).cuda().type_as(sentence_tensor)	# shape: B x max_len, 0 for padding, 1 for token
		self.mask=mask
		# mask = words.ne(self.pad_index)
		# lens = mask.sum(dim=1)

		# get outputs from embedding layers
		x = sentence_tensor

		# apply MLPs to the BiLSTM output states
		rel_h = rel_d = arc_h = arc_d = x
		if self.factorize:
			arc_h = self.mlp_arc_h(arc_h)
			arc_d = self.mlp_arc_d(arc_d)
		rel_h = self.mlp_rel_h(rel_h)	#head of span
		rel_d = self.mlp_rel_d(rel_d)	#end of span

		# get arc and rel scores from the bilinear attention
		# [batch_size, seq_len, seq_len]
		if self.factorize:
			s_arc = self.arc_attn(arc_d, arc_h)
		else:
			s_arc = torch.tensor(0.)
			
		# [batch_size, seq_len, seq_len, n_rels]
		# s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
		#when use ner-dp bilinear module
		s_rel = self.rel_attn(rel_d, rel_h)

		# add second order using mean field variational inference
		if self.use_second_order:
			mask_unary, mask_sib, mask_cop, mask_gp = self.from_mask_to_3d_mask(mask)
			unary = mask_unary*s_arc
			arc_sib, arc_cop, arc_gp = self.encode_second_order(x)
			layer_sib, layer_cop, layer_gp = self.get_edge_second_order_node_scores(arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp)
			s_arc = self.mean_field_variational_infernece(unary, layer_sib, layer_cop, layer_gp) 
		# set the scores that exceed the length of each sentence to -inf
		if not self.binary and not self.factorize:
			s_arc.masked_fill_(~mask.unsqueeze(1).bool(), float(-1e9))	# s_arc.shape: B x max_len x max_len, mask.unsqueeze(1): B x 1 x max_len
		return s_arc, s_rel

	def mean_field_variational_infernece(self, unary, layer_sib=None, layer_cop=None, layer_gp=None):
		layer_gp2 = layer_gp.permute(0,2,3,1)
		# modify from (dep, head) to (head, dep), in order to fit my code
		unary = unary.transpose(1,2)
		unary_potential = unary.clone()
		q_value = unary_potential.clone()
		for i in range(self.iterations):
			if self.binary:
				q_value=torch.sigmoid(q_value)
			else:
				q_value=F.softmax(q_value,1)
			if self.use_sib:
				second_temp_sib = torch.einsum('nac,nabc->nab', (q_value, layer_sib))
				#(n x ma x mb) -> (n x ma) -> (n x ma x 1) | (n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,a)*p(a,b,a) 
				diag_sib1 = torch.diagonal(q_value,dim1=1,dim2=2).unsqueeze(-1) * torch.diagonal(layer_sib.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				# (n x ma x mb x mc) -> (n x ma x mb)
				#Q(a,b)*p(a,b,b)
				diag_sib2 = q_value * torch.diagonal(layer_sib,dim1=-2,dim2=-1)
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				second_temp_sib = second_temp_sib - diag_sib1 - diag_sib2
			else:
				second_temp_sib=0

			if self.use_gp:
				second_temp_gp = torch.einsum('nbc,nabc->nab', (q_value, layer_gp))
				second_temp_gp2 = torch.einsum('nca,nabc->nab', (q_value, layer_gp2))
				#Q(b,a)*p(a,b,a)
				diag_gp1 = q_value.transpose(1,2) * torch.diagonal(layer_gp.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,b)*p(a,b,b)
				diag_gp2 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(1) * torch.diagonal(layer_gp,dim1=-2,dim2=-1)
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,a)*p(a,b,a)
				diag_gp21 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(-1) * torch.diagonal(layer_gp2.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,a)*p(a,b,b)
				diag_gp22 = q_value.transpose(1,2) * torch.diagonal(layer_gp2,dim1=-2,dim2=-1)

				second_temp_gp = second_temp_gp - diag_gp1 - diag_gp2
				#c->a->b
				second_temp_gp2 = second_temp_gp2 - diag_gp21 - diag_gp22
			else:
				second_temp_gp=second_temp_gp2=0

			if self.use_cop:
				second_temp_cop = torch.einsum('ncb,nabc->nab', (q_value, layer_cop))
				#(n x ma x mb x mc) -> (n x mb x ma x mc) -> (n x mb x ma) -> (n x ma x mb)
				#Q(a,b)*p(a,b,a)
				diag_cop1 = q_value * torch.diagonal(layer_cop.transpose(1,2),dim1=-2,dim2=-1).transpose(1,2)
				# diag_cop1 = q_value * tf.transpose(tf.linalg.diag_part(tf.transpose(layer_cop,perm=[0,2,1,3])),perm=[0,2,1])
				#(n x ma x mb) -> (n x mb) -> (n x 1 x mb) | (n x ma x mb x mc) -> (n x ma x mb)
				#Q(b,b)*p(a,b,b)
				diag_cop2 = torch.diagonal(q_value,dim1=-2,dim2=-1).unsqueeze(1) * torch.diagonal(layer_cop,dim1=-2,dim2=-1)
				# diag_cop2 = tf.expand_dims(tf.linalg.diag_part(q_value),1) * tf.linalg.diag_part(layer_cop)
				second_temp_cop = second_temp_cop - diag_cop1 - diag_cop2
			else:
				second_temp_cop=0

			second_temp = second_temp_sib + second_temp_gp + second_temp_gp2 + second_temp_cop
			q_value = unary_potential + second_temp
		# transpose from (head, dep) to (dep, head)
		return q_value.transpose(1,2)

	def encode_second_order(self, memory_bank):

		if self.use_sib:
			edge_node_sib_h = self.mlp_sib_h(memory_bank)
			edge_node_sib_m = self.mlp_sib_d(memory_bank)
			arc_sib=(edge_node_sib_h, edge_node_sib_m)
		else:
			arc_sib=None

		if self.use_cop:
			edge_node_cop_h = self.mlp_cop_h(memory_bank)
			edge_node_cop_m = self.mlp_cop_d(memory_bank)
			arc_cop=(edge_node_cop_h, edge_node_cop_m)
		else:
			arc_cop=None

		if self.use_gp:
			edge_node_gp_h = self.mlp_gp_h(memory_bank)
			edge_node_gp_m = self.mlp_gp_d(memory_bank)
			edge_node_gp_hm = self.mlp_gp_hd(memory_bank)
			arc_gp=(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m)
		else:
			arc_gp=None

		return arc_sib, arc_cop, arc_gp

	def get_edge_second_order_node_scores(self, arc_sib, arc_cop, arc_gp, mask_sib, mask_cop, mask_gp):

		if self.use_sib:
			edge_node_sib_h, edge_node_sib_m = arc_sib
			layer_sib = self.trilinear_sib(edge_node_sib_h, edge_node_sib_m, edge_node_sib_m) * mask_sib
			# keep (ma x mb x mc) -> (ma x mb x mb)
			#layer_sib = 0.5 * (layer_sib + layer_sib.transpose(3,2))
			one_mask=torch.ones(layer_sib.shape[-2:]).cuda()
			tril_mask=torch.tril(one_mask,-1)
			triu_mask=torch.triu(one_mask,1)
			layer_sib = layer_sib-layer_sib*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_sib*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
			
		else:
			layer_sib = None
		if self.use_cop:
			edge_node_cop_h, edge_node_cop_m = arc_cop
			layer_cop = self.trilinear_cop(edge_node_cop_h, edge_node_cop_m, edge_node_cop_h) * mask_cop
			# keep (ma x mb x mc) -> (ma x mb x ma)
			one_mask=torch.ones(layer_cop.shape[-2:]).cuda()
			tril_mask=torch.tril(one_mask,-1)
			triu_mask=torch.triu(one_mask,1)
			layer_cop=layer_cop.transpose(1,2)
			layer_cop = layer_cop-layer_cop*tril_mask.unsqueeze(0).unsqueeze(0) + (layer_cop*triu_mask.unsqueeze(0).unsqueeze(0)).permute([0,1,3,2])
			layer_cop=layer_cop.transpose(1,2)
		else:
			layer_cop = None

		if self.use_gp:
			edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m = arc_gp
			layer_gp = self.trilinear_gp(edge_node_gp_h, edge_node_gp_hm, edge_node_gp_m) * mask_gp
		else:
			layer_gp = None
		
		return layer_sib,layer_cop,layer_gp

	def from_mask_to_3d_mask(self,token_weights):
		root_weights = token_weights.clone()
		root_weights[:,0] = 0
		token_weights3D = token_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
		token_weights2D = root_weights.unsqueeze(-1) * root_weights.unsqueeze(-2)
		# abc -> ab,ac
		#token_weights_sib = tf.cast(tf.expand_dims(root_, axis=-3) * tf.expand_dims(tf.expand_dims(root_weights, axis=-1),axis=-1),dtype=tf.float32)
		#abc -> ab,cb
		if self.use_cop:
			token_weights_cop = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * token_weights.unsqueeze(1).unsqueeze(1)
			token_weights_cop[:,0,:,0] = 0
		else:
			token_weights_cop=None
		#data=np.stack((devprint['printdata']['layer_cop'][0][0]*devprint['token_weights3D'][0].T)[None,:],devprint['printdata']['layer_cop'][0][1:])
		#abc -> ab, bc
		if self.use_gp:
			token_weights_gp = token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
		else:
			token_weights_gp = None

		if self.use_sib:
			#abc -> ca, ab
			if self.use_gp:
				token_weights_sib = token_weights_gp.clone()
			else:
				token_weights.unsqueeze(-1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(-1) * root_weights.unsqueeze(1).unsqueeze(1)
		else:
			token_weights_sib = None
		return token_weights3D, token_weights_sib, token_weights_cop, token_weights_gp



	def forward_loss(
		self, data_points: Union[List[Sentence], Sentence], sort=True
		) -> torch.tensor:
		s_arc, s_rel = self.forward(data_points)

		loss = self._calculate_loss_dp(s_arc, s_rel, data_points, self.mask)
		return loss

	def simple_forward_distillation_loss(
		self, data_points: Union[List[Sentence], Sentence], teacher_data_points: Union[List[Sentence], Sentence]=None, teacher=None, sort=True,
		interpolation=0.5, train_with_professor=False, professor_interpolation=0.5, language_attention_warmup = False, calc_teachers_target_loss = False,
		language_weight = None, biaffine = None, language_vector = None,
		) -> torch.tensor:
		arc_scores, rel_scores = self.forward(data_points)
		lengths = [len(sentence.tokens) for sentence in data_points]
		max_len = arc_scores.shape[1]
		mask=self.mask.clone()
		posterior_loss = 0
		if self.distill_posterior:
			# mask[:,0] = 0
			if hasattr(data_points,'teacher_features') and 'posteriors' in data_points.teacher_features:
				teacher_scores = data_points.teacher_features['posteriors'].to(flair.device)
			else:
				teacher_scores = torch.stack([sentence.get_teacher_posteriors() for sentence in data_points],0)
			if self.distill_arc:
				root_mask = mask.clone()
				root_mask[:,0] = 0
				binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
				arc_scores.masked_fill_(~binary_mask.bool(), float(-1e9))
				for i in range(teacher_scores.shape[-2]):
					if self.distill_rel:
						assert 0
						marginals = convert_score_back(teacher_scores[:,:,:,i])
						arc_probs = arc_scores.softmax(-1)
						rel_probs = rel_scores.softmax(-1)
						student_probs = arc_probs.unsqueeze(-1) * rel_probs
						student_scores = (student_probs+1e-12).log()
						student_scores = student_scores.view(list(student_scores.shape[0:2])+[-1])
						marginals = marginals.reshape(list(marginals.shape[0:2])+[-1])
						# create the mask
						binary_mask = binary_mask.unsqueeze(-1).expand(list(binary_mask.shape)+[rel_probs.shape[-1]]).reshape(list(binary_mask.shape[0:2])+[-1])
					else:
						marginals = convert_score_back(teacher_scores[:,:,i])
					posterior_loss += self._calculate_distillation_loss(student_scores, marginals, root_mask, binary_mask, T=self.temperature, teacher_is_score = False)
			else:
				root_mask = mask.clone()
				root_mask[:,0] = 0
				binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
				inside_outside_prob = crf(arc_scores, root_mask.bool(),marginal_gradient=True)
				inside_outside_score = (inside_outside_prob + 1e-12).log()
				for i in range(teacher_scores.shape[-2]):
					posterior_loss += self._calculate_distillation_loss(inside_outside_score, teacher_scores[:,:,i], root_mask, binary_mask, T=self.temperature, teacher_is_score = False)
				# temp_mask = mask[:,1:]
				# dist=generate_tree(arc_scores,temp_mask.squeeze(-1).long(),is_mst=self.is_mst)
				# forward_backward_score = dist.marginals
				# # change back to relation of (dependency, head)
				# input_forward_score = (forward_backward_score.transpose(-1,-2)+1e-12).log()
				# binary_mask = temp_mask.unsqueeze(-1) * temp_mask.unsqueeze(-2)
				# input_forward_score.masked_fill_(~binary_mask.bool(), float(-1e9))
				# for i in range(teacher_scores.shape[-2]):
				# 	posterior_loss += self._calculate_distillation_loss(input_forward_score, teacher_scores[:,:,i].transpose(-1,-2), temp_mask, binary_mask, T=self.temperature, teacher_is_score = False)
			posterior_loss/=teacher_scores.shape[-2]
		
		distillation_loss = 0
		if self.distill_crf:
			# [batch, length, kbest]
			mask[:,0] = 0
			if hasattr(data_points,'teacher_features') and 'topk' in data_points.teacher_features:
				teacher_tags = data_points.teacher_features['topk'].to(flair.device)
				teacher_weights = data_points.teacher_features['weights'].to(flair.device)
				if self.distill_rel:
					teacher_rel_tags = data_points.teacher_features['topk_rels'].to(flair.device)
			else:
				teacher_tags = torch.stack([sentence.get_teacher_target() for sentence in data_points],0)
				teacher_weights = torch.stack([sentence.get_teacher_weights() for sentence in data_points],0)
				if self.distill_rel:
					teacher_rel_tags = torch.stack([sentence.get_teacher_rel_target() for sentence in data_points],0)
			# proprocess, convert k best to batch wise
			teacher_mask = (mask.unsqueeze(-1) * (teacher_weights.unsqueeze(1)>0).type_as(mask)).bool()
			
			student_arc_scores = arc_scores.unsqueeze(-2).expand(list(arc_scores.shape[:2])+[teacher_mask.shape[-1],arc_scores.shape[-1]])[teacher_mask]
			teacher_topk_arcs = teacher_tags[teacher_mask]
			if self.distill_rel:
				# gold_arcs = arcs[mask]
				# rel_scores, rels = rel_scores[mask], rels[mask]
				# rel_scores = rel_scores[torch.arange(len(gold_arcs)), gold_arcs]

				student_rel_scores = rel_scores.unsqueeze(-3).expand(list(rel_scores.shape[:2])+[teacher_mask.shape[-1]]+list(rel_scores.shape[-2:]))[teacher_mask]
				teacher_topk_rels = teacher_rel_tags[teacher_mask]
				student_rel_scores = student_rel_scores[torch.arange(len(teacher_topk_arcs)),teacher_topk_arcs]
			if self.crf_attention:
				weights = teacher_weights.unsqueeze(1).expand([teacher_weights.shape[0],arc_scores.shape[1],teacher_weights.shape[1]])[teacher_mask]
				distillation_loss = self.distill_criterion(student_arc_scores, teacher_topk_arcs)
				# the loss calculates only one times because the sum of weight is 1
				distillation_loss = (distillation_loss * weights).sum() / mask.sum()
				if self.distill_rel:
					rel_distillation_loss = self.distill_rel_criterion(student_rel_scores, teacher_topk_rels)
					rel_distillation_loss = (rel_distillation_loss * weights).sum() / mask.sum()
			else:
				# the loss calculates for k times
				distillation_loss = self.arc_criterion(student_arc_scores, teacher_topk_arcs)
				if self.distill_rel:
					rel_distillation_loss = self.rel_criterion(student_rel_scores, teacher_topk_rels)

		arc_loss,rel_loss = self._calculate_loss(arc_scores, rel_scores, data_points, self.mask.clone(), return_arc_rel=True)
		if (self.distill_arc or self.distill_rel) and not self.distill_posterior and not self.distill_crf:
			root_mask = mask.clone()
			root_mask[:,0] = 0
			binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)

			if hasattr(data_points,'teacher_features') and 'distributions' in data_points.teacher_features:
				teacher_features = data_points.teacher_features['distributions'].to(flair.device)
			else:
				teacher_features = torch.stack([sentence.get_teacher_prediction() for sentence in data_points],0)

			if self.distill_arc:
				features = arc_scores
			if self.distill_rel:
				# features = arc_scores.unsqueeze(-1) * rel_scores
				if self.distill_factorize:
					rel_binary_mask = binary_mask.unsqueeze(-1).expand(list(binary_mask.shape)+[rel_scores.shape[-1]]).reshape(list(binary_mask.shape[0:2])+[-1])
					if hasattr(data_points,'teacher_features') and 'rel_distributions' in data_points.teacher_features:
						teacher_rel_features = data_points.teacher_features['rel_distributions'].to(flair.device)
					else:
						teacher_rel_features = torch.stack([sentence.get_teacher_rel_prediction() for sentence in data_points],0)
					rel_probs = rel_scores.softmax(-1)
					
					rel_probs = rel_probs.view(list(rel_probs.shape[0:2])+[-1])
					rel_scores = (rel_probs+1e-12).log()

					teacher_rel_features = teacher_rel_features.view(list(teacher_rel_features.shape[0:2])+[-1])

					rel_distillation_loss = self._calculate_distillation_loss(rel_scores, teacher_rel_features, root_mask, rel_binary_mask, T=self.temperature, teacher_is_score=(not self.distill_prob) and (not self.distill_rel))
					features = arc_scores
				else:
					arc_probs = arc_scores.softmax(-1)
					rel_probs = rel_scores.softmax(-1)
					features = arc_probs.unsqueeze(-1) * rel_probs
					features = features.view(list(features.shape[0:2])+[-1])
					features = (features+1e-12).log()
					teacher_features = teacher_features.view(list(teacher_features.shape[0:2])+[-1])
					# create the mask
					binary_mask = binary_mask.unsqueeze(-1).expand(list(binary_mask.shape)+[rel_probs.shape[-1]]).reshape(list(binary_mask.shape[0:2])+[-1])

			else:
				teacher_features.masked_fill_(~self.mask.unsqueeze(1).bool(), float(-1e9))

			distillation_loss = self._calculate_distillation_loss(features, teacher_features, root_mask, binary_mask, T=self.temperature, teacher_is_score=(not self.distill_prob) and (not self.distill_rel))
		# target_loss2 = super()._calculate_loss(features,data_points)
		# distillation_loss2 = super()._calculate_distillation_loss(features, teacher_features,torch.tensor(lengths))
		# (interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * target_loss).backward()
		if self.distill_rel:
			# if distilling both arc and rel distribution, just use the same interpolation
			target_loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)
			if self.distill_factorize:
				# balance the relation distillation loss and arc distillation loss through a new interpolation
				distillation_loss = 2 * ((1-self.factorize_interpolation) * distillation_loss + self.factorize_interpolation * rel_distillation_loss)
			if self.distill_crf:
				distillation_loss = 2 * ((1-self.interpolation) * distillation_loss + self.interpolation * rel_distillation_loss)
			return interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * target_loss
		else:
			# otherwise, balance between the (arc distillation loss + arc loss) and (rel loss)
			return 2*((1-self.interpolation) * (interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * arc_loss) + self.interpolation * rel_loss)

	def sequence_mask(self, lengths, max_len=None):
		"""
		Creates a boolean mask from sequence lengths.
		"""
		
		batch_size = lengths.numel()
		max_len = max_len or lengths.max()
		return (torch.arange(0, max_len)
				.type_as(lengths)
				.repeat(batch_size, 1)
				.lt(lengths.unsqueeze(1)))
	def _calculate_distillation_loss(self, features, teacher_features, mask, binary_mask, T = 1, teacher_is_score=True, student_is_score = True):
		# TODO: time with mask, and whether this should do softmax
		# pdb.set_trace()
		if teacher_is_score:
			teacher_prob=F.softmax(teacher_features/T, dim=-1)
		else:
			if T>1:
				teacher_scores = (teacher_features+1e-12).log()
				teacher_prob=F.softmax(teacher_scores/T, dim=-1)
			else:
				teacher_prob=teacher_features
		KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), teacher_prob,reduction='none') * binary_mask * T * T

		# KD_loss = KD_loss.sum()/mask.sum()
		
		if self.sentence_level_loss:
			KD_loss = KD_loss.sum()/KD_loss.shape[0]
		else:
			KD_loss = KD_loss.sum()/mask.sum()
		return KD_loss
		# return torch.nn.functional.MSELoss(features, teacher_features, reduction='mean')
	def _calculate_loss(
		self, arc_scores: torch.tensor, rel_scores: torch.tensor, sentences: List[Sentence], mask: torch.tensor, return_arc_rel = False,
		) -> float:
		
		if self.binary:
			root_mask = mask.clone()

			binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
			# triangular binary mask
			binary_mask = torch.tril(binary_mask, diagonal=0)
			if self.factorize:
				if hasattr(sentences, self.tag_type+'_arc_tags'):
					arc_mat=getattr(sentences,self.tag_type+'_arc_tags').to(flair.device).float()
				else:
					arc_mat=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in sentences],0).float()
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rel_mat=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()	# B x max_len x max_len
			else:
				rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()

			if self.factorize:
				arc_loss = self.arc_criterion(arc_scores, arc_mat)
				arc_loss = (arc_loss*binary_mask).sum()/binary_mask.sum()
			else:
				arc_loss = 0
			rel_loss = self.rel_criterion(rel_scores.reshape(-1,self.tagset_size), rel_mat.reshape(-1))	# rel_scores shape (1,9,9,5), rel_mat shape(1,9,9), rel_loss shape (81,)
			# tag index 0 for 'None'
			# rel_mask = (rel_mat>0)*binary_mask
			rel_mask = binary_mask
			num_rels=rel_mask.sum()
			if num_rels>0:
				# rel_loss = (rel_loss*rel_mask.view(-1)).sum()/num_rels
				rel_loss = (rel_loss*rel_mask.view(-1)).sum()

			else:
				rel_loss = 0
			# rel_loss = (rel_loss*rel_mat.view(-1)).sum()/rel_mat.sum()
			#--------------ner_dp--------------
			if self.factorize:
				loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)
			else:
				loss = rel_loss
				

		else:
			if hasattr(sentences,self.tag_type+'_arc_tags'):
				arcs=getattr(sentences,self.tag_type+'_arc_tags').to(flair.device).long()
			else:
				arcs=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in sentences],0).long()
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rels=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()
			else:
				rels=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()
			self.arcs=arcs
			self.rels=rels
			mask[:,0] = 0
			mask = mask.bool()
			gold_arcs = arcs[mask]
			rel_scores, rels = rel_scores[mask], rels[mask]
			rel_scores = rel_scores[torch.arange(len(gold_arcs)), gold_arcs]
			if self.use_crf:
				arc_loss, arc_probs = crf(arc_scores, mask, arcs)
				arc_loss = arc_loss/mask.sum()
				rel_loss = self.rel_criterion(rel_scores, rels)

			else:
				arc_scores, arcs = arc_scores[mask], arcs[mask]
				arc_loss = self.arc_criterion(arc_scores, arcs)
			
				
				rel_loss = self.rel_criterion(rel_scores, rels)
			loss = 2 * ((1-self.interpolation) * arc_loss + self.interpolation * rel_loss)

		if return_arc_rel:
			return (arc_loss,rel_loss)
	
		return loss

	def _calculate_loss_dp(self, s_arc, s_rel, sentences, mask=None):
		if self.binary:
			if hasattr(sentences,self.tag_type+'_rel_tags'):
				rel_mat=getattr(sentences,self.tag_type+'_rel_tags').to(flair.device).long()	# B x max_len x max_len
			else:
				rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in sentences],0).long()
			
			binary_mask = self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)
			binary_mask_l = torch.tril(binary_mask, diagonal=0).to(bool)
			flattened_candidate_scores_mask = binary_mask_l.view(-1)
			gold_labels = torch.masked_select(rel_mat, binary_mask_l)
			candidate_ner_scores = s_rel.reshape(-1, self.tagset_size)[flattened_candidate_scores_mask==True]
			loss = self.rel_criterion(input=candidate_ner_scores, target=gold_labels).sum()
			# pdb.set_trace()
		return loss

	def evaluate(
		self,
		data_loader: DataLoader,
		out_path: Path = None,
		embeddings_storage_mode: str = "cpu",
		prediction_mode: bool = False,
		) -> (Result, float):
		data_loader.assign_embeddings()
		with torch.no_grad():
			if self.binary:
				eval_loss = 0

				batch_no: int = 0

				# metric = Metric("Evaluation")
				# sentence_writer = open('temps/'+str(uid)+'_eval'+'.conllu','w')
				lines: List[str] = []
				utp = 0
				ufp = 0
				ufn = 0
				ltp = 0
				lfp = 0
				lfn = 0

				for batch in data_loader:
					pred_spans = set()	# predicted spans after decoding.
					gold_spans = set()

					batch_no += 1
					
					arc_scores, rel_scores = self.forward(batch)

					mask=self.mask
					root_mask = mask.clone()
					# root_mask[:,0] = 0
					# B * max_len * max_len
					binary_mask = root_mask.unsqueeze(-1) * mask.unsqueeze(-2)
					binary_mask = torch.tril(binary_mask, diagonal=0)
					if self.factorize:
						arc_predictions = (arc_scores.sigmoid() > 0.5) * binary_mask
					# rel_predictions = (rel_scores.softmax(-1)*binary_mask.unsqueeze(-1)).argmax(-1)
					rel_predictions = rel_scores.argmax(-1)*binary_mask
					if not prediction_mode:
						
						arc_mat=torch.stack([getattr(sentence,self.tag_type+'_arc_tags').to(flair.device) for sentence in batch],0).float()
						rel_mat=torch.stack([getattr(sentence,self.tag_type+'_rel_tags').to(flair.device) for sentence in batch],0).long()
						loss = self._calculate_loss(arc_scores, rel_scores, batch, mask)
						if self.factorize:
							# UF1
							true_positives = arc_predictions * arc_mat
							# (n x m x m) -> ()
							n_predictions = arc_predictions.sum()
							n_unlabeled_predictions = n_predictions
							n_targets = arc_mat.sum()
							n_unlabeled_targets = n_targets
							n_true_positives = true_positives.sum()
							# () - () -> ()
							n_false_positives = n_predictions - n_true_positives
							n_false_negatives = n_targets - n_true_positives
							# (n x m x m) -> (n)
							n_targets_per_sequence = arc_mat.sum([1,2])
							n_true_positives_per_sequence = true_positives.sum([1,2])
							# (n) x 2 -> ()
							n_correct_sequences = (n_true_positives_per_sequence==n_targets_per_sequence).sum()
							utp += n_true_positives
							ufp += n_false_positives
							ufn += n_false_negatives

							# LF1
							# (n x m x m) (*) (n x m x m) -> (n x m x m)
							true_positives = (rel_predictions == rel_mat) * arc_predictions
							correct_label_tokens = (rel_predictions == rel_mat) * arc_mat
							# (n x m x m) -> ()
							# n_unlabeled_predictions = tf.reduce_sum(unlabeled_predictions)
							# n_unlabeled_targets = tf.reduce_sum(unlabeled_targets)
							n_true_positives = true_positives.sum()
							n_correct_label_tokens = correct_label_tokens.sum()
							# () - () -> ()
							n_false_positives = n_unlabeled_predictions - n_true_positives
							n_false_negatives = n_unlabeled_targets - n_true_positives
							# (n x m x m) -> (n)
							n_targets_per_sequence = arc_mat.sum([1,2])
							n_true_positives_per_sequence = true_positives.sum([1,2])
							n_correct_label_tokens_per_sequence = correct_label_tokens.sum([1,2])
							# (n) x 2 -> ()
							n_correct_sequences = (n_true_positives_per_sequence == n_targets_per_sequence).sum()
							n_correct_label_sequences = ((n_correct_label_tokens_per_sequence == n_targets_per_sequence)).sum()
							ltp += n_true_positives
							lfp += n_false_positives
							lfn += n_false_negatives

							eval_loss += loss
							eval_loss /= batch_no


							UF1=self._compute_F1(utp,ufp,ufn)
							LF1=self._compute_F1(ltp,lfp,lfn)
						else:	#unfactorized

							#=========================================================
							if self.tag_type == 'ner_dp':
								# pdb.set_trace()
								rel_preds = self.decode_ner_dp(arc_scores, rel_scores, self.mask)	# spans list (s, e, label_index)
								rel_gold_nonzeros_batch = rel_mat.permute(0,2,1).nonzero()	# batch_size * n_nonzero * 2(coordination), entry: label coordination in rel_mat(end, start)
								for coordination in rel_gold_nonzeros_batch:
										sent_idx, s, e = int(coordination[0]), int(coordination[1]), int(coordination[2])
										label_idx = int(rel_mat[sent_idx, e, s])
										gold_spans.add((sent_idx, s, e, label_idx))

								for sent_idx, sentence in enumerate(batch):
									#gold
									#pred					
									spans = rel_preds[sent_idx]
									for span in spans:
										if len(span) >0:
											s, e, t = span
											pred_spans.add((sent_idx, s, e, t))

							ltp += len(gold_spans & pred_spans)
							lfp += len(gold_spans - pred_spans)
							lfn += len(pred_spans - gold_spans)
							UF1=0
							LF1=self._compute_F1(ltp,lfp,lfn)
							eval_loss += loss
							eval_loss /= batch_no

					if out_path is not None:
						# pdb.set_trace()
						# if self.target
						# lengths = [len(sentence.tokens) for sentence in batch]
						
						# temp_preds = eisner(arc_scores, mask)
						# -----------------for ner_dp-----------------
						if self.tag_type == 'ner_dp':
							# pdb.set_trace()
							rel_preds = self.decode_ner_dp(arc_scores, rel_scores, self.mask)
							for sent_idx, sentence in enumerate(batch):
								pred_spans = rel_preds[sent_idx]	#list of spans, span formation: (s, e, label_idx)

								labels = ['O' for _ in sentence]
								# pdb.set_trace()
								for span in pred_spans:
									s, e, t = span
									label = str(self.tag_dictionary.idx2item[t].decode('utf-8'))
									labels[s] = 'B-' + label
									if s<e:
										for i in range(s+1, e+1):
											labels[i] = 'I-' + label
								
								for token_idx, token in enumerate(sentence):
									
									eval_line = "{}\t{}\t{}\t{}\n".format(
										token.text,
										token.get_tag('pos').value,
										token.get_tag('chunk').value,
										labels[token_idx]
										)
									lines.append(eval_line)
								lines.append("\n")								
							
						
						else:
							if not self.is_mst:
								temp_preds = eisner(arc_scores, root_mask.bool())
							
							for (sent_idx, sentence) in enumerate(batch):
								if self.is_mst:
									preds=MST_inference(torch.softmax(masked_arc_scores[sent_idx],-1).cpu().numpy(), len(sentence), binary_mask[sent_idx].cpu().numpy())
								else:
									preds=temp_preds[sent_idx]

								for token_idx, token in enumerate(sentence):
									# pdb.set_trace()
									if token_idx == 0:
										continue
									

									# append both to file for evaluation
									arc_heads = torch.where(arc_predictions[sent_idx,token_idx]>0)[0]
									if preds[token_idx] not in arc_heads:
										val=torch.zeros(1).type_as(arc_heads)
										val[0]=preds[token_idx].item()
										arc_heads=torch.cat([arc_heads,val],0)
									if len(arc_heads) == 0:
										arc_heads = masked_arc_scores[sent_idx,token_idx].argmax().unsqueeze(0)
									rel_index = rel_predictions[sent_idx,token_idx,arc_heads]
									rel_labels = [self.tag_dictionary.get_item_for_index(x) for x in rel_index]
									arc_list=[]
									for i, label in enumerate(rel_labels):
										if '+' in label:
											labels = label.split('+')
											for temp_label in labels:
												arc_list.append(str(arc_heads[i].item())+':'+temp_label)
										else:
											arc_list.append(str(arc_heads[i].item())+':'+label)
									eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
										token_idx,
										token.text,
										'X',
										'X',
										'X',
										'X=X',
										str(token_idx-1),
										'root' if token_idx-1==0 else 'det',
										'|'.join(arc_list),
										'X',
									)
									lines.append(eval_line)
								lines.append("\n")

				if out_path is not None:
					with open(out_path, "w", encoding="utf-8") as outfile:
						outfile.write("".join(lines))
				if prediction_mode:
					return None, None

				result = Result(
					main_score=LF1,
					log_line=f"\nUF1: {UF1} - LF1 {LF1}",
					log_header="PRECISION\tRECALL\tF1",
					detailed_results=f"\nUF1: {UF1} - LF1 {LF1}",
				)
			else:
				if prediction_mode:
					eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path,prediction_mode=prediction_mode)
					return eval_loss, metric
				else:   
					eval_loss, metric=self.dependency_evaluate(data_loader,out_path=out_path)
				
				UAS=metric.uas
				LAS=metric.las
				result = Result(main_score=LAS,log_line=f"\nUAS: {UAS} - LAS {LAS}",log_header="PRECISION\tRECALL\tF1",detailed_results=f"\nUAS: {UAS} - LAS {LAS}",)
			return result, eval_loss
	
	def _compute_F1(self, tp, fp, fn):
		precision = tp/(tp+fp + 1e-12)
		recall = tp/(tp+fn + 1e-12)
		return 2 * (precision * recall) / (precision + recall+ 1e-12)


	@torch.no_grad()
	def dependency_evaluate(self, loader, out_path=None, prediction_mode=False):
		# self.model.eval()

		loss, metric = 0, Metric()
		# total_start_time=time.time()
		# forward_time=0
		# loss_time=0
		# decode_time=0
		# punct_time=0
		lines=[]
		for batch in loader:
			forward_start=time.time()
			arc_scores, rel_scores = self.forward(batch)
			# forward_end=time.time()
			mask = self.mask
			if not prediction_mode:
				loss += self._calculate_loss(arc_scores, rel_scores, batch, mask)
			# loss_end=time.time()
			# forward_time+=forward_end-forward_start
			# loss_time+=loss_end-forward_end
			mask=mask.bool()
			# decode_start=time.time()
			arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
			# decode_end=time.time()
			# decode_time+=decode_end-decode_start
			# ignore all punctuation if not specified

			if not self.punct and out_path is None :
				for sent_id,sentence in enumerate(batch):
					for token_id, token in enumerate(sentence):
						upos=token.get_tag('upos').value
						xpos=token.get_tag('pos').value
						word=token.text
						if is_punctuation(word,upos,self.punct_list) or is_punctuation(word,upos,self.punct_list):
							mask[sent_id][token_id]=0
				# mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
			if out_path is not None:
				for (sent_idx, sentence) in enumerate(batch):
					for token_idx, token in enumerate(sentence):
						if token_idx == 0:
							continue

						# append both to file for evaluation
						eval_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
							token_idx,
							token.text,
							'X',
							'X',
							'X',
							'X',
							arc_preds[sent_idx,token_idx],
							self.tag_dictionary.get_item_for_index(rel_preds[sent_idx,token_idx]),
							'X',
							'X',
						)
						lines.append(eval_line)
					lines.append("\n")
				
			
			if not prediction_mode:
				# punct_end=time.time()
				# punct_time+=punct_end-decode_end
				metric(arc_preds, rel_preds, self.arcs, self.rels, mask)
		if out_path is not None:
			with open(out_path, "w", encoding="utf-8") as outfile:
				outfile.write("".join(lines))
		if prediction_mode:
			return None, None

		
		loss /= len(loader)

		return loss, metric




	
	@torch.no_grad()
	def decode_ner_dp(self, arc_scores, rel_scores, mask, is_flat_ner=True):
		"""
		input: arc_scores, rel_scores, ner type(flat/nested)
		output: arc_preds, rel_preds (matrix representation)
		rel_scores, shape1: dependent; shape2: head
		"""
		if len(arc_scores.shape)>0:
			arc_scores = arc_scores.permute(0,2,1)
		rel_scores = rel_scores.permute(0,2,1,3)

		candidates = []

		# else:	
		lengths = mask.sum(-1).tolist()		# length list
		if self.factorize:
			arc_preds = (arc_scores.sigmoid() > 0.5).float()
		rel_preds = rel_scores.argmax(-1)	# B * max_len * max_len, entry: ner label index.
		# print(rel_preds)
		n_sentences = int(rel_scores.shape[0])
		
		for sid, len_sent in enumerate(lengths):
			for s in range(int(len_sent)):
				for e in range(s,int(len_sent)):
					candidates.append((sid,s,e))
		
		top_spans = [[] for _ in range(n_sentences)]
		for sid, s, e in candidates:
			# sid, s, e = candidates[sid]
			if self.factorize:
				label_idx = int(rel_preds[sid, s, e]*arc_preds[sid, s, e])
			else:
				label_idx = int(rel_preds[sid, s, e])
			if label_idx > 0:	# 0 is None 
				top_spans[sid].append((s, e, label_idx, rel_scores[sid, s, e, label_idx].item()))

		top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans] # 对于每个句子中的所有spans，按分数降序排序
		sent_pred_mentions = [[] for _ in range(n_sentences)]
		# pdb.set_trace()
		for sid, spans in enumerate(top_spans):
			for ns,ne,t,_ in spans:
				for ts,te,_ in sent_pred_mentions[sid]:   # ts, te 是之前记录的span start/end position，ns, ne必须与所有的 ts,te相容。
					if ns < ts <= ne < te or ts < ns <= te < ne:      # clash 发生，break, ns, ne不相容，看下一span.
						#for both nested and flat ner no clash is allowed
						break
					if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
						#for flat ner nested mentions are not allowed
						break
				else:
					sent_pred_mentions[sid].append((ns,ne,t))

		# pred_mentions = set((sid,s,e,t) for sid, spr in enumerate(sent_pred_mentions) for s,e,t in spr)	# (sent_id, s, e, label_idx)
		return sent_pred_mentions
		

	def distill_dp_transform(self, batch, logits, student_tag_dictionary, temperature):
		"""
		logits， n_batch * max_seq_len * max_seq_len * n_role
		"""
		T = temperature
		# logits = (logits/T)	# permute the logits, so shape1: head, shape2: dependent
		logits = logits.permute(0,2,1,3)
		# if 'None' tags has non-zero scores.
		logits = logits - logits[...,0].unsqueeze(-1)	# 'None' label has zero score.
		batch_size, max_seq_len, _, _ = logits.shape	#n_roles: None, LOC, ...
		n_seq_roles = len(student_tag_dictionary.item2idx)
		bioes_probs = torch.zeros(batch_size, max_seq_len, n_seq_roles)
		for sent_id, sent_scores in enumerate(logits):	# sent_scores: max_seq_len * max_seq_len * n_role
			sent_len = self.lengths[sent_id]
			bioes_prob_sent = self._distill_dp_sent_exp(batch[sent_id], sent_scores, sent_len,student_tag_dictionary, temperature)	# seq_len * seq_roles
			# bioes_prob_sent = self._distill_dp_sent_logsumexp_1(batch[sent_id], sent_scores, sent_len,student_tag_dictionary, temperature)	# seq_len * seq_roles
			bioes_probs[sent_id, :sent_len, :] = bioes_prob_sent

		return bioes_probs




	def _distill_dp_sent_logsumexp_1(self, sent, sent_scores, sent_len, student_tag_dictionary, temperature):
		"""
		sent_scores: score for a sentence, [batch, head, tail, role]
		"""
		import math
		from decimal import Decimal 
		self.sent = sent
		sent_scores = sent_scores.detach().cpu()
		F=0
		O=0; B=1;I=2;E=3;S=4
		# NEG_INF = -float('inf')
		NEG_INF = -1e32
		# prefix_dict = {'O':O, 'B':B, 'I':I, 'E':E, 'S':S}
		prefix_tags = [b'O',b'B',b'I',b'E',b'S']
		# R = self.tag_dictionary.idx2item	# [b'None', b'ORG', b'MISC', b'LOC', b'PER']
		# rel_scores: sent_len * sent_len * n_roles
		Roles = self.tag_dictionary.idx2item[1:]
		role2id = self.tag_dictionary.item2idx
		prefixes = [B,I,E,S]    # 'O' is not prefix
		n = sent_len    
		# DPF, DPB is in log domain.( log_sum_exp version)
		DPF = {}
		DPB = {}

		def _dec(num):
			return Decimal(str(num))
		def _dec_exp(num):
			return _dec(math.exp(num))
		def _dec_log(num):
			return _dec(math.log(num))
		def _dec_lse(list_x):
			if len(list_x)>0:
				m = max(list_x)
				list_x1 = [_dec_exp(x-m) for x in list_x]
				return _dec_log(sum(list_x1)) + m
			else:
				return _dec(NEG_INF)

		def _exp(num):
			return math.exp(num)
		def _log(num):
			return math.log(num)

		def _lse(list_x):
			"""
			list_x: list of numbers
			"""
			if len(list_x)>0:
				m = max(list_x)
				list_x1 = [_exp(x-m) for x in list_x]
				return _log(sum(list_x1)) + m
			else:
				return NEG_INF

		# def _exp_score(head, tail, role):
		# 	# pdb.set_trace()
		# 	return math.exp(sent_scores[head-1, tail-1, role2id[role]])
		def _get_score(head, tail, role):
			return sent_scores[head-1, tail-1, role2id[role]].item()

		# aux function for calculating of prefix 'I'
		def _score_ij(sent_scores, DPF, DPB):
			score_ij = {}
			for role in Roles:
				for i in range(1, n):
					for j in range(2, n+1):
						score_ij[(i,j,role)]= _get_score(i,j,role) + DPF[(i,F)] + DPB[(j,F)]
			return score_ij

		def _DPI(sent, score_ij, DPC):
			"""
			calculate lse score of I-tag
			"""
			if n>2:
				cum = {}
				for role in Roles:
					cum[(2,2,role)] = sum([_exp(score_ij[(1,v,role)]) for v in range(3,n+1)])
					for i in range(2,n-1):
						sum1 = sum([_exp(score_ij[(u,i+1,role)]) for u in range(1,i)])
						sum2 = sum([_exp(score_ij[(i,v,role)]) for v in range(i+2, n+1)])
						cum[(i+1,i+1,role)] = max(cum[(i,i,role)] - sum1 + sum2, 0)
						# if cum[(i+1,i+1,role)] < 0:
						# 	cum[(i+1,i+1,role)] = 0
					# socre of tags with prefix 'I'
					DPC[(I,role,1)] = 0
					for m in range(2,n):
						if cum[(m,m,role)]==0:
							DPC[(I, role, m)] = NEG_INF
						else:
							DPC[(I, role, m)] = math.log(float(cum[(m, m, role)]))
			return DPC

		#initialization
		for i in range(1,n+1):
			DPF[(i,F)] = NEG_INF
			DPB[(i,F)] = NEG_INF
			for role in Roles:
				DPF[(i,role)] = NEG_INF
				DPB[(i,role)] = NEG_INF
			

		# base case
		for role in Roles:
			DPF[(1, role)] = _get_score(1,1,role)
			DPB[(n, role)] = _get_score(n,n,role)
		DPF[(1, F)] = 0
		DPB[(n, F)] = 0

		#define rules for DPF(m, F) and DPB(m, F)
		# DPF(m, R) = exp(S(1,m,R))*DPF(1,F) + exp(S(2,m,R))*DPF(2,F) + ... + exp(S(m,m,R))*DPF(m,F)
		# DPF(m, F) = DPF(m-1, F) + DPF(m-1, role1) + DPF(m-2, role2) + ... 
		
		rules_F = []
		for m in range(2,n+1):
			#DPF(m,F)
			rule_f = [(m, F), (m-1,F)]
			for role in Roles:
				rule_f.append((m-1, role))
			rules_F.append(rule_f)
			#DPF(m,R)
			for role in Roles:
				rule_fr = [(m, role)]
				for i in range(1, m+1):
					rule_fr.append((i, F))
				rules_F.append(rule_fr)

		rules_B = []
		for m in range(n-1, 0, -1):
			rule_b = [(m, F), (m+1, F)]
			for role in Roles:
				rule_b.append((m+1, role))
			rules_B.append(rule_b)
			#DPB(m, R)
			for role in Roles:
				rule_br = [(m, role)]
				for j in range(m, n+1):
					rule_br.append((j, F))
				rules_B.append(rule_br)
		
		# an example of a sentence length n=2 , 
		# rules_F= [[(2, 0), (1, 0), (1, b'ORG'), (1, b'MISC'), (1, b'LOC'), (1, b'PER')], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)]]
		# rules_B= [[(1, 0), (2, 0), (2, b'ORG'), (2, b'MISC'), (2, b'LOC'), (2, b'PER')], [(1, b'ORG'), (1, 0), (2, 0)], [(1, b'MISC'), (1, 0), (2, 0)], [(1, b'LOC'), (1, 0), (2, 0)], [(1, b'PER'), (1, 0), (2, 0)]]
		
		# update DPF, DPB
		for rule_entries in rules_F: # l_entry: (m, F) or (m, role)
			l_entry = rule_entries[0]
			r_entries = rule_entries[1:]
			lse_input_list = []
			m, tag = l_entry
			if tag in {0, '0'}: #DPF(m,F)
				for r_entry in r_entries:
					lse_input_list.append(DPF[r_entry])
					# score += DPF[r_entry]
				DPF[l_entry] = _lse(lse_input_list)
			else:               #DPF(m, R)
				for r_entry in r_entries:
					i, role_mask = r_entry
					lse_input_list.append(_get_score(i,m,tag) + DPF[(i, role_mask)])
				DPF[l_entry] =  _lse(lse_input_list)
				
		for rule_entries in rules_B:
			l_entry = rule_entries[0]
			r_entries = rule_entries[1:]
			lse_input_list = []
			m, tag = l_entry
			if tag in {0, '0'}:
				for r_entry in r_entries:
					lse_input_list.append(DPB[r_entry])
				DPB[l_entry] = _lse(lse_input_list)
			else:               #DPF(m, R)
				for r_entry in r_entries:
					j, role_mask = r_entry
					lse_input_list.append(_get_score(m,j,tag) + DPB[(j,role_mask)])
				DPB[l_entry] = _lse(lse_input_list)
		

		#------------------------------------
		# caculate scores for BIOES tags in sequence model
		DPC = {}
		# student_tags = ['O']
		tag_indices = {}
		#initialization
		for m in range(1,n+1):
			DPC[(O, m)] = NEG_INF
			for prefix in prefixes:	#[B,I,E,S]
				for role in Roles:	#[b'ORG', b'MISC', b'LOC', b'PER']
					# student_tags.append(prefix_tags[prefix] + b'-' + role)
					tag_indices[(prefix, role)] = prefix_tags[prefix] + b'-' + role
					DPC[(prefix, role, m)] = NEG_INF
		score_ij = _score_ij(sent_scores, DPF, DPB)
		#calculate DPC(I-R, m)
		DPC = _DPI(sent, score_ij, DPC)
		#update DPC
		for m in range(1, n+1):
			DPC[(O, m)] = DPF[(m,F)] + DPB[(m,F)]
			for role in Roles:
				DPC[(S, role, m)] = DPF[(m, F)] + _get_score(m,m,role) + DPB[(m, F)]

				# DPC[(B, role, m)] = DPF[(m, F)]*DPB[(m, role)] - DPC[(S, role, m)]
				#calculate B-tag score
				if m<n:
					lse_input_list = []
					for j in range(m+1, n+1):
						lse_input_list.append(_get_score(m,j,role) + DPB[(j,F)])
					DPC[(B, role, m)] = DPF[(m, F)] + _lse(lse_input_list)

				# DPC[(E, role, m)] = DPF[(m, role)]*DPB[(m,F)] - DPC[(S, role, m)]
				#calculate E tag score.
				if m>1:
					lse_input_list = []
					for i in range(1, m):
						lse_input_list.append(_get_score(i, m, role) + DPF[(i, F)])
					DPC[(E,role, m)] = DPB[(m, F)] + _lse(lse_input_list)
				
		for role in Roles:
			DPC[(B, role, n)] = NEG_INF
			DPC[(I, role, 1)] = DPC[(I, role, n)] = NEG_INF
			DPC[(E, role, 1)] = NEG_INF


		# {b'<unk>': 0, b'O': 1, b'B-PER': 2, b'E-PER': 3, b'S-LOC': 4, b'B-MISC': 5, b'I-MISC': 6, b'E-MISC': 7, b'S-MISC': 8, b'S-PER': 9, b'B-ORG': 10, b'E-ORG': 11, b'S-ORG': 12, b'I-ORG': 13, b'B-LOC': 14, b'E-LOC': 15, b'I-PER': 16, b'I-LOC': 17, b'<START>': 18, b'<STOP>': 19}
		student_tags_map = student_tag_dictionary.item2idx	
		bioes_prob_sent = torch.zeros(size=(n, len(student_tags_map)))

		
		for k, student_tag in tag_indices.items():
			student_tag_index = student_tags_map[student_tag]
			tag_indices[k] = student_tag_index

		T = temperature
		for m in range(1,n+1):
			# exp_scores = torch.zeros(len(student_tags_map))
			Z = 0
			for key in DPC:
				if key[-1]==m:
					try:
						Z += _exp(DPC[key]/T)
					except:
						pdb.set_trace()
			prob = max(float(_exp(DPC[(O, m)]/T)/Z), 0)

			bioes_prob_sent[m-1, student_tags_map[b'O']] = prob
			for prefix in [B,I,E,S]:
				for role in Roles:
					prob = max(float(_exp(DPC[(prefix, role, m)]/T)/Z), 0)

					bioes_prob_sent[m-1, tag_indices[prefix, role]] = prob


		return bioes_prob_sent

	def _distill_dp_sent_exp(self, sent, sent_scores, sent_len, student_tag_dictionary, temperature):
		"""
		sent_scores: score for a sentence, [batch, head, tail, role]
		"""
		# pdb.set_trace()
		import math
		from decimal import Decimal 
		sent_scores = sent_scores.detach().cpu()
		self.sent  = sent
		F=0
		O=0; B=1;I=2;E=3;S=4
		# prefix_dict = {'O':O, 'B':B, 'I':I, 'E':E, 'S':S}
		prefix_tags = [b'O',b'B',b'I',b'E',b'S']
		# R = self.tag_dictionary.idx2item	# [b'None', b'ORG', b'MISC', b'LOC', b'PER']
		# rel_scores: sent_len * sent_len * n_roles
		Roles = self.tag_dictionary.idx2item[1:]
		role2id = self.tag_dictionary.item2idx
		prefixes = [B,I,E,S]    # 'O' is not prefix
		n = sent_len    
		DPF = {}
		DPB = {}

		def _dec(num):
			return Decimal(str(num))
		def _dec_exp(num):
			return num.exp()
		def _dec_log(num):
			return num.ln()

		def _exp(num):
			return _dec(math.exp(num))
		def _log(num):
			return _dec(math.log(num))
			

		def _exp_score(head, tail, role):
			return _dec(sent_scores[head-1, tail-1, role2id[role]].item()).exp()

		# aux function for calculating of prefix 'I'
		def _score_ij(sent_scores, DPF, DPB):
			score_ij = {}
			for role in Roles:
				for i in range(1, n-1):
					for j in range(2, n+1):
						score_ij[(i,j,role)]= _exp_score(i,j,role) * DPF[(i,F)]*DPB[(j,F)]
			return score_ij

		def _DPI(score_ij, DPC):
			#intial
			cum = {}
			for role in Roles:
				cum[(1,1,role)] = _dec(0)
				cum[(n,n,role)] = _dec(0)
				for i in range(1,n-1):
					# for j in range(i-1, i+1):
						#calculate cum(i+1,j,R)
						cum[(i+1, i+1, role)] = cum[(i,i,role)]
						for v in range(i+2, n+1):
							cum[(i+1, i+1, role)] += score_ij[(i, v, role)]
						#calculate cum(i+1,j+1,R)
						for u in range(1,i):
							cum[(i+1,i+1, role)] -= score_ij[(u,i+1,role)]
				# socre of tags with prefix 'I'
				for m in range(1,n+1):
					DPC[(I, role, m)] = cum[(m, m, role)]
				
			return DPC

		#initialization
		for i in range(1,n+1):
			DPF[(i,F)] = _dec(0)
			DPB[(i,F)] = _dec(0)
			for role in Roles:
				DPF[(i,role)] = _dec(0)
				DPB[(i,role)] = _dec(0)
			

		# base case
		for role in Roles:
			DPF[(1, role)] = _exp_score(1,1,role)
			DPB[(n, role)] = _exp_score(n,n,role)
		DPF[(1, F)] = _dec(1)
		DPB[(n, F)] = _dec(1)

		#define rules for DPF(m, F) and DPB(m, F)
		# DPF(m, R) = exp(S(1,m,R))*DPF(1,F) + exp(S(2,m,R))*DPF(2,F) + ... + exp(S(m,m,R))*DPF(m,F)
		# DPF(m, F) = DPF(m-1, F) + DPF(m-1, role1) + DPF(m-2, role2) + ... 
		
		rules_F = []
		for m in range(2,n+1):
			#DPF(m,F)
			rule_f = [(m, F), (m-1,F)]
			for role in Roles:
				rule_f.append((m-1, role))
			rules_F.append(rule_f)
			#DPF(m,R)
			for role in Roles:
				rule_fr = [(m, role)]
				for i in range(1, m+1):
					rule_fr.append((i, F))
				rules_F.append(rule_fr)

		rules_B = []
		for m in range(n-1, 0, -1):
			rule_b = [(m, F), (m+1, F)]
			for role in Roles:
				rule_b.append((m+1, role))
			rules_B.append(rule_b)
			#DPB(m, R)
			for role in Roles:
				rule_br = [(m, role)]
				for j in range(m, n+1):
					rule_br.append((j, F))
				rules_B.append(rule_br)
		
		# an example of a sentence length n=2 , 
		# rules_F= [[(2, 0), (1, 0), (1, b'ORG'), (1, b'MISC'), (1, b'LOC'), (1, b'PER')], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)], [(2, b'PER'), (1, 0), (2, 0)]]
		# rules_B= [[(1, 0), (2, 0), (2, b'ORG'), (2, b'MISC'), (2, b'LOC'), (2, b'PER')], [(1, b'ORG'), (1, 0), (2, 0)], [(1, b'MISC'), (1, 0), (2, 0)], [(1, b'LOC'), (1, 0), (2, 0)], [(1, b'PER'), (1, 0), (2, 0)]]
		
		# update DPF, DPB
		for rule_entries in rules_F: # l_entry: (m, F) or (m, role)
			l_entry = rule_entries[0]
			r_entries = rule_entries[1:]
			score = _dec(0)
			m, tag = l_entry
			if tag in {0, '0'}: #DPF(m,F)
				for r_entry in r_entries:
					score += DPF[r_entry]
				DPF[l_entry] = score
			else:               #DPF(m, R)
				for r_entry in r_entries:
					i, role_mask = r_entry
					score += _exp_score(i,m,tag)*DPF[(i, role_mask)]
				DPF[l_entry] = score
				
	
		# for m in range(m, 0, -1):
		for rule_entries in rules_B:
			l_entry = rule_entries[0]
			r_entries = rule_entries[1:]
			score = _dec(0)
			m, tag = l_entry
			if tag in {0, '0'}:
				for r_entry in r_entries:
					score += DPB[r_entry]
				DPB[l_entry] = score
			else:               #DPF(m, R)
				for r_entry in r_entries:
					j, role_mask = r_entry
					score += _exp_score(m,j,tag)*DPB[(j,role_mask)]
				DPB[l_entry] = score



		#------------------------------------
		# caculate scores for BIOES tags in sequence model
		DPC = {}
		# student_tags = ['O']
		tag_indices = {}
		#initialization
		for m in range(1,n+1):
			DPC[(O, m)] = _dec(0)
			for prefix in prefixes:	#[B,I,E,S]
				for role in Roles:	#[b'ORG', b'MISC', b'LOC', b'PER']
					# student_tags.append(prefix_tags[prefix] + b'-' + role)
					tag_indices[(prefix, role)] = prefix_tags[prefix] + b'-' + role
					DPC[(prefix, role, m)] = _dec(0)
		score_ij = _score_ij(sent_scores, DPF, DPB)
		#calculate DPC(I-R, m)
		DPC = _DPI(score_ij, DPC)
		#update DPC
		for m in range(1, n+1):
			DPC[(O, m)] = DPF[(m,F)]*DPB[(m,F)]
			for role in Roles:
				DPC[(S, role, m)] = DPF[(m, F)]*_exp_score(m,m,role)*DPB[(m, F)]
				# DPC[(B, role, m)] = DPF[(m, F)]*DPB[(m, role)] - DPC[(S, role, m)]
				DPC[(B, role, m)] = _dec(0)
				for j in range(m+1, n+1):
					DPC[(B,role, m)] += _exp_score(m,j,role)*DPB[(j,F)]
				DPC[(B,role,m)] *= DPF[(m, F)]
				# DPC[(E, role, m)] = DPF[(m, role)]*DPB[(m,F)] - DPC[(S, role, m)]
				DPC[(E,role, m)] = _dec(0)
				for i in range(1, m):
					DPC[(E, role, m)] += _exp_score(i, m, role)*DPF[(i, F)]
				DPC[(E, role, m)] *= DPB[(m, F)]
				
		for role in Roles:
			DPC[(B, role, n)] = _dec(0)
			DPC[(I, role, 1)] = DPC[(I, role, n)] = _dec(0)
			DPC[(E, role, 1)] = _dec(0)

		# {b'<unk>': 0, b'O': 1, b'B-PER': 2, b'E-PER': 3, b'S-LOC': 4, b'B-MISC': 5, b'I-MISC': 6, b'E-MISC': 7, b'S-MISC': 8, b'S-PER': 9, b'B-ORG': 10, b'E-ORG': 11, b'S-ORG': 12, b'I-ORG': 13, b'B-LOC': 14, b'E-LOC': 15, b'I-PER': 16, b'I-LOC': 17, b'<START>': 18, b'<STOP>': 19}
		student_tags_map = student_tag_dictionary.item2idx	
		bioes_prob_sent = torch.zeros(size=(n, len(student_tags_map)))
		
		for k, student_tag in tag_indices.items():
			student_tag_index = student_tags_map[student_tag]
			tag_indices[k] = student_tag_index
		T = _dec(temperature)

		for m in range(1,n+1):
			Z = 0
			for key in DPC:
				if key[-1]==m:
					if DPC[key] > 0:	#
						Z += _dec_exp(_dec_log(DPC[key])/T)

			if DPC[(O,m)]>0:
				prob = float(_dec_exp(_dec_log(DPC[(O, m)])/T)/Z)
			else:
				prob = 0

			bioes_prob_sent[m-1, student_tags_map[b'O']] = prob
			for prefix in [B,I,E,S]:
				for role in Roles:
					if DPC[(prefix, role, m)]>0:
						prob = float(_dec_exp(_dec_log(DPC[(prefix, role, m)])/T)/Z)
					else:
						prob = 0
					bioes_prob_sent[m-1, tag_indices[(prefix, role)]] = prob
		
		return bioes_prob_sent


