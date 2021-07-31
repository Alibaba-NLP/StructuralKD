import torch
import torch.nn as nn

from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence



#-----------modules------------------

def get_shape(t):
	return list(t.shape)

class Sparse_dropout(nn.Module):
	def __init__(self, p):
		super(Sparse_dropout, self).__init__()
		self.dropout_rate = p
	
	def forward(self, input, noise_shape):
		dropout_rate = self.dropout_rate if self.training else 0
		shapes = input.shape
		noise_shape = list(noise_shape)
		broadcast_dims = []
		# pdb.set_trace()
		for idx, dim_pair in enumerate(zip(shapes, noise_shape)):
			if dim_pair[1]>1:
				broadcast_dims.append((idx, dim_pair[0]))

		mask_dims = []
		for dim in broadcast_dims:
			mask_dims.append(dim[1])
		mask = torch.bernoulli((torch.ones(mask_dims, device=input.device)*(1-dropout_rate)).reshape(noise_shape))*(1/(1-dropout_rate))
		mask.to(input.dtype)
		return input*mask

class bilinear_classifier(nn.Module):

	def __init__(self, dropout, input_size_x, input_size_y, output_size, bias_x=True, bias_y=True):
		super(bilinear_classifier, self).__init__()
		# self.dropout = dropout
		# self.batch_size = batch_size
		# self.bucket_size = bucket_size
		# self.input_size = input_size
		# pdb.set_trace()
		self.dropout_rate = 0
		self.output_size = output_size
		
		self.dropout = Sparse_dropout(p=self.dropout_rate)
		self.biaffine = biaffine_mapping(
						input_size_x, input_size_y,
						output_size, bias_x, bias_y,
						)
	def forward(self, x_bnv, y_bnv):
		batch_size, input_size_x = x_bnv.shape[0], x_bnv.shape[-1]
		input_size_y = y_bnv.shape[-1]
		noise_shape_x = [batch_size, 1, input_size_x]
		noise_shape_y = [batch_size, 1, input_size_y]
		x = self.dropout(x_bnv, noise_shape_x)
		y = self.dropout(y_bnv, noise_shape_y)

		output = self.biaffine(x, y)
		#TODO reshape output
		if self.output_size == 1:
		  output = output.squeeze(-1)
		return output

class biaffine_mapping(nn.Module):
	def __init__(self, input_size_x, input_size_y, output_size, bias_x, bias_y, initializer=None):
		super(biaffine_mapping, self).__init__()
		self.bias_x = bias_x
		self.bias_y = bias_y
		self.output_size = output_size
		self.initilizer = None
		if self.bias_x:
		  input_size1 = input_size_x + 1
		  input_size2 = input_size_y + 1
		self.biaffine_map = nn.Parameter(torch.Tensor(input_size1, output_size, input_size2))
		
		self.initialize()

	def initialize(self):
		if self.initilizer == None:
			torch.nn.init.orthogonal_(self.biaffine_map)
		else:
			self.initilizer(self.biaffine_map)


	def forward(self, x, y):
		batch_size, bucket_size = x.shape[0], x.shape[1]
		if self.bias_x:
		  x = torch.cat([x, torch.ones([batch_size, bucket_size, 1], device=x.device)], axis=2)
		if self.bias_y:
		  y = torch.cat([y, torch.ones([batch_size, bucket_size, 1], device=y.device)], axis=2)

		#reshape
		x_set_size, y_set_size = x.shape[-1], y.shape[-1]
		# b,n,v1 -> b*n, v1
		x = x.reshape(-1, x_set_size)
		# # b,n,v2 -> b*n, v2
		# y = y.reshape(-1, y_set_size)
		biaffine_map = self.biaffine_map.reshape(x_set_size, -1)  # v1, r, v2 -> v1, r*v2
		# b*n, r*v2 -> b, n*r, v2
		biaffine_mapping = (torch.matmul(x, biaffine_map)).reshape(batch_size, -1, y_set_size)
		# (b, n*r, v2) bmm (b, n, v2) -> (b, n*r, n) -> (b, n, r, n)
		biaffine_mapping = (biaffine_mapping.bmm(torch.transpose(y, 1, 2))).reshape(batch_size, bucket_size, self.output_size, bucket_size)
		# (b, n, r, n) -> (b, n, n, r)
		biaffine_mapping = biaffine_mapping.transpose(2, 3)

		return biaffine_mapping

def projection(emb_size, output_size, initializer=None):
  return ffnn(emb_size, 0, -1, output_size, dropout=0, output_weights_initializer=initializer)

class ffnn(nn.Module):
  
	def __init__(self, emb_size, num_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
		super(ffnn, self).__init__()
		
		self.dropout = torch.nn.Dropout(p=dropout)
		self.weights = nn.Parameter(torch.Tensor(emb_size, output_size))
		self.bias = nn.Parameter(torch.Tensor(output_size))
		self.activation = torch.nn.ReLU()
		self.num_layers = num_layers
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.initializer = output_weights_initializer

		self.initialize()
		
	def initialize(self):
		if self.initializer == None:
			torch.nn.init.xavier_uniform_(self.weights, gain=1)
		else:
			# pdb.set_trace()
			self.initializer(self.weights, gain=1)
		nn.init.zeros_(self.bias)

	def forward(self, inputs):
		# pdb.set_trace()
		current_inputs = inputs
		if len(get_shape(inputs))==3:
			batch_size, seqlen, emb_size = get_shape(inputs)
			current_inputs = inputs.reshape(batch_size*seqlen, emb_size)
		emb_size = get_shape(current_inputs)[-1]
		# if emb_size != self.emb_size:
		# 	pdb.set_trace()
		assert emb_size==self.emb_size,'last dim of input does not match this layer'
		
		# if self.dropout is not None or self.dropout > 0:
		# 	output = self.dropout(current_inputs)
		#TODO num_layers>0 case.

		outputs = current_inputs.matmul(self.weights) + self.bias

		if len(get_shape(inputs))==3:
			outputs = outputs.reshape(batch_size, seqlen, self.output_size)
		
		return outputs

class BiLSTM_1(nn.Module):

	def __init__(self, input_size, hidden_size, num_layers, dropout=None):
		super(BiLSTM_1, self).__init__()

		self.input_size = input_size	#emb_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout_rate = dropout
		
		self.f_cells = nn.ModuleList()
		self.b_cells = nn.ModuleList()
		
		for _ in range(self.num_layers):
			self.f_cells.append(LstmCell(input_size, hidden_size, dropout))
			self.b_cells.append(LstmCell(input_size, hidden_size, dropout))

			input_size = 2*hidden_size

		self.dropout = torch.nn.Dropout(p=dropout)
		self.mlp = projection(emb_size=input_size, output_size=input_size)
		# self.initialize()

	def __repr__(self):
		s = self.__class__.__name__ + '('
		s += f"{self.input_size}, {self.hidden_size}"
		if self.num_layers > 1:
			s += f", num_layers={self.num_layers}"
		if self.dropout_rate > 0:
			s += f", dropout={self.dropout_rate}"
		s += ')'
		return s

	def permute_hidden(self, hx, permutation):
		if permutation is None:
			return hx
		h = apply_permutation(hx[0], permutation)
		c = apply_permutation(hx[1], permutation)

		return h, c

	def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
		hx_0 = hx_i = hx
		hx_n, output = [], []
		steps = reversed(range(len(x))) if reverse else range(len(x))
		# if self.training:
		#     hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

		for t in steps:
			last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
			if last_batch_size < batch_size:
				hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
						for h, ih in zip(hx_i, hx_0)]
			else:
				hx_n.append([h[batch_size:] for h in hx_i])
				hx_i = [h[:batch_size] for h in hx_i]
			# pdb.set_trace()
			hx_i = [h for h in cell(x[t], hx_i)]
			output.append(hx_i[0])
			# if self.training:
			#     hx_i[0] = hx_i[0] * hid_mask[:batch_size]
		if reverse:
			hx_n = hx_i
			output.reverse()
		else:
			hx_n.append(hx_i)
			hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
		# pdb.set_trace()
		output = torch.cat(output)

		return output, hx_n


	def forward(self, sequence, hx=None):
		# pdb.set_trace()
		x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
		
		batch_size = batch_sizes[0]
		h_n, c_n = [], []

		if hx is None:
			# pdb.set_trace()
			
			h = self.f_cells[0].initial_state[0].repeat([batch_size, 1])
			c = self.f_cells[0].initial_state[1].repeat([batch_size, 1])

			h = torch.unsqueeze(torch.unsqueeze(h, 0), 0).repeat([self.num_layers, 2, 1, 1])
			c = torch.unsqueeze(torch.unsqueeze(c, 0), 0).repeat([self.num_layers, 2, 1, 1])
		else:
			h, c = self.permute_hidden(hx, sequence.sorted_indices)
		h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
		c = c.view(self.num_layers, 2, batch_size, self.hidden_size)
		
		for i in range(self.num_layers):
			current_input = x
			x = torch.split(x, batch_sizes)
			
			# if self.training:
			# 	mask = SharedDropout.get_mask(x[0], self.dropout)
			# 	x = [i * mask[:len(i)] for i in x]
			x_f, (h_f, c_f) = self.layer_forward(x=x,
												 hx=(h[i,0], c[i,0]),
												 cell=self.f_cells[i],
												 batch_sizes=batch_sizes												 
												 )
			x_b, (h_b, c_b) = self.layer_forward(x=x,
												 hx=(h[i, 1], c[i, 1]),
												 cell=self.b_cells[i],
												 batch_sizes=batch_sizes,
												 reverse=True)			
			h_n.append(torch.stack((h_f, h_b)))
			c_n.append(torch.stack((c_f, c_b)))
			text_outputs = torch.cat((x_f, x_b), -1)
			text_outputs = self.dropout(text_outputs)
			if i > 0:
				# pdb.set_trace()
				highway_gates = torch.sigmoid(self.mlp(text_outputs))
				text_outputs = highway_gates*text_outputs + (1-highway_gates)*current_input
			x = text_outputs


		x = PackedSequence(x,
						   sequence.batch_sizes,
						   sequence.sorted_indices,
						   sequence.unsorted_indices)
		hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
		hx = self.permute_hidden(hx, sequence.unsorted_indices)

		return x, hx

class LstmCell(nn.Module):
	def __init__(self, input_size, hidden_size, dropout=0):
		super(LstmCell, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = torch.nn.Dropout(p=dropout)
		self.mlp = projection(emb_size=input_size+hidden_size, output_size=3*hidden_size, 
							initializer=self._block_orthonormal_initializer(output_sizes=[hidden_size] * 3)
							)
		
		self.initial_cell_state = nn.Parameter(torch.Tensor(1, hidden_size))
		self.initial_hidden_state = nn.Parameter(torch.Tensor(1, hidden_size))
		self.initialize()
		self._initial_state = (self.initial_cell_state, self.initial_hidden_state)
		

	def initialize(self):
		torch.nn.init.xavier_uniform_(self.initial_cell_state, gain=1)
		torch.nn.init.xavier_uniform_(self.initial_hidden_state, gain=1)

	def forward(self, inputs, states):
		batch_size = get_shape(inputs)[0]
		_dropout_mask = self.dropout(torch.ones(batch_size, self.hidden_size, device=inputs.device))
		h, c = states
		if self.training:
			h *= _dropout_mask
		# pdb.set_trace()
		concat = self.mlp(inputs=torch.cat([inputs, h], axis=1))	
		i, j, o = torch.chunk(input=concat, chunks=3, dim=1)
		i = torch.sigmoid(i)
		new_c = (1-i)*c + i*torch.tanh(j)		
		new_h = torch.tanh(new_c) * torch.sigmoid(o)	
		new_state = (new_h, new_c)
		return new_state

	@property
	def initial_state(self):
		return self._initial_state
	

	def _orthonormal_initializer(self, weights, gain=1.0):
		if len(weights.shape)>2:
			pdb.set_trace()
		device = weights.device
		dtype = weights.dtype
		# pdb.set_trace()
		shape0, shape1 = get_shape(weights)
		M1 = torch.randn(size=(shape0, shape0), dtype=dtype, device=device)
		M2 = torch.randn(size=(shape1, shape1), dtype=dtype, device=device)
		Q1, R1 = torch.qr(M1)	# let weights.shape= (s0,s1) and sm = min(s0, s1), then Q1:(s0,sm), R1:(sm,s1)
		Q2, R2 = torch.qr(M2)
		Q1 = Q1 * torch.sign(torch.diag(R1))
		Q2 = Q2 * torch.sign(torch.diag(R2))
		n_min = min(shape0, shape1)
		
		with torch.no_grad():
			q = torch.matmul(Q1[:, :n_min], Q2[:n_min, :])
			weights.view_as(q).copy_(q)
			weights.mul_(gain)
		return weights

	def _block_orthonormal_initializer(self, output_sizes):
		def _initializer(weights, gain=1.0):
			shape = get_shape(weights)
			assert len(shape) == 2
			assert sum(output_sizes) == shape[1]
			initializer = self._orthonormal_initializer
			
			
			with torch.no_grad():
				# pdb.set_trace()
				q_list = [initializer(a, gain) for a in torch.split(weights,split_size_or_sections=output_sizes, dim=1)]
				q = torch.cat(q_list, axis=1)
				weights.view_as(q).copy_(q)
			return weights
		return _initializer

class cnn(nn.Module):
	def __init__(self, emb_size, kernel_sizes, num_filter):
		super(cnn, self).__init__()
		self.emb_size = emb_size
		self.num_layers = len(kernel_sizes)
		self.conv_layers = nn.ModuleList()
		# self.weights = nn.ModuleList()
		# self.biases = nn.ModuleList()
		
		for i, filter_size in enumerate(kernel_sizes):
			self.conv_layers.append(cnn_layer(in_channels=emb_size, out_channels=num_filter, 
											  kernel_size=kernel_sizes[i], stride=1, 
											  padding=0, bias=True))
	
	def forward(self, input):
		outputs = []
		# pdb.set_trace()
		for i in range(self.num_layers):
			output = self.conv_layers[i](input)	# (n_words, n_chars-filter_size+1, n_filters)
			pooled = torch.max(output, dim=2)[0]	# channel is dim1.
			outputs.append(pooled)
		return torch.cat(outputs, 1)
	
class cnn_layer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
		super(cnn_layer, self).__init__()
		self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
									kernel_size=kernel_size, stride=stride, 
									padding=padding, bias=bias)
		self.relu = torch.nn.ReLU()
	def forward(self, input):
		return self.relu(self.conv(input))