import os
base_path = 'config/with_unlabeled/kd-2nd'
def rec_config():
	files = os.listdir(base_path)
	for file_name in files:
		str = ['3000', '10000', '30000', '50000', '100000']
		for i in str:
			with open(os.path.join(base_path,file_name),'r') as fr:
				new_file_name = file_name.replace('3000', i, 1)
				with open(os.path.join(base_path,new_file_name), 'w') as fw:
					for line in fr:
						if 'train_file' in line:
							line = line.replace('3000', i, 1)
							fw.write(line)
						elif 'model_name' in line:
							line = line.replace('3000', i, 1)
							fw.write(line)
						else:
							fw.write(line)




def copy_config():
	copy_path = 'config/with_unlabeled/kd/ptb-dp_as_sl-anneal0.5-temp3-100000.yaml'
	for i in range(4):
		with open(copy_path, 'r') as fr:
			write_path = copy_path.rstrip('.yaml')+'-'+str(i)+'.yaml'
			with open(write_path, 'w') as fw:
				for line in fr:
					if 'model_name' in line:
						line = line.rstrip() + '-'+str(i)+'\n'
						fw.write(line)
					else:
						fw.write(line)

import yaml
# def get_yaml_data(yaml_file):
#     # 打开yaml文件
#     # print("***获取yaml文件数据***")
#     file = open(yaml_file, 'r', encoding="utf-8")
#     file_data = file.read()
#     file.close()
#     data = yaml.load(file_data, Loader=yaml.SafeLoader)
#     return data
def teacher_config():
	files = os.listdir(base_path)
	for file_name in files:
		str = ['3000', '10000', '30000', '50000', '100000']
		for i in str:
			with open(os.path.join(base_path,file_name),'r') as fr:
				new_file_name = file_name.rstrip('.yaml')+'-'+i+'.yaml'
				config = yaml.load(fr, Loader=yaml.SafeLoader)
				with open(os.path.join(base_path,new_file_name), 'w') as fw:
					# for line in fr:
					# 	if 'train_file' in line:
					# 		train_file = 'train_modified.bllip.60.3000.conllu.sl'
					# 		line = line.split(':')[0]+':'+'\t'+train_file.replace('3000', i, 1)+'\n'
					# 		fw.write(line)
					# 	elif 'model_name' in line:
					# 		line = line.rstrip()+'-'+i+'\n'
					# 		fw.write(line)
					# 	else:
					# 		fw.write(line)
					train_file = 'train_modified.bllip.60.3000.conllu.sl'
					train_file = train_file.replace('3000', i, 1)
					config['dep']['train_file'] = train_file
					config['model_name'] = config['model_name'].rstrip()+'-'+i
					_ = yaml.safe_dump(config, fw, encoding='utf-8', allow_unicode=True, default_flow_style=False)

# def process():
#
# 	files = os.listdir(base_path)
# 	for file_name in files:
# 		with open(os.path.join(base_path, file_name), 'r') as fr:
# 			for line in fr:
# 				if 'train_file' in line:
# 					print(line.lstrip('\t'))
# 				if 'dev_file' in line:
# 					print(line)
# 					exit()
def change_config():
	files = os.listdir(base_path)
	for file_name in files:
		with open(os.path.join(base_path, file_name), 'r') as fr:
			new_file_name = file_name.rstrip('.yaml') + '-2nd' + '.yaml'
			config = yaml.load(fr, Loader=yaml.SafeLoader)
			with open(os.path.join(base_path, new_file_name), 'w') as fw:
				train_file = config['dep']['train_file'].replace('.60','',1)
				config['dep']['train_file'] = train_file
				config['model_name'] = config['model_name'].rstrip() + '-' + '2nd'
				config['dep']['teachers'] = config['dep']['teachers'].replace('nocrf','2nd_nocrf',1)
				_ = yaml.safe_dump(config, fw, encoding='utf-8', allow_unicode=True, default_flow_style=False)


def split_sh(sh_file):
	# sh_file = 'run_unlabeled_2nd.sh'
	num_line = 0
	comm = {'3000':[],'10000':[],'30000':[],'50000':[],'100000':[]}
	with open(sh_file, 'r') as fr:
		for line in fr:
			if '-3000-' in line:
				comm['3000'].append(line)
			elif '-10000-' in line:
				comm['10000'].append(line)
			elif '-30000-' in line:
				comm['30000'].append(line)
			elif '-50000' in line:
				comm['50000'].append(line)
			elif '-100000-' in line:
				comm['50000'].append(line)
		for i in comm:
			w_file = 'run_unlabeled_2nd_'+i+'.sh'
			with open(w_file,'w') as fw:
				for k, string in enumerate(comm[i]):
					fw.write(string.replace('CUDA_VISIBLE_DEVICES=3','CUDA_VISIBLE_DEVICES='+str(k%4)))


def gen_sh():
	str1 = 'CUDA_VISIBLE_DEVICES=3 nohup python train_with_teacher.py --config config/with_unlabeled/kd-2nd/'
	str2= '>logs/'
	with open('run_unlabeled_teacher_2nd.sh','w') as fw:
		for _, _, files in os.walk(base_path,topdown=False):
			for file_name in files:
				if '2nd' in file_name:
					fw.write(str1+file_name+'\t'+str2+file_name+'.log&'+'\n')
gen_sh()
split_sh('run_unlabeled_teacher_2nd.sh')