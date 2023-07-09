# uncompyle6 version 3.6.5
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.6.10 |Anaconda, Inc.| (default, Jan  7 2020, 21:14:29) 
# [GCC 7.3.0]
# Embedded file name: /root/workspace/flair/tookits/cfg_parser.py
# Compiled at: 2020-03-17 14:43:42
import pdb

class cfg_parser(object):
    """docstring for cfg_parser"""
    __module__ = __name__

    def __init__(self, args=None):
        super(cfg_parser, self).__init__()

    def parse(self, cfg, kwargs=None, trainer_name='ModelFinetuner', model_class='SequenceTagger'):
        if kwargs == None:
            kwargs = {}
        split_cfg = cfg.split('_')
        for parameter in split_cfg:
            if 'batch' in parameter:
                kwargs['train']['mini_batch_size'] = int(parameter.split('batch')[0])
                continue
            elif 'lr' in parameter and 'lrrate' not in parameter:
                kwargs['train']['learning_rate'] = float(parameter.split('lr')[0])
                continue
            elif 'decay' in parameter and 'weightdecay' not in parameter:
                kwargs['train']['anneal_factor'] = float('.' + parameter.split('decay')[0])
                continue
            elif 'hidden' in parameter:
                kwargs['model'][model_class]['hidden_size'] = int(parameter.split('hidden')[0])
                continue
            elif 'lrrate' in parameter:
                kwargs['train']['lr_rate'] = int(parameter.split('lrrate')[0])
                continue
            elif 'inter' in parameter:
                if model_class == 'SequenceTagger':
                    kwargs['model'][model_class]['posterior_interpolation'] = float(parameter.split('inter')[0])
                else:
                    kwargs['model'][model_class]['interpolation'] = float(parameter.split('inter')[0])
                continue
            elif 'attentropy' in parameter:
                kwargs['train']['entropy_loss_rate'] = float(parameter.split('attentropy')[0])
                continue
            elif 'iter' in parameter:
                kwargs['MFVI']['iterations'] = int(parameter.split('iter')[0])
                continue
            elif 'window' in parameter:
                kwargs['MFVI']['window_size'] = int(parameter.split('window')[0])
                continue
            elif 'quadstd' in parameter:
                kwargs['MFVI']['quad_std'] = float(parameter.split('quadstd')[0])
                continue
            elif 'quadrank' in parameter:
                kwargs['MFVI']['quad_rank'] = int(parameter.split('quadrank')[0])
                continue
            elif 'hexastd' in parameter:
                kwargs['MFVI']['hexa_std'] = float(parameter.split('hexastd')[0])
                continue
            elif 'hexarank' in parameter:
                kwargs['MFVI']['hexa_rank'] = int(parameter.split('hexarank')[0])
                continue
            elif 'tag' in parameter:
                kwargs['MFVI']['tag_dim'] = int(parameter.split('tag')[0])
                continue
            elif 'anneal' in parameter:
                kwargs['teacher_annealing'] = True
                kwargs['anneal_factor'] = float(parameter.split('anneal')[0])
                continue
            elif 'professor' in parameter:
                kwargs['train']['professor_interpolation'] = float('.' + parameter.split('professor')[0])
                continue
            elif 'patience' in parameter:
                kwargs['train']['patience'] = int(parameter.split('patience')[0])
                continue
            elif 'epoch' in parameter:
                kwargs['train']['max_epochs'] = int(parameter.split('epoch')[0])
                continue
            elif 'layer' in parameter:
                kwargs['model'][model_class]['rnn_layers'] = int(parameter.split('layer')[0])
            elif 'best' in parameter:
                kwargs['train']['best_k'] = int(parameter.split('best')[0])
            elif 'goldkd' in parameter:
                kwargs['model'][model_class]['gold_const'] = float(parameter.split('goldkd')[0])
            elif 'upsample' in parameter:
                val = parameter.split('upsample')[0]
                if val == '':
                    kwargs[trainer_name]['direct_upsample_rate'] = 10
                else:
                    kwargs[trainer_name]['direct_upsample_rate'] = int(val)
            elif 'downsample' in parameter:
                kwargs[trainer_name]['down_sample_amount'] = int(parameter.split('downsample')[0])
            # ner-dp
            elif 'lstmdropout' in parameter:
                kwargs['model'][model_class]['lstm_dropout']=float(parameter.split('lstmdropout')[0])
                continue
            elif 'worddropout' in parameter:
                kwargs['model'][model_class]['word_dropout'] = float(parameter.split('worddropout')[0])
                continue
            elif 'mlpdropout' in parameter:
                kwargs['model'][model_class]['mlp_dropout'] = float(parameter.split('mlpdropout')[0])
                continue
            elif 'mlprel' in parameter:
                kwargs['model'][model_class]['n_mlp_rel'] = int(parameter.split('mlprel')[0])
                continue
            

        return kwargs
# okay decompiling cfg_parser.pyc
