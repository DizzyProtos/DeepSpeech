import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid

from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model


class DeepSpeechSTT:
    def __init__(self, vocab_path, mean_std_path, model_path,
                 spectrogram_type='linear',
                 place=fluid.CPUPlace(),
                 num_conv_layers=2, num_rnn_layers=3, rnn_layer_size=2048,
                 use_gru=False, share_rnn_weights=True,
                 decoding_method='ctc_beam_search',
                 beam_size = 500, cutoff_prob=1.0, cutoff_top_n=40,
                 alpha=2.5, beta=0.3, lang_model_path=''):
        self.spectrogram_type = spectrogram_type
        self.place = place
        self.num_conv_layers = num_conv_layers
        self.num_rnn_layers = num_rnn_layers
        self.rnn_layer_size = rnn_layer_size
        self.use_gru = use_gru
        self.share_rnn_weights = share_rnn_weights
        self.decoding_method = decoding_method
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.alpha = alpha
        self.beta = beta
        self.lang_model_path = lang_model_path

        self.data_generator = DataGenerator(
            vocab_filepath=vocab_path,
            mean_std_filepath=mean_std_path,
            augmentation_config='{}',
            specgram_type=spectrogram_type,
            keep_transcription_text=True,
            place=place,
            is_training=False)
        # prepare ASR model
        self.ds2_model = DeepSpeech2Model(
            vocab_size=self.data_generator.vocab_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_layer_size=rnn_layer_size,
            use_gru=use_gru,
            init_from_pretrained_model=model_path,
            place=place,
            share_rnn_weights=share_rnn_weights)

        self.vocab_list = [chars for chars in self.data_generator.vocab_list]

        if decoding_method == "ctc_beam_search":
            self.ds2_model.init_ext_scorer(alpha, beta, lang_model_path, self.vocab_list)

    @staticmethod
    def index_to_sec(index):
        stride_ms = 10
        return index * stride_ms * 3 / 1000

    def transcript_file(self, filename, silence_between_phrases_sec=0.5, visual=False):
        feature = self.data_generator.process_utterance(filename, "")
        audio_len = feature[0].shape[1]
        mask_shape0 = (feature[0].shape[0] - 1) // 2 + 1
        mask_shape1 = (feature[0].shape[1] - 1) // 3 + 1
        mask_max_len = (audio_len - 1) // 3 + 1
        mask_ones = np.ones((mask_shape0, mask_shape1))
        mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
        mask = np.repeat(
            np.reshape(
                np.concatenate((mask_ones, mask_zeros), axis=1),
                (1, mask_shape0, mask_max_len)),
            32,
            axis=0)
        feature = (np.array([feature[0]]).astype('float32'),
                   None,
                   np.array([audio_len]).astype('int64').reshape([-1, 1]),
                   np.array([mask]).astype('float32'))
        probs_split = self.ds2_model.infer_batch_probs(infer_data=feature,
                                                       feeding_dict=self.data_generator.feeding)

        probs_frames = []
        # Mark voice and silence frames
        for i, probs in enumerate(probs_split[0]):
            max_prob_ind = np.argmax(probs)
            if probs[len(self.vocab_list)] > 0.99:
                probs_frames.append((i, 'silence'))
            else:
                probs_frames.append((i, 'voice'))
        # Identify phrases periods
        voice_frames = [frame for frame in probs_frames if frame[1] == 'voice']
        splited_voice_frames = [[]]
        for prev_frame, curr_frame in zip(voice_frames, voice_frames[1:]):
            splited_voice_frames[-1].append(prev_frame)
            space_sec = self.index_to_sec(curr_frame[0] - prev_frame[0])
            # Join pharases if silence is shorter than silence_between_phrases_sec
            if space_sec > silence_between_phrases_sec:
                splited_voice_frames.append([])
                splited_voice_frames[-1].append(curr_frame)
        # phrase - (begin_ind, end_ind)
        phrases_indexes = [(pf[0][0], pf[-1][0]) for pf in splited_voice_frames]

        if visual:
            fig, ax = plt.subplots(1, 1, figsize=(80, 100))
            ax.imshow(probs_split[0], cmap='cool', interpolation='nearest')
            for i, phrase in enumerate(phrases_indexes):
                ax.text(1, phrase[0], f'{phrase[0], self.index_to_sec(phrase[0])}')
                ax.text(1, phrase[1], f'{phrase[1], self.index_to_sec(phrase[1])}')

                ax.hlines(phrase[0], xmin=0, xmax=27, color='red')
                ax.hlines(phrase[1], xmin=0, xmax=27, color='green')
            fig.savefig('probs_split.png')
            plt.close(fig)

        result_transcript_phrases = []
        for phrase in phrases_indexes:
            probs_split_phrase = probs_split[0][phrase[0]:phrase[1]]
            if self.decoding_method == "ctc_greedy":
                transcript = self.ds2_model.decode_batch_greedy(
                    probs_split=[probs_split_phrase],
                    vocab_list=self.vocab_list)
            else:
                transcript = self.ds2_model.decode_batch_beam_search(
                    probs_split=[probs_split_phrase],
                    beam_alpha=self.alpha,
                    beam_beta=self.beta,
                    beam_size=self.beam_size,
                    cutoff_prob=self.cutoff_prob,
                    cutoff_top_n=self.cutoff_top_n,
                    vocab_list=self.vocab_list,
                    num_processes=1)
            if len(transcript[0]) > 0:
                result_transcript_phrases.append({
                    'text': transcript[0],
                    'start': self.index_to_sec(phrase[0]),
                    'end': self.index_to_sec(phrase[1])
                })
        return result_transcript_phrases


if __name__ == '__main__':
    stt_model = DeepSpeechSTT(vocab_path='models/baidu_en8k/vocab.txt',
                              mean_std_path='models/baidu_en8k/mean_std.npz',
                              model_path='models/baidu_en8k',
                              spectrogram_type='linear',
                              lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm')
    text = stt_model.transcript_file(r'/home/testserver/Downloads/call_sample.wav')
    print(text)