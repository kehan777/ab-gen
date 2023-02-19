import logging
import torch
from torch.utils.data import TensorDataset
import numpy as np

logger = logging.getLogger(__name__)


class AASeqDictionary(object):
    """
    A fixed dictionary for protein sequences.
    Enables sequence<->token conversion.
    With a space:0 for padding, B:1 as the start token and end_of_line \n:2 as the stop token.
    """
    PAD, BEGIN, END = ' ', 'B', '\n'

    def __init__(self) -> None:
        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, 'A': 3, 'R': 4, 'N': 5, 'D': 6, 'C': 7, 'E': 8,
                         'Q': 9, 'G': 10, 'H': 11, 'I': 12, 'L': 13, 'K': 14, 'M': 15, 'F': 16, 'P': 17, 'S': 18,
                         'T': 19, 'W': 20, 'Y': 21, 'V': 22, 'X': 23}  # X for unknown AA
        self.idx_char = {v: k for k, v in self.char_idx.items()}

    def get_char_num(self) -> int:
        """
        Returns:
            number of characters in the alphabet
        """
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[self.PAD]

    def matrix_to_seqs(self, array):
        """
        Converts a matrix of indices into their Sequence representations
        Args:
            array: torch tensor of indices, one sequence per row

        Returns: a list of Sequence, without the termination symbol
        """
        seqs_strings = []

        for row in array:
            predicted_chars = []

            for j in row:
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)

            seq = ''.join(predicted_chars)
            seqs_strings.append(seq)

        return seqs_strings

    def seqs_to_matrix(self, seqs, max_len=100):
        """
        Converts a list of seqs into a matrix of indices

        Args:
            seqs: a list of Sequence, without the termination symbol
            max_len: the maximum length of seqs to encode, default=100

        Returns: a torch tensor of indices for all the seqs
        """
        batch_size = len(seqs)
        seqs = [self.BEGIN + seq + self.END for seq in seqs]
        idx_matrix = torch.zeros((batch_size, max_len))
        for i, seq in enumerate(seqs):
            enc_seq = self.BEGIN + seq + self.END
            for j in range(max_len):
                if j >= len(enc_seq):
                    break
                idx_matrix[i, j] = self.char_idx[enc_seq[j]]

        return idx_matrix.to(torch.int64)


def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.

    Args:
        list_with_duplicates: list that possibly contains duplicates

    Returns:
        A list with no duplicates.
    """

    unique_set = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list


def load_seqs_from_list(seqs_list, rm_duplicates=True, max_len=100):
    """
    Given a list of Sequence strings, provides a zero padded NumPy array
    with their index representation. Sequences longer than `max_len` are
    discarded. The final array will have dimension (all_valid_seqs, max_len+2)
    as a beginning and end of sequence tokens are added to each string.

    Args:
        seqs_list: a list of Sequence strings
        rm_duplicates: bool if True return remove duplicates from final output. Note that if True the length of the
          output does not equal the size of the input  `seqs_list`. Default True
        max_len: dimension 1 of returned array, sequences will be padded

    Returns:
        sequences: list a numpy array of Sequence character indices
    """
    sd = AASeqDictionary()

    # filter valid seqs strings
    valid_seqs = []
    valid_mask = [False] * len(seqs_list)
    for i, s in enumerate(seqs_list):
        s = s.strip()
        if len(s) <= max_len:
            valid_seqs.append(s)
            valid_mask[i] = True

    if rm_duplicates:
        unique_seqs = remove_duplicates(valid_seqs)
    else:
        unique_seqs = valid_seqs

    # max len + two chars for start token 'Q' and stop token '\n'
    max_seq_len = max_len + 2
    num_seqs = len(unique_seqs)
    logger.info(f'Number of sequences: {num_seqs}, max length: {max_len}')

    # allocate the zero matrix to be filled
    sequences = np.zeros((num_seqs, max_seq_len), dtype=np.int32)

    for i, seq in enumerate(unique_seqs):
        enc_seq = sd.BEGIN + seq + sd.END
        for c in range(len(enc_seq)):
            try:
                sequences[i, c] = sd.char_idx[enc_seq[c]]
            except KeyError as e:
                logger.info(f'KeyError: {seq}, key: {i}, {enc_seq[c]}')

    return sequences, valid_mask


def get_tensor_dataset(numpy_array):
    """
    Gets a numpy array of indices, convert it into a Torch tensor,
    divided it into inputs and targets and wrap it into a TensorDataset

    Args:
        numpy_array: to be converted

    Returns: a TensorDataset
    """

    tensor = torch.from_numpy(numpy_array).long()

    inp = tensor[:, :-1]
    target = tensor[:, 1:]

    return TensorDataset(inp, target)


def rnn_start_token_vector(batch_size, device='cpu'):
    """
    Returns a vector of start tokens. This vector can be used to start sampling a batch of Sequence strings.

    Args:
        batch_size: how many Sequence will be generated at the same time
        device: cpu | cuda

    Returns:
        a tensor (batch_size x 1) containing the start token
    """
    sd = AASeqDictionary()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)


