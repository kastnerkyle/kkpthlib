from .core import get_logger
import numpy as np

class ListIterator(object):
    # iterate group of sequences
    # no_random means to directly use the order of the input sequences
    # single_shuffle generates random minibatch ordering one time, repeats it
    # infinite_iterator means it will never raise StopIteration
    # random_state
    def __init__(self, list_of_sequences, batch_size,
                 no_random=False,
                 single_shuffle=False,
                 infinite_iterator=False,
                 truncate_if_ragged_last_batch=True,
                 random_state=None):
        if random_state is None:
            if no_random is False: raise ValueError("Must pass random state to use random iteration! Otherwise, set no_random=True")
            self.random_state = random_state
        self.list_of_sequences = list_of_sequences
        self.logger = get_logger()

        base_len = len(list_of_sequences[0])
        if len(list_of_sequences) > 0:
            for i in range(len(list_of_sequences)):
                if len(list_of_sequences[i]) != base_len:
                    raise ValueError("Sequence lengths for iteration do not match! Check element {} and {} of list_of_sequences".format(0, i))

        self.base_len = base_len
        self.is_ragged = False
        if (self.base_len % batch_size) != 0:
           self.is_ragged = True
           if not truncate_if_ragged_last_batch:
                self.logger.info("WARNING: batch_size for ListIterator is not evenly divisible, providing uneven last batch due to truncate_if_ragged_last_batch=False")

        self.no_random = no_random
        self.batch_size = batch_size
        self.single_shuffle = single_shuffle
        self.infinite_iterator = infinite_iterator
        self.truncate_if_ragged_last_batch = truncate_if_ragged_last_batch
        self.random_state = random_state

        batch_indices = np.arange(base_len)
        if self.no_random is False:
            self.random_state.shuffle(batch_indices)
        batches = [batch_indices[i:i + batch_size] for i in range(0, len(batch_indices), batch_size)]
        if self.is_ragged:
            if self.truncate_if_ragged_last_batch:
                batches = batches[:-1]

        self.current_batches_ = batches
        self.batches_index_ = 0

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        if self.batches_index_ >= len(self.current_batches_):
            # reset and raise StopIteration
            self.batches_index_ = 0
            if self.single_shuffle is False:
                batch_indices = np.arange(self.base_len)
                if self.no_random is False:
                    self.random_state.shuffle(batch_indices)
                batches = [batch_indices[i:i + self.batch_size] for i in range(0, len(batch_indices), self.batch_size)]
                if self.is_ragged:
                    if self.truncate_if_ragged_last_batch:
                        batches = batches[:-1]
                self.current_batches_ = batches
            raise StopIteration("End of sequence")
        else:
            i = self.current_batches_[self.batches_index_]
            this_batch = [np.array([ls[_ii] for _ii in i]) for ls in self.list_of_sequences]
            self.batches_index_ += 1
        return this_batch


class StepIterator(object):
    # iterate group of sequences
    # no_random means to directly use the order of the input sequences
    # single_shuffle generates random minibatch ordering one time, repeats it
    # infinite_iterator means it will never raise StopIteration
    # random_state
    def __init__(self, list_of_sequences, slice_size=1,
                 step_size=1,
                 random_shuffle=False,
                 circular_rotation=False,
                 reorder_once=False,
                 infinite_iterator=False,
                 truncate_if_ragged_last_batch=True,
                 random_state=None):
        if random_state is None:
            raise ValueError("Must pass random state to StepIterator!")
        self.random_state = random_state
        self.list_of_sequences = list_of_sequences
        self.logger = get_logger()

        base_len = len(list_of_sequences[0])
        if len(list_of_sequences) > 0:
            for i in range(len(list_of_sequences)):
                if len(list_of_sequences[i]) != base_len:
                    raise ValueError("Sequence lengths for iteration do not match! Check element {} and {} of list_of_sequences".format(0, i))

        self.base_len = base_len
        self.is_ragged = False
        if (self.base_len % step_size) != 0:
           self.is_ragged = True
           if not truncate_if_ragged_last_batch:
                self.logger.info("WARNING: step_size for OrderedIterator is not evenly divisible, providing uneven last batch due to truncate_if_ragged_last_batch=False")

        self.slice_size = slice_size
        self.step_size = step_size
        self.random_shuffle = random_shuffle
        self.circular_rotation = circular_rotation
        self.reorder_once = reorder_once
        self.infinite_iterator = infinite_iterator
        self.truncate_if_ragged_last_batch = truncate_if_ragged_last_batch

        batch_indices = np.arange(base_len)
        if self.random_shuffle is True:
            self.random_state.shuffle(batch_indices)
        if self.circular_rotation is True:
            rotate_point = random_state.randint(base_len)

        if self.is_ragged:
            if self.truncate_if_ragged_last_batch:
                batch_indices = batch_indices[:len(batch_indices) - len(batch_indices) % min(1, self.slice_size - self.step_size) + self.step_size]
        self.index_ = 0
        self.batch_indices = batch_indices

    def next(self):
        return self.__next__()

    def __iter__(self):
        while True:
            yield next(self)

    def __next__(self):
        if self.index_ >= self.base_len:
            # reset and raise StopIteration
            self.index_ = 0
            if self.reorder_once is False:
                batch_indices = np.arange(self.base_len)
                if self.random_shuffle is True:
                    self.random_state.shuffle(batch_indices)

                if self.circular_rotation is True:
                    rotate_point = self.random_state.randint(self.base_len)

                if self.is_ragged:
                    if self.truncate_if_ragged_last_batch:
                        batch_indices = batch_indices[:len(batch_indices) - len(batch_indices) % min(1, self.slice_size - self.step_size) + self.step_size]
                self.batch_indices = batch_indices
            raise StopIteration("End of sequence")
        else:
            i = self.index_
            this_batch = [ls[i:i + self.slice_size] for ls in self.list_of_sequences]
            if self.slice_size == 1:
                this_batch = [t[0] for t in this_batch]
            self.index_ += self.step_size
            if len(self.list_of_sequences) == 1:
                 return this_batch[0]
        return this_batch


