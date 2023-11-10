from typing import Any
from segment_tree import SegmentTree
import numpy as np

class SimplePriorityExperienceBuffer:
    def __init__(self, max_len: int, eps: float=0.01, alpha: float=0.6, alpha_decay: float=0.99):
        """ Initialises a buffer where each item has a priority that affects how likely it is to be selected in a 
        random sample from the buffer

        Args:
            max_len (int): The max capacity of the buffer
            eps (float, optional): A small constant to ensure the priority does not fall to 0 for any item. Defaults to 0.01.
            alpha (float, optional): The strength that prioritisation plays on sampling. Defaults to 0.6.
            alpha_decay (float, optional): The amount that the prioritisation strength decays over time. Defaults to 0.99.
        """
        # initialise everything required for the buffer
        self.buffer = np.array([None for _ in range(max_len)], dtype="object")
        self.priorities = np.array([0 for _ in range(max_len)], dtype=float)
        self.max_len = max_len
        # uses a segment tree
        self.segment_tree = SegmentTree([0 for _ in range(max_len)])
        self.update_pointer = 0
        self.current_filled = 0
        self.max_prio = 1
        self.eps = eps
        self.alpha = alpha
        self.alpha_decay = alpha_decay
    
    def get_length(self) -> int:
        """ Gets the amount of items stored in the buffer

        Returns:
            int: Number of items stored in the buffer
        """
        return self.current_filled
    
    def append(self, exp: Any):
        """ Adds an item in to the buffer while giving it the max priority 

        Args:
            exp (Any): The item to add
        """
        # sets incoming priority to the max value
        prio = self.max_prio ** self.alpha

        index = self.update_pointer
    
        # adds the experience into the buffer
        self.buffer[index] = exp

        # updates the priority
        self.priorities[index] = prio
        self.segment_tree.update(index, prio)

        if self.current_filled < self.max_len:
            self.current_filled += 1

        self.update_pointer += 1
        if self.update_pointer == self.max_len:
            self.update_pointer = 0


    def get_sample(self, amount: int) -> [np.ndarray[int], np.ndarray[Any], np.ndarray[float]]:
        """ Gets a sample from the buffer according to the priorities 

        Args:
            amount (int): The amount of items to select for the sample

        Returns:
            [np.ndarray[int], np.ndarray[Any], np.ndarray[float]]: The indexs of the sample, the samples themselves and the priorities of the samples
        """
        # generates the normalized weights for each of the experiences 
        weights = self.priorities / self.segment_tree.query(0, self.max_len-1, "sum")

        # generates the sample
        sampled_transitions = np.random.choice(self.max_len, size=amount, p=weights, replace=False)

        transition_priorities = self.priorities[sampled_transitions]
        samples = self.buffer[sampled_transitions]

        return sampled_transitions, samples, transition_priorities

    def update_priorities(self, indx: np.ndarray[int], new_priorities: np.ndarray[float]):
        """ Updates the priorities for a set of items in the buffer

        Args:
            indx (np.ndarray[int]): The indexs of the item to update the priority of
            new_priorities (np.ndarray[float]): The priorities of the items
        """
        # decays alpha 
        self.alpha = max(0.05, self.alpha * self.alpha_decay)

        # updates the priorities of the indexes to the new values coming in
        for i, p in zip(indx, new_priorities):

            # scales p accorsing to the value of alpha whilst ensuring it is not 0
            p = min(p.item(), self.max_prio)
            p = (p + self.eps) ** self.alpha

            self.priorities[i] = p
            self.segment_tree.update(i, p)
