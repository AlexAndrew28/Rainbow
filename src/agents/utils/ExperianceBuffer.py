import collections
import random
from typing import Any, List

class BasicExperienceBuffer:
    def __init__(self, size: int):
        """ Initialises a buffer that stores a number of items upto a max amount 

        Args:
            size (int): The maximum number of items to store 
        """
        self.buffer = collections.deque(maxlen=size)
        self.max_len = size
    
    def getLength(self) -> int:
        """ Gets the number of items stored in the buffer

        Returns:
            int: The number of items in the buffer
        """
        return len(self.buffer)
    
    def append(self, exp: Any):
        """ Adds an item into the buffer

        Args:
            exp (Any): Item to add into the buffer
        """
        self.buffer.append(exp)

    def getSample(self, amount: int) -> List[Any]:
        """ Gets a random sample of item in the buffer

        Args:
            amount (int): The amount of items in the sample

        Returns:
            List[Any]: The random sample
        """
        transitions = random.sample(self.buffer, k=amount)

        return transitions