from collections import OrderedDict

class LRUCache:
    """A simple LRU cache."""
    
    def __init__(self, capacity):
        """Initializes an LRU cache.

        Args:
            capacity : The number of entries the LRU cache holds.
        """
        
        self.cache = OrderedDict()
        self.capacity = capacity
        
    def get(self, key):
        """Gets value tied to the key. If the key exists the tied value is moved to the front.

        Args:
            key : The key.

        Returns:
            The tied value if it exists, otherwise None.
        """
        
        if key not in self.cache:
            return None
        
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key, value):
        """Puts the value into the cache under the given key. Key-value is moved to the front.

        Args:
            key : The key.
            value : The value.
        """
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(False)