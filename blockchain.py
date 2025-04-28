import hashlib
import time

# Blockchain Class Definition
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block in the blockchain (genesis block)"""
        genesis_block = self.create_new_block('GENESIS', 0, 'GENESIS')
        self.chain.append(genesis_block)

    def create_new_block(self, predicted_label, confidence, prev_hash=''):
        """
        Function to create a new block and append it to the blockchain.
        The block stores the prediction label and confidence score.
        """
        block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'prev_hash': prev_hash,
            'hash': ''
        }

        block['hash'] = self.calculate_hash(block)
        self.chain.append(block)
        return block

    def calculate_hash(self, block):
        """
        Function to calculate the hash of a block. We use the SHA-256 algorithm.
        """
        block_string = str(block['index']) + str(block['timestamp']) + block['predicted_label'] + str(block['confidence']) + block['prev_hash']
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def get_last_block(self):
        """
        Function to get the last block in the chain.
        """
        return self.chain[-1] if len(self.chain) > 0 else None
