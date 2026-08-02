"""Custom DataLoader — batching from list-of-dict rows + dummy dataset generator."""
import math
import random


class CustomDataLoader:
    """
    Load translation rows and yield batches of source and target text.
    """
    def __init__(self, rows, batch_size, shuffle=True):
        self.rows = rows
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """
        Return the total number of batches.
        """
        batches = math.ceil(len(self.rows)/self.batch_size)
        return batches

    def __iter__(self):
        """
        Yield each batch as source_text and target_text lists.
        """
        indices = list(range(len(self.rows)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size ):
            batch_chunk = indices[i:i + self.batch_size]
            source = [self.rows[idx]["source_text"] for idx in batch_chunk]
            target = [self.rows[idx]["target_text"] for idx in batch_chunk]
            yield {"source_text": source, "target_text": target}


def create_dummy_dataset(num_samples=500):
    en = ["hello", "how are you", "thank you", "good morning", "good night",
          "what is your name", "my name is", "i love you", "i am sorry",
          "please", "you are welcome", "excuse me", "how much", "i am hungry",
          "where is the bathroom", "can you help me", "i do not understand",
          "the cat is on the table", "the weather is nice", "i like to read"]
    fr = ["bonjour", "comment allez-vous", "merci", "bonjour", "bonne nuit",
          "quel est votre nom", "je m'appelle", "je t'aime", "je suis desole",
          "s'il vous plait", "de rien", "excusez-moi", "combien", "j'ai faim",
          "ou sont les toilettes", "pouvez-vous m'aider", "je ne comprends pas",
          "le chat est sur la table", "il fait beau", "j'aime lire"]

    rows = []
    for _ in range(num_samples):
        i = random.randint(0, len(en) - 1)
        j = random.randint(0, len(fr) - 1)
        rows.append({"source_text": en[i], "target_text": fr[j]})
    return rows


if __name__ == "__main__":
    rows = create_dummy_dataset(20)
    loader = CustomDataLoader(rows, batch_size=5, shuffle=True)
    print(f"Batches: {len(loader)}")
    for batch in loader:
        print(f"  src: {batch['source_text']}")
        print(f"  tgt: {batch['target_text']}")
        print("---")