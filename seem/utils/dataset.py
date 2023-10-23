
class Entity(object):
    def __init__(self, _id, _text, _mask, _interactive, _type, _start_idx, _end_idx, _image=None):
        self.id = _id
        self.text = _text
        self.mask = _mask
        self.interactive = _interactive
        self.type = _type
        self.start_idx = _start_idx
        self.end_idx = _end_idx

        self.image = _image

def split_by_ordered_substrings(sentence, substrings):
    results = []
    substring_indices = []

    start_index = 0
    for i, substring in enumerate(substrings):
        # Find the start of the substring in the remaining part of the sentence
        index = sentence[start_index:].find(substring)

        if index == -1:
            continue

        # Append any text before the substring to the results, including spaces
        if index > 0:
            results.append(sentence[start_index:start_index+index])
            substring_indices.append(None)  # No match in the `substrings` list for this segment
        
        # Append the substring to the results
        results.append(substring)
        substring_indices.append(i)  # Append the index from the `substrings` list
        start_index += index + len(substring)

    # If there's any remaining part of the sentence after all substrings, append it to the results
    if start_index < len(sentence):
        results.append(sentence[start_index:])
        substring_indices.append(None)  # No match in the `substrings` list for this segment

    return results, substring_indices
