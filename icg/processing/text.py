import string

from collections import OrderedDict


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


def frequency(descriptions):
    freq = {}
    total_words = 0
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            for word in desc.split(" "):
                word = word.lower()
                if word in freq:
                    freq[word] = freq[word] + 1
                    total_words += 1
                else:
                    freq[word] = 1
                    total_words += 1
    frequ = {k: v / int(total_words) for k, v in freq.items()}
    return frequ, total_words


def clean_descriptions(descriptions):
        # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)

def select_nums_words(freq, num_words):
    order_freq = dict(OrderedDict(sorted(freq[0].items(), key=lambda x: x[1])))
    frec_asc = OrderedDict(sorted(order_freq.items(), key=lambda kv: kv[1], reverse=True)) 
    dic_frec_asc = dict(frec_asc)
    list_key = list(dic_frec_asc.keys())
    new_list_key = list_key[:num_words]
    new_dic = dict((key , dic_frec_asc[key]) for key in new_list_key)
    return new_dic.keys()

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()