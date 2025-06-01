#src/utils.py

import re
import string
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

slang_dict = {
    # Bahasa Indonesia slang dan informal umum
    'gk': 'tidak', 'nggak': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'tdk': 'tidak',
    'btw': 'omong-omong', 'bgt': 'banget', 'bngt': 'banget', 'skrg': 'sekarang',
    'jd': 'jadi', 'kpn': 'kapan', 'dpt': 'dapat', 'kmrn': 'kemarin', 'krn': 'karena',
    'msh': 'masih', 'aja': 'saja', 'jgn': 'jangan', 'trs': 'terus', 'trus': 'terus',
    'plg': 'pulang', 'dlm': 'dalam', 'lg': 'lagi', 'jg': 'juga', 'cpt': 'cepat',
    'lamaa': 'lama', 'seru': 'menyenangkan', 'bagusss': 'bagus', 'oke': 'baik',
    'yuk': 'ayo', 'cuy': 'bro', 'bro': 'teman', 'sob': 'teman', 'mantap': 'bagus',
    'keren': 'hebat', 'cakep': 'cantik', 'gemes': 'lucu', 'mager': 'malas gerak',
    'nongkrong': 'berkumpul', 'nongkrongin': 'mengunjungi', 'nyesel': 'menyesal',
    'capek': 'lelah', 'malas': 'enggan', 'ngangenin': 'mengingatkan', 'bosen': 'bosan',
    'nyaman': 'enak', 'murah meriah': 'murah dan bagus', 'hits': 'populer',
    'fasilitas': 'sarana', 'bersih banget': 'sangat bersih', 'ramah': 'baik hati',
    'tiket masuk': 'harga masuk', 'jalan-jalan': 'berkeliling', 'kulineran': 'mencicipi makanan',
    'liburan': 'berlibur', 'wisata': 'tempat wisata', 'pantai': 'tempat pantai',
    'gunung': 'tempat gunung', 'hutan': 'tempat hutan', 'air terjun': 'curug',
    'sunset': 'matahari terbenam', 'sunrise': 'matahari terbit', 'nyantai': 'bersantai',
    'pemandangan': 'view', 'spot foto': 'tempat foto', 'hits': 'populer',

    # Bahasa Inggris umum + singkatan + istilah travel
    'lol': 'tertawa', 'omg': 'astaga', 'btw': 'omong-omong', 'idk': 'saya tidak tahu',
    'tbh': 'jujur', 'fyi': 'untuk informasi', 'asap': 'secepatnya',
    'plz': 'tolong', 'thx': 'terima kasih', 'u': 'kamu', 'ur': 'mu',
    'imho': 'menurut saya', 'np': 'tidak masalah', 'ikr': 'saya tahu, benar',
    'tmi': 'terlalu banyak informasi', 'afaik': 'sejauh yang saya tahu',
    'spot': 'tempat', 'pic': 'foto', 'pics': 'foto-foto', 'view': 'pemandangan',
    'scenery': 'pemandangan', 'crowded': 'ramai', 'quiet': 'tenang',
    'cozy': 'nyaman', 'friendly': 'ramah', 'vibes': 'suasana', 'chill': 'santai',
    'hangout': 'berkumpul', 'trip': 'perjalanan', 'staycation': 'liburan lokal',
    'backpacker': 'pelancong hemat', 'touristy': 'banyak turis', 'hidden gem': 'tempat tersembunyi yang menarik',
    'instagrammable': 'bagus untuk foto', 'must visit': 'harus dikunjungi',
    'off the beaten path': 'tempat yang jarang dikunjungi',

    # Kata informal, typo, dan singkatan umum
    'gmna': 'bagaimana', 'gimana': 'bagaimana', 'gmn': 'bagaimana',
    'bgtu': 'begitu', 'yg': 'yang', 'nya': 'itu', 'udh': 'sudah', 'td': 'tadi',
    'krg': 'kurang', 'byk': 'banyak', 'klo': 'kalau', 'dr': 'dari',
    'klau': 'kalau', 'jln': 'jalan', 'sm': 'sama', 'dgn': 'dengan',
    'trnyata': 'ternyata', 'ny': 'nya', 'sgt': 'sangat', 'dah': 'sudah',
    'bener': 'benar', 'knp': 'kenapa', 'msh': 'masih', 'krn': 'karena',

    # Ekspresi dan kata yang menunjukkan kesan positif/negatif informal
    'mantul': 'mantap betul', 'kece': 'keren', 'ngab': 'nggak', 'bosen': 'bosan',
    'capek': 'lelah', 'malas': 'enggan', 'nyesel': 'menyesal', 'ngangenin': 'mengingatkan',
    'asik': 'asyik', 'enak banget': 'sangat enak', 'pewe': 'percaya wae (santai)',
    'ngecewain': 'mengecewakan', 'rempong': 'ribet', 'receh': 'murahan', 'lelet': 'lambat',
    'cepett': 'cepat', 'cpt': 'cepat', 'lumayan': 'cukup', 'macet': 'padat',

    # Kata gaul dan istilah umum dalam wisata/kuliner
    'ngopi': 'minum kopi', 'nongkrong': 'berkumpul', 'makan-makan': 'bermakan',
    'jajan': 'membeli makanan ringan', 'kuliner': 'makanan', 'foodcourt': 'tempat makan',
    'wartel': 'warung telepon (jadul)', 'staycation': 'liburan di tempat sendiri',
    'jalan-jalan': 'berkeliling', 'selfie': 'foto diri', 'hangout': 'berkumpul',
}

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#Membuat fungsi untuk menghapus emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002700-\U000027BF"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#Membuat fungsi untuk ganti kata slang
def replace_slang(tokens, slang_dict):
    return [slang_dict.get(word, word) for word in tokens]

#Memfilter stopwords
def filtering_text(tokens):
  stop_words = set(stopwords.words('indonesian'))
  stop_words2 = set(stopwords.words('english'))
  stop_words.update(stop_words2)

  #Tambahan stopwords
  additional_stopwords = list(set([
    "iya", "ya", "yaa", "yoo", "yo", "woi", "woii", "woy", "wew", "loh", "dong", "deh", "nih", "sih", "kok", "nah", "ah", "lah", "lho", "yg", "ke", "pd",
    "ga", "gak", "gk", "gaaa", "gaa", "nggak", "ngga", "tdk", "enggak", "blm", "udah", "sdh", "dr", "dpt", "tp", "trs", "jg", "aja", "doang", "banget", "bgt", "bngt",
    "nya", "ku", "mu", "kau", "gue", "gua", "elo", "loe", "lu", "dia", "kami", "kita", "mereka", "kan", "cok", "cuk", "ges", "geys",
    "mau", "lagi", "dulu", "saja", "hanya", "cuma", "pun", "lah", "kah", "tak", "biar", "supaya", "agar",
    "apa", "siapa", "kenapa", "mengapa", "kapan", "dimana", "gimana", "bagaimana",
    "apaan", "nih", "tuh", "udah", "ye", "yes", "no", "ha", "he", "eh", "hmm", "hahaha", "haha", "hehe", "wkwk", "wkwkwk", "nyaaa",
    "nyaa",
    ]))
  
  additional_stopwords.sort()

  stop_words.update(additional_stopwords)

  filtered = [word for word in tokens if word not in stop_words]

  return filtered

#Fungsi stemming
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

#Membuat fungsi untuk preprocess text
def preprocess_text(text, slang_dict):
  if not isinstance(text, str):
        return ""

  text = text.lower() #Konvert teks ke lower case
  text = ''.join(c for c in text if c.isprintable())
  text = re.sub(r'[\u2066\u2067\u2068\u2069]', '', text)
  text = remove_emoji(text) #Menghapus emoji
  text = re.sub(r"http\S+|www\S+", '', text) #Menghilangkan URL
  text = re.sub(r'#[A-Za-z0-9_]+|@[A-Za-z0-9_]+', '', text) #Menghilangkan hashtag dan mention
  text = text.replace('\n', ' ') #Mengganti newline ke spasi
  text = re.sub(r'\d+', '', text)
  text = text.translate(str.maketrans('', '', string.punctuation))
  # Normalisasi spasi ganda dan strip berurutan
  text = re.sub(r'[-]{2,}', ' ', text)  # Ganti strip beruntun "--" jadi spasi
  text = re.sub(r'\s{2,}', ' ', text).strip()  # Ganti spasi ganda jadi satu
  tokens = word_tokenize(text) #Tokenisasi
  tokens = replace_slang(tokens, slang_dict) #Mengganti slang
  tokens = filtering_text(tokens) #Filtering stopwords
  tokens = stemming(tokens)
  return " ".join(tokens)