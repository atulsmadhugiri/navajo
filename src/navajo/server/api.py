import backoff
import concurrent.futures
import hashlib
import json
import logging
import openai
import os
import requests
import tiktoken
import traceback

from . import logconf
from . import database
from .vector import aggregate_embeddings, VectorRunningAverage

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

def _load_api_key():
    config_dir = os.path.expanduser('~/.config/navajo')
    settings_file = os.path.join(config_dir, 'settings.json')
    os.makedirs(config_dir, exist_ok=True)

    if os.path.isfile(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
    else:
        settings = {}

    api_key = settings.get('openai_api_key')

    if not api_key:
        api_key = input('Please enter your OpenAI API key: ')
        settings['openai_api_key'] = api_key

        with open(settings_file, 'w') as f:
            json.dump(settings, f)

    return api_key

openai.api_key = _load_api_key()
TOTAL_EMBED_CALLS = 0
ENGINE_NAME = os.environ.get('NAVAJO_ENGINE_NAME', 'gpt-4')
EMBED_LEN = 1536
MAX_EMBED_TOKENS = 8191
MAX_CHAT_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}[ENGINE_NAME]

database.init_embedding_db()

@lru_cache(maxsize=512)
def count_tokens(data):
    if isinstance(data, list):
        data = ' '.join(data).strip()
    encoding = tiktoken.encoding_for_model(ENGINE_NAME)
    num_tokens = len(encoding.encode(data))

    return int(num_tokens * 1.052)

def split_text(text, max_tokens):
    words = text.split()
    result = [words[0]]

    for word in words[1:]: # TODO handle very long first word
        extended = result[-1] + ' ' + word
        if count_tokens(extended) > max_tokens:
            result.append(word)
        else:
            result[-1] += ' ' + word

    return result

def open_stream(messages, args):
    headers = {'Accept': 'text/event-stream', 'Authorization': 'Bearer ' + openai.api_key}
    data = {
        'model': ENGINE_NAME,
        'messages': messages,
        'stream': True,
        **args,
    }

    url = 'https://api.openai.com/v1/chat/completions'
    response = requests.post(url, headers=headers, json=data, stream=True)
    if response.status_code != 200:
        logging.info(f'Received status {response.status_code} from OpenAI: {response.text}')
        raise openai.error.InvalidRequestError('Issue reaching the OpenAI API', response.status_code)

    return response

def send_messages(messages, args):
    response = openai.ChatCompletion.create(
        model=ENGINE_NAME,
        messages=messages,
        **args,
    )

    return response.choices[0]['message']['content'].strip()

def send_in_chunks(text, template, args, max_tokens=None):
    prompt_len = sum(count_tokens(x['content']) for x in template)
    if not max_tokens:
        max_tokens = 4096 - prompt_len - 50
    texts = split_text(text, max_tokens)

    def _append_text(s):
        t = deepcopy(template)
        t[-1]['content'] += s
        return t

    messages = [_append_text(t) for t in texts]
    full_response = ""
    for m in messages:
        try:
            response = openai.ChatCompletion.create(
                model=ENGINE_NAME,
                messages=m,
                **args,
            )

        except openai.error.InvalidRequestError:
            return send_in_chunks(text, template, args,
                max_tokens=max_tokens-100)

        sub = response.choices[0]['message']['content'].strip()
        full_response += sub + " "

    return full_response.strip()

@lru_cache(maxsize=512)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def _embedding_call(text):
    global TOTAL_EMBED_CALLS
    TOTAL_EMBED_CALLS += 1

    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002",
    )

    return response['data'][0]['embedding']

def _get_cached_embedding(text):
    t_hash = hashlib.sha256(text.encode()).hexdigest()
    conn = database.get_connection('embeddings')
    c = conn.cursor()
    c.execute('SELECT embedding FROM embeddings WHERE hash=?', (t_hash,))
    result = c.fetchone()
    database.close_connection('embeddings', conn)

    if result:
        logging.info('embedding cache hit %s', t_hash)
        return json.loads(result[0])
    else:
        return None

def get_embedding(text):
    cached = _get_cached_embedding(text)
    if cached:
        return cached

    texts = split_text(text, MAX_EMBED_TOKENS)
    embeddings = []

    for t in texts:
        e = _embedding_call(t)
        embeddings.append(e)

    aggregated_embedding = aggregate_embeddings(embeddings)

    # cache the aggregated embedding for the entire text
    t_hash = hashlib.sha256(text.encode()).hexdigest()
    conn = database.get_connection('embeddings')
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO embeddings (hash, embedding) VALUES (?, ?)',
          (t_hash, json.dumps(aggregated_embedding)))
    conn.commit()
    database.close_connection('embeddings', conn)

    return aggregated_embedding

def _aggregate_chunks(chunks):
    embeddings = defaultdict(lambda: VectorRunningAverage(size=EMBED_LEN))
    for i, chunk in chunks:
        embeddings[i].send(chunk)

    orig_order = sorted(embeddings.items(), key=lambda p: p[0])
    return [e.normalize() for _, e in orig_order]

def embed_all(texts, threads=15):
    chunks = []
    for i, text in enumerate(texts):
        split = split_text(text, MAX_EMBED_TOKENS)
        for s in split:
            chunks.append((i, s))

    def _embed_chunk(pair):
        index, chunk = pair
        return (index, _embedding_call(chunk))

    threads = min(threads, len(texts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(_embed_chunk, chunks))

    embeddings = _aggregate_chunks(results)
    return embeddings

def log_call_count():
    logging.info(f"Total embedding API calls: {TOTAL_EMBED_CALLS}")
