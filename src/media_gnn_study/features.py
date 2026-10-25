
import os, sys, json, hashlib, re, time, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.ensemble import RandomForestClassifier
try:
    from IPython.display import display
except Exception:
    def display(x): print(x)

from .config import *
from .contracts import *

def safe_unit_rows(x, eps=1e-10):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Bloco de features inválido.')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    out = x.copy()
    mask = norms[:, 0] >= eps
    out[mask] /= norms[mask]
    return out


def encode_documents(texts, encoder, batch_size=32):
    tok = encoder.tokenizer
    if not tok.is_fast:
        raise ValueError('É necessário tokenizer rápido.')
    limit = int(encoder.max_seq_length)
    counts = np.array([
        len(tok(t, add_special_tokens=False, truncation=False, verbose=False)['input_ids'])
        for t in texts
    ], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError('Sinopse sem tokens.')

    packed = tok(
        list(texts), add_special_tokens=True, truncation=True, max_length=limit,
        stride=0, return_overflowing_tokens=True, return_special_tokens_mask=True,
        return_attention_mask=True, padding=True, return_tensors='pt'
    )
    owner = np.asarray(packed.pop('overflow_to_sample_mapping'), dtype=np.int64)
    special_mask = packed.pop('special_tokens_mask')
    weights = (packed['attention_mask'] * (1 - special_mask)).sum(1).numpy().astype(np.int64)
    covered = np.bincount(owner, weights=weights, minlength=len(texts)).astype(np.int64)
    assert np.array_equal(covered, counts), 'Token perdido ou duplicado.'

    encoder.eval(); encoder.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(owner), batch_size):
            features = {
                k: v[start:start+batch_size].to(encoder.device)
                for k, v in packed.items()
                if k in ('input_ids','attention_mask','token_type_ids')
            }
            z = encoder(features)['sentence_embedding']
            chunks.append(F.normalize(z.float(), p=2, dim=1).cpu().numpy())
    chunks = np.concatenate(chunks)
    x = np.zeros((len(texts), chunks.shape[1]), dtype=np.float32)
    np.add.at(x, owner, chunks * weights[:, None])
    x /= counts[:, None]
    return safe_unit_rows(x)


def get_minilm_features(dev):
    cache_path = CACHE/'minilm_embeddings.npz'
    legacy = ROOT/'03_resultados'/'02_mvp_mlp_vs_gnn'/REFERENCE_MINILM_RUN/'embeddings_dev.npz'
    for source in (cache_path, legacy):
        if source.is_file():
            with np.load(source, allow_pickle=False) as z:
                ids = z['ids'].astype(str).tolist(); x = z['X'].astype(np.float32)
            if ids == dev.Id.tolist() and x.shape == (len(dev), 384):
                x = safe_unit_rows(x)
                if source != cache_path:
                    np.savez_compressed(cache_path, ids=dev.Id.to_numpy(dtype=str), X=x)
                return x, {'cache_hit': True, 'source': str(source), 'model': MODEL_ID,
                           'revision': MODEL_REVISION, 'rule': EMBED_RULE}

    _ensure_package('sentence_transformers', 'sentence-transformers>=3.4,<6')
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device='cpu', trust_remote_code=False)
    x = encode_documents(dev.texto.tolist(), encoder)
    np.savez_compressed(cache_path, ids=dev.Id.to_numpy(dtype=str), X=x)
    return x, {'cache_hit': False, 'source': 'computed', 'model': MODEL_ID,
               'revision': MODEL_REVISION, 'rule': EMBED_RULE, 'device': 'cpu'}


def _resolve_hf_revision(model_id):
    try:
        _ensure_package('huggingface_hub', 'huggingface_hub>=0.25')
        from huggingface_hub import model_info
        return model_info(model_id).sha
    except Exception:
        return 'main'


def _encode_emotion_chunks(texts, tokenizer, model, batch_size=16):
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Tokenizer rápido necessário para agregação de chunks de emoção.')
    max_len = int(getattr(tokenizer, 'model_max_length', 512))
    if max_len > 4096 or max_len < 16:
        max_len = 512
    packed = tokenizer(
        list(texts), add_special_tokens=True, truncation=True, max_length=max_len,
        stride=0, return_overflowing_tokens=True, return_special_tokens_mask=True,
        return_attention_mask=True, padding=True, return_tensors='pt'
    )
    owner = np.asarray(packed.pop('overflow_to_sample_mapping'), dtype=np.int64)
    special = packed.pop('special_tokens_mask')
    weights = (packed['attention_mask'] * (1-special)).sum(1).numpy().astype(np.float32)
    probs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(owner), batch_size):
            batch = {k: v[start:start+batch_size].to(DEVICE) for k,v in packed.items()
                     if k in ('input_ids','attention_mask','token_type_ids')}
            probs.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(probs).astype(np.float32)
    x = np.zeros((len(texts), probs.shape[1]), dtype=np.float32)
    den = np.zeros(len(texts), dtype=np.float32)
    np.add.at(x, owner, probs * weights[:, None]); np.add.at(den, owner, weights)
    x /= np.maximum(den[:, None], 1e-8)
    return x


def get_emotion_features(dev):
    cache = CACHE/'emocao_xlm_emo_t_v1.npz'
    meta_path = CACHE/'emocao_xlm_emo_t_v1.json'
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            ids = z['ids'].astype(str).tolist(); x = z['X'].astype(np.float32); labels = z['labels'].astype(str).tolist()
        if ids == dev.Id.tolist() and len(x) == len(dev):
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
            return x, labels, {**meta, 'cache_hit': True, 'source': str(cache)}

    _ensure_package('transformers', 'transformers>=4.45,<6')
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    revision = _resolve_hf_revision(EMOTION_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID, revision=revision, use_fast=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID, revision=revision, trust_remote_code=False).to(DEVICE)
    x = _encode_emotion_chunks(dev.texto.tolist(), tokenizer, model)
    labels = [str(model.config.id2label[i]) for i in range(model.config.num_labels)]
    np.savez_compressed(cache, ids=dev.Id.to_numpy(dtype=str), X=x, labels=np.asarray(labels, dtype=str))
    meta = {
        'model': EMOTION_MODEL_ID, 'resolved_revision': revision, 'labels': labels,
        'dimensions': int(x.shape[1]),
        'aggregation': 'nonoverlapping_token_chunks_content_weighted_mean_softmax_v1',
        'domain_note': 'multilingual social-media emotion model used frozen on synopses',
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return x, labels, {**meta, 'cache_hit': False, 'source': 'computed', 'device': 'cpu'}


def _entity_key(text):
    text = clean_text(text).lower().strip()
    text = re.sub(r'[^\wÀ-ÿ]+', '_', text, flags=re.UNICODE).strip('_')
    return text or 'vazio'


def _safe_lemma(token):
    lemma = (token.lemma_ or token.text).lower().strip()
    return re.sub(r'[^\wÀ-ÿ]+', '_', lemma, flags=re.UNICODE).strip('_')


def get_spacy_text_views(dev):
    """Uma única etapa spaCy para NER, ações e features linguísticas; reutiliza caches legados quando possível."""
    cache = CACHE/f'spacy_text_views_{SPACY_MODEL_ID}_v2.npz'
    meta_path = CACHE/f'spacy_text_views_{SPACY_MODEL_ID}_v2.json'
    if cache.is_file() and meta_path.is_file():
        with np.load(cache, allow_pickle=False) as z:
            ids = z['ids'].astype(str).tolist()
            x_ent = z['X_entities'].astype(np.float32)
            entity_labels = z['entity_labels'].astype(str).tolist()
            action_docs = z['action_docs'].astype(str).tolist()
            x_ling = z['X_linguistic_raw'].astype(np.float32)
            ling_names = z['linguistic_feature_names'].astype(str).tolist()
        if ids == dev.Id.tolist():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            return x_ent, entity_labels, action_docs, x_ling, ling_names, {**meta, 'cache_hit': True, 'source': str(cache)}

    # Compatibilidade: os caches produzidos pela versão anterior contêm exatamente estas representações.
    old_ent = CACHE/f'entidades_{SPACY_MODEL_ID}_{ENTITY_HASH_DIM}d_v1.npz'
    old_ent_meta = CACHE/f'entidades_{SPACY_MODEL_ID}_{ENTITY_HASH_DIM}d_v1.json'
    old_narr = CACHE/f'narrativa_{SPACY_MODEL_ID}_v1.npz'
    old_narr_meta = CACHE/f'narrativa_{SPACY_MODEL_ID}_v1.json'
    if old_ent.is_file() and old_narr.is_file():
        with np.load(old_ent, allow_pickle=False) as ze, np.load(old_narr, allow_pickle=False) as zn:
            ids_e = ze['ids'].astype(str).tolist(); ids_n = zn['ids'].astype(str).tolist()
            if ids_e == dev.Id.tolist() == ids_n:
                x_ent = ze['X'].astype(np.float32)
                entity_labels = ze['labels'].astype(str).tolist()
                action_docs = zn['action_docs'].astype(str).tolist()
                x_ling = zn['X_linguistic_raw'].astype(np.float32)
                ling_names = zn['feature_names'].astype(str).tolist()
                np.savez_compressed(
                    cache, ids=dev.Id.to_numpy(dtype=str), X_entities=x_ent,
                    entity_labels=np.asarray(entity_labels, dtype=str), action_docs=np.asarray(action_docs, dtype=str),
                    X_linguistic_raw=x_ling, linguistic_feature_names=np.asarray(ling_names, dtype=str)
                )
                meta = {
                    'model': SPACY_MODEL_ID, 'data_sha256': DATA_SHA,
                    'rule': 'reused_legacy_entity_and_narrative_caches', 'reused_legacy_caches': True,
                    'entity_dimensions': int(x_ent.shape[1]), 'linguistic_dimensions': int(x_ling.shape[1]),
                }
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
                return x_ent, entity_labels, action_docs, x_ling, ling_names, {**meta, 'cache_hit': True, 'source': 'legacy_caches'}

    _ensure_package('spacy', 'spacy>=3.7,<4')
    import spacy
    try:
        nlp = spacy.load(SPACY_MODEL_ID)
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', SPACY_MODEL_ID], check=True)
        nlp = spacy.load(SPACY_MODEL_ID)

    entity_labels = sorted(nlp.get_pipe('ner').labels)
    label_to_i = {lab:i for i,lab in enumerate(entity_labels)}
    pos_names = ['VERB','AUX','NOUN','PROPN','ADJ','ADV','PRON']
    ling_names = [
        'log_tokens','log_sentences','avg_sentence_len','avg_alpha_token_len','lexical_diversity',
        *[f'prop_{p.lower()}' for p in pos_names],
        'negation_ratio','question_ratio','exclamation_ratio','punct_ratio','verb_object_pairs_per_token',
    ]

    entity_docs, type_props, ent_stats, action_docs, ling_rows = [], [], [], [], []
    texts = dev.texto.tolist()
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=32)):
        ents = list(doc.ents)
        counts = np.zeros(len(entity_labels), dtype=np.float32)
        ent_tokens, surfaces = [], []
        for ent in ents:
            key = _entity_key(ent.text)
            ent_tokens.append(f'{ent.label_}::{key}'); surfaces.append(key)
            if ent.label_ in label_to_i:
                counts[label_to_i[ent.label_]] += 1
        total_ent = len(ents)
        n_word = max(1, len(re.findall(r'\w+', text, flags=re.UNICODE)))
        entity_docs.append(' '.join(ent_tokens))
        type_props.append(counts / max(total_ent, 1))
        ent_stats.append([np.log1p(total_ent), total_ent/n_word, len(set(surfaces))/max(total_ent,1), float(total_ent==0)])

        alpha = [t for t in doc if t.is_alpha]
        n_alpha = max(1, len(alpha)); sentences = list(doc.sents); n_sent = max(1, len(sentences))
        pos_counts = {p:0 for p in pos_names}; lemmas = []
        for t in alpha:
            if t.pos_ in pos_counts: pos_counts[t.pos_] += 1
            lem = _safe_lemma(t)
            if lem: lemmas.append(lem)
        action_tokens, pair_count = [], 0
        for t in alpha:
            if t.pos_ != 'VERB': continue
            verb = _safe_lemma(t)
            if not verb: continue
            action_tokens.append(verb)
            for child in t.children:
                if child.dep_ in {'obj','iobj','obl'} and child.pos_ in {'NOUN','PROPN','PRON'}:
                    obj = _safe_lemma(child)
                    if obj:
                        action_tokens.append(f'{verb}__{obj}'); pair_count += 1
        action_docs.append(' '.join(action_tokens) if action_tokens else 'semacao')
        token_count = max(1, len([t for t in doc if not t.is_space]))
        neg_count = sum(1 for t in doc if t.dep_ == 'neg' or _safe_lemma(t) in {'nao','não','nunca','jamais'})
        punct_count = sum(1 for t in doc if t.is_punct)
        ling_rows.append([
            np.log1p(n_alpha), np.log1p(n_sent), n_alpha/n_sent,
            float(np.mean([len(t.text) for t in alpha])) if alpha else 0.0,
            len(set(lemmas))/max(len(lemmas),1),
            *[pos_counts[p]/n_alpha for p in pos_names],
            neg_count/token_count, text.count('?')/token_count, text.count('!')/token_count,
            punct_count/token_count, pair_count/n_alpha,
        ])

    hasher = HashingVectorizer(
        n_features=ENTITY_HASH_DIM, alternate_sign=False, norm=None,
        tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False,
    )
    h = hasher.transform(entity_docs).toarray().astype(np.float32)
    x_ent = safe_unit_rows(np.concatenate([h, np.asarray(type_props,np.float32), np.asarray(ent_stats,np.float32)], axis=1))
    x_ling = np.asarray(ling_rows, dtype=np.float32)
    if not np.isfinite(x_ling).all(): raise ValueError('Features linguísticas inválidas.')

    np.savez_compressed(
        cache, ids=dev.Id.to_numpy(dtype=str), X_entities=x_ent,
        entity_labels=np.asarray(entity_labels,dtype=str), action_docs=np.asarray(action_docs,dtype=str),
        X_linguistic_raw=x_ling, linguistic_feature_names=np.asarray(ling_names,dtype=str)
    )
    meta = {
        'model': SPACY_MODEL_ID, 'spacy_version': spacy.__version__, 'data_sha256': DATA_SHA,
        'entity_rule': 'entity_surface_hash_plus_type_proportions_and_counts_v1',
        'action_rule': 'verb_lemmas_plus_simple_dependency_objects_v1',
        'linguistic_rule': 'length_lexical_pos_negation_punctuation_v1',
        'entity_dimensions': int(x_ent.shape[1]), 'linguistic_dimensions': int(x_ling.shape[1]),
        'reused_legacy_caches': False,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return x_ent, entity_labels, action_docs, x_ling, ling_names, {**meta, 'cache_hit': False, 'source': 'computed'}


def prepare_fixed_features(dataset):
    dev = dataset.frame
    timing_rows = []
    t0=time.perf_counter(); x_minilm,minilm_meta=get_minilm_features(dev); timing_rows.append({'stage':'minilm','observed_seconds':time.perf_counter()-t0,'cache_hit':minilm_meta.get('cache_hit')})
    t0=time.perf_counter(); x_emotion,emotion_labels,emotion_meta=get_emotion_features(dev); timing_rows.append({'stage':'emotion','observed_seconds':time.perf_counter()-t0,'cache_hit':emotion_meta.get('cache_hit')})
    t0=time.perf_counter(); x_entities,entity_labels,action_docs,x_linguistic_raw,linguistic_names,spacy_meta=get_spacy_text_views(dev); timing_rows.append({'stage':'spacy_shared','observed_seconds':time.perf_counter()-t0,'cache_hit':spacy_meta.get('cache_hit')})
    n=len(dev)
    assert len(x_minilm)==len(x_emotion)==len(x_entities)==len(action_docs)==len(x_linguistic_raw)==n
    return {
        'arrays': {'minilm':x_minilm, 'entities':x_entities, 'emotion':x_emotion},
        'action_docs': action_docs, 'linguistic_raw': x_linguistic_raw,
        'labels': {'emotion':emotion_labels, 'entities':entity_labels, 'linguistic':linguistic_names},
        'meta': {'minilm':minilm_meta, 'emotion':emotion_meta, 'spacy':spacy_meta},
        'timing': pd.DataFrame(timing_rows),
    }


BASE_TFIDF_CFG = dict(
    lowercase=True, strip_accents='unicode', analyzer='word', ngram_range=(1,2),
    token_pattern=r'(?u)\b\w\w+\b', min_df=2, max_df=.95, max_features=20000,
    sublinear_tf=True, smooth_idf=True, norm='l2', dtype=np.float64,
)


WORD_CFGS = {
    'word_1': {**BASE_TFIDF_CFG, 'ngram_range': (1,1)},
    'word_1_2': {**BASE_TFIDF_CFG, 'ngram_range': (1,2)},
    'word_1_3': {**BASE_TFIDF_CFG, 'ngram_range': (1,3)},
}


CHAR_CFG = dict(
    lowercase=True, strip_accents='unicode', analyzer='char_wb', ngram_range=(3,5),
    min_df=2, max_df=.95, max_features=30000, sublinear_tf=True,
    smooth_idf=True, norm='l2', dtype=np.float64,
)


ACTION_CFG = dict(
    lowercase=True, strip_accents='unicode', analyzer='word', ngram_range=(1,2),
    token_pattern=r'(?u)\b\w\w+\b', min_df=2, max_df=1.0, max_features=8000,
    sublinear_tf=True, smooth_idf=True, norm='l2', dtype=np.float64,
)


SOURCE_COMPONENTS = {
    'minilm': (), 'entities': (), 'emotion': (),
    'word_1': ('word_1_base','word_1_svd'),
    'word_1_2': ('word_1_2_base','word_1_2_svd'),
    'word_1_3': ('word_1_3_base','word_1_3_svd'),
    'char_3_5': ('char_3_5_base','char_3_5_svd'),
    'topics': ('word_1_2_base','topics_nmf'),
    'actions': ('actions_base','actions_svd'),
    'linguistic': ('linguistic_scaler',),
}


SOURCE_ORDER = tuple(SOURCE_COMPONENTS)


def _fit_tfidf_base(texts_tr, texts_va, cfg):
    vec = TfidfVectorizer(**cfg)
    t0 = time.perf_counter(); a_tr = vec.fit_transform(texts_tr); fit_s = time.perf_counter()-t0
    if a_tr.shape[1] < 2: raise RuntimeError('Vocabulário TF-IDF insuficiente.')
    t1 = time.perf_counter(); a_va = vec.transform(texts_va); transform_s = time.perf_counter()-t1
    return vec, a_tr, a_va, fit_s, transform_s


def _fit_svd(a_tr, a_va, target_dim, seed=CV_SEED):
    dim = min(target_dim, a_tr.shape[1]-1, a_tr.shape[0]-1)
    svd = TruncatedSVD(n_components=dim, algorithm='randomized', n_iter=7, n_oversamples=10, random_state=seed)
    t0 = time.perf_counter(); xtr = safe_unit_rows(svd.fit_transform(a_tr)); fit_s = time.perf_counter()-t0
    t1 = time.perf_counter(); xva = safe_unit_rows(svd.transform(a_va)); transform_s = time.perf_counter()-t1
    return xtr, xva, svd, fit_s, transform_s


def build_fold_features(dataset, fixed, tr, va, fold_number):
    texts_tr=dataset.frame.iloc[tr].texto.tolist(); texts_va=dataset.frame.iloc[va].texto.tolist()
    arrays=fixed['arrays']
    train={'minilm':arrays['minilm'][tr], 'entities':arrays['entities'][tr], 'emotion':arrays['emotion'][tr]}
    valid={'minilm':arrays['minilm'][va], 'entities':arrays['entities'][va], 'emotion':arrays['emotion'][va]}
    components = {}; meta = {'fold': fold_number}

    # word n-grams; word_1_2 também alimenta NMF para não ajustar o mesmo TF-IDF duas vezes.
    word_bases = {}
    for name, cfg in WORD_CFGS.items():
        _, a_tr, a_va, fit_b, trans_b = _fit_tfidf_base(texts_tr, texts_va, cfg)
        components[f'{name}_base'] = {'fit_seconds': fit_b, 'transform_seconds': trans_b}
        xtr, xva, svd, fit_s, trans_s = _fit_svd(a_tr, a_va, LEX_SVD_DIM)
        components[f'{name}_svd'] = {'fit_seconds': fit_s, 'transform_seconds': trans_s}
        train[name] = xtr; valid[name] = xva
        meta[f'{name}_vocab'] = int(a_tr.shape[1]); meta[f'{name}_dim'] = int(xtr.shape[1])
        meta[f'{name}_evr'] = float(svd.explained_variance_ratio_.sum())
        if name == 'word_1_2': word_bases[name] = (a_tr, a_va)

    # char n-grams
    _, a_tr, a_va, fit_b, trans_b = _fit_tfidf_base(texts_tr, texts_va, CHAR_CFG)
    components['char_3_5_base'] = {'fit_seconds': fit_b, 'transform_seconds': trans_b}
    xtr, xva, svd, fit_s, trans_s = _fit_svd(a_tr, a_va, LEX_SVD_DIM)
    components['char_3_5_svd'] = {'fit_seconds': fit_s, 'transform_seconds': trans_s}
    train['char_3_5'] = xtr; valid['char_3_5'] = xva
    meta['char_3_5_vocab'] = int(a_tr.shape[1]); meta['char_3_5_dim'] = int(xtr.shape[1]); meta['char_3_5_evr'] = float(svd.explained_variance_ratio_.sum())

    # tópicos no mesmo TF-IDF word(1,2)
    a_tr, a_va = word_bases['word_1_2']
    nmf_dim = min(NMF_DIM, a_tr.shape[1], len(tr))
    nmf = NMF(n_components=nmf_dim, init='nndsvda', solver='cd', beta_loss='frobenius', max_iter=500, random_state=CV_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter(); xtr = safe_unit_rows(nmf.fit_transform(a_tr)); fit_s = time.perf_counter()-t0
        t1 = time.perf_counter(); xva = safe_unit_rows(nmf.transform(a_va)); trans_s = time.perf_counter()-t1
    components['topics_nmf'] = {'fit_seconds': fit_s, 'transform_seconds': trans_s}
    train['topics'] = xtr; valid['topics'] = xva; meta['topics_dim'] = int(xtr.shape[1])

    # ações narrativas
    actions_tr=[fixed['action_docs'][i] for i in tr]; actions_va=[fixed['action_docs'][i] for i in va]
    _, a_tr, a_va, fit_b, trans_b = _fit_tfidf_base(actions_tr, actions_va, ACTION_CFG)
    components['actions_base'] = {'fit_seconds': fit_b, 'transform_seconds': trans_b}
    xtr, xva, svd, fit_s, trans_s = _fit_svd(a_tr, a_va, ACTION_SVD_DIM)
    components['actions_svd'] = {'fit_seconds': fit_s, 'transform_seconds': trans_s}
    train['actions'] = xtr; valid['actions'] = xva
    meta['actions_vocab'] = int(a_tr.shape[1]); meta['actions_dim'] = int(xtr.shape[1]); meta['actions_evr'] = float(svd.explained_variance_ratio_.sum())

    # features linguísticas
    scaler = StandardScaler()
    t0 = time.perf_counter(); xtr=scaler.fit_transform(fixed['linguistic_raw'][tr]).astype(np.float32); fit_s = time.perf_counter()-t0
    t1 = time.perf_counter(); xva=scaler.transform(fixed['linguistic_raw'][va]).astype(np.float32); trans_s = time.perf_counter()-t1
    components['linguistic_scaler'] = {'fit_seconds': fit_s, 'transform_seconds': trans_s}
    train['linguistic'] = safe_unit_rows(xtr); valid['linguistic'] = safe_unit_rows(xva)
    meta['linguistic_dim'] = int(xtr.shape[1])

    for source in SOURCE_ORDER:
        if source not in train or train[source].shape[1] != valid[source].shape[1]:
            raise AssertionError(f'Fonte ausente ou dimensão instável: {source}')
        if not np.isfinite(train[source]).all() or not np.isfinite(valid[source]).all():
            raise ValueError(f'NaN/Inf em {source}')
    return {'train': train, 'valid': valid, 'components': components, 'meta': meta}


def prepare_feature_context(dataset):
    print('Preparando extratores congelados/cacheados...')
    fixed=prepare_fixed_features(dataset)
    fixed['timing'].to_csv(STUDY_DIR/'fixed_extraction_timing.csv',index=False)

    fold_packs=[]; feature_meta=[]; component_rows=[]
    print('Preparando transformações fold-local...')
    for fold,(tr,va) in enumerate(dataset.folds,1):
        pack=build_fold_features(dataset,fixed,tr,va,fold)
        fold_packs.append(pack); feature_meta.append(pack['meta'])
        for component,timing in pack['components'].items():
            component_rows.append({'fold':fold,'component':component,**timing})
        print(f'  fold {fold}/{len(dataset.folds)} pronto')

    feature_meta_df=pd.DataFrame(feature_meta)
    component_timing=pd.DataFrame(component_rows)
    feature_meta_df.to_csv(STUDY_DIR/'feature_meta_by_fold.csv',index=False)
    component_timing.to_csv(STUDY_DIR/'feature_component_timing.csv',index=False)

    catalog=[]
    for source in SOURCE_ORDER:
        dims=sorted({p['train'][source].shape[1] for p in fold_packs})
        catalog.append({
            'source':source, 'dim':dims[0] if len(dims)==1 else str(dims),
            'fit_scope':'frozen/cache' if not SOURCE_COMPONENTS[source] else 'outer-train',
            'components':'+'.join(SOURCE_COMPONENTS[source]) or 'precomputed',
        })
    catalog_df=pd.DataFrame(catalog)
    return FeatureContext(
        fold_packs=fold_packs, catalog=catalog_df, fixed_meta=fixed['meta'],
        fixed_extraction_timing=fixed['timing'], fold_meta=feature_meta_df,
        component_timing=component_timing,
    )
