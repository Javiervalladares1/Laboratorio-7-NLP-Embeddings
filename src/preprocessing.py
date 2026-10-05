"""Texto -> tokens -> IDs. Se usa la misma función para WikiText y AG News."""
import re,unicodedata,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TOKEN_RE=re.compile(r"[a-z]+|[0-9]+(?:[.,][0-9]+)*")
ARTICLE_RE=re.compile(r'^\s*=\s+[^=]+\s+=\s*$')

def normalize(text):
    text=unicodedata.normalize('NFKC',text).lower()
    text=re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'@[-.,]@', ' ', text)

def tokenize(text):
    """Conservar palabras y números literales; separar puntuación y guiones.
    Sin stemming, stopwords ni lematización. No crear un <num> ausente en GloVe.
    """
    return TOKEN_RE.findall(normalize(text))

def dump_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def prepare(target=22_000_000,min_count=20):
    from datasets import load_from_disk
    data=load_from_disk(str(ROOT/'data/wikitext'))
    counts=Counter(); full=Counter(); lines=[]; subset_articles=0
    total=0; stats={}
    for split,ds in data.items():
        n_tokens=n_articles=n_nonempty=0
        for row in ds:
            text=row['text']; words=tokenize(text)
            n_articles+=bool(ARTICLE_RE.match(text)); n_nonempty+=bool(text.strip())
            n_tokens+=len(words)
            if split=='train':
                full.update(words)
                if total<target and words:
                    take=words[:target-total]
                    lines.append(take); counts.update(take); total+=len(take)
                    subset_articles+=bool(ARTICLE_RE.match(text))
        stats[split]={'lines':len(ds),'nonempty_lines':n_nonempty,'articles':n_articles,'tokens':n_tokens}
    words=sorted((w for w,n in counts.items() if n>=min_count),key=lambda w:(-counts[w],w))
    mapping={w:i for i,w in enumerate(words)}
    ids=[]; offsets=[0]
    with (ROOT/'data/corpus.txt').open('w') as f:
        for line in lines:
            # Borrar OOV antes de subsampling, como gensim: ventanas no cruzan líneas.
            known=[w for w in line if w in mapping]
            if known:
                f.write(' '.join(known)+'\n'); ids.extend(mapping[w] for w in known); offsets.append(len(ids))
    np.save(ROOT/'data/token_ids.npy',np.asarray(ids,dtype=np.int32))
    np.save(ROOT/'data/offsets.npy',np.asarray(offsets,dtype=np.int64))
    np.save(ROOT/'data/counts.npy',np.asarray([counts[w] for w in words],dtype=np.int64))
    dump_json(ROOT/'data/vocab.json',words)
    ranked=np.asarray(sorted(full.values(),reverse=True)); ranks=np.arange(1,len(ranked)+1)
    np.savez_compressed(ROOT/'data/zipf.npz',frequencies=ranked)
    slope=float(np.polyfit(np.log10(ranks[99:10000]),np.log10(ranked[99:10000]),1)[0])
    thresholds=[{'min_count':m,'vocab_size':sum(n>=m for n in counts.values()),'oov_percent':100*sum(n for n in counts.values() if n<m)/total} for m in [1,5,20,50]]
    out={'splits':stats,'subset_tokens':total,'subset_articles_started':subset_articles,'subset_lines':len(lines),'known_tokens':len(ids),'min_count':min_count,'vocab_size':len(words),'oov_percent':100*(total-len(ids))/total,'thresholds':thresholds,'zipf_slope_rank_100_10000':slope,'coverage_full_train':{str(k):float(100*ranked[:k].sum()/ranked.sum()) for k in [10,1000,30000]},'normalization':'NFKC, minúsculas, puntuación y guiones separados, números literales, marcadores y <unk> eliminados, sin stopwords/stemming.'}
    assert len(ids)>=20_000_000, 'Ampliar el subconjunto: deben quedar >=20M tokens conocidos.'
    dump_json(ROOT/'results/preprocessing.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
    examples=["I can't believe it's already January!", "Don't re-enter the state-of-the-art building.", "Dr. Smith lives in the U.S.A.", "The price is $3.50, not $350.", "We're testing king-man+woman=queen.", "They've won 3-0 in 2026.", "John's computer isn't working.", "Email me at hello@example.com.", "The co-founder said: 'Good morning!'", "New York-based firms grew 12.5%.", "Café naïve — Unicode matters.", "WikiText uses @-@ and <unk> markers."]
    from nltk.tokenize import TreebankWordTokenizer
    tok=TreebankWordTokenizer()
    dump_json(ROOT/'results/tokenizer_comparison.json',[{'sentence':s,'own':tokenize(s),'nltk_treebank':tok.tokenize(normalize(s))} for s in examples])
    return out
if __name__=='__main__': prepare()
