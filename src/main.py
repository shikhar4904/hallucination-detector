import re
import string
import wikipedia
from typing import List, Dict


STOPWORDS = {
    "a","an","the","in","on","at","of","to","for","is","was","were",
    "are","be","been","being","he","she","it","they","his","her","its",
    "their","and","or","but","with","by","from","as","into","that",
    "this","which","who","whom","whose","have","has","had","do","does",
    "did","will","would","also","so","than","then","there",
    "about","after","before","during","over","under","between","through",
    "upon","across","against","along","around","behind","below","beside",
    "beyond","during","except","inside","near","outside","since","toward",
    "throughout","within","without",
}

SUBJECT_PRONOUNS = {"he", "she", "they", "it"}
OBJECT_PRONOUNS = {"him", "her", "them", "it"}
POSSESSIVE_PRONOUNS = {"his", "her", "their", "its"}


def tokenize(text: str) -> List[str]:
    
    text   = text.lower()
    text   = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def make_ngrams(tokens: List[str], n: int) -> List[str]:
    
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]



DATE_PATTERN = re.compile(
    r'\b('
    r'\d{4}'                          # year: 1876
    r'|(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'(?:\s+\d{1,2})?(?:,\s*\d{4})?'  # March 7, 1876
    r'|\d{1,2}\s+'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'(?:\s+\d{4})?'                   # 7 March 1876
    r')\b',
    re.IGNORECASE,
)

PLACE_PREPS = {"in","at","from","near","outside","inside","across","to","into"}


def find_dates(text: str) -> List[str]:
    
    return [m.group().strip() for m in DATE_PATTERN.finditer(text)]


def find_capitalized_phrases(text: str) -> List[str]:
    
    pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    phrases = []
    for m in pattern.finditer(text):
        phrase = m.group().strip()
        # Skip single common words
        if phrase.lower() in STOPWORDS or phrase.lower() in {
            "he","she","it","his","her","its","they","their",
            "i","we","you","my","your","our",
        }:
            continue
        phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def find_entities(text: str) -> Dict[str, List[str]]:
    
    dates    = find_dates(text)
    phrases  = find_capitalized_phrases(text)

    places, names = [], []
    for phrase in phrases:
        start     = text.find(phrase)
        pre_text  = text[:start].strip().split()
        prev_word = pre_text[-1].lower().strip(string.punctuation) if pre_text else ""

        if prev_word in PLACE_PREPS:
            places.append(phrase)
        else:
            names.append(phrase)

    return {"dates": dates, "names": names, "places": places}



def find_main_entity(text: str) -> str:
    
    pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    matches = list(pattern.finditer(text))

    if not matches:
        return text[:40]

    scored = []
    for m in matches:
        phrase   = m.group().strip()
        position = m.start()
        length   = len(phrase.split())

        pos_score = 1.0 / (position + 1)

        len_score = length * 0.8

        scored.append((phrase, pos_score + len_score))

    if not scored:
        return text[:40]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def claim_with_entity_pronouns(claim: str, main_entity: str) -> List[str]:
    
    variants = [claim]
    if not main_entity:
        return variants

    surname = main_entity.split()[-1] if len(main_entity.split()) > 1 else main_entity
    possessive = f"{main_entity}'" if main_entity.endswith("s") else f"{main_entity}'s"

    replace_map = {
        **{p: main_entity for p in SUBJECT_PRONOUNS},
        **{p: main_entity for p in OBJECT_PRONOUNS},
        **{p: possessive for p in POSSESSIVE_PRONOUNS},
    }

    for pronoun, replacement in replace_map.items():
        pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
        if pattern.search(claim):
            variants.append(pattern.sub(replacement, claim))
            variants.append(pattern.sub(surname, claim))

    return list(dict.fromkeys(variants))



def fetch_wikipedia(entity: str) -> str:
    
    attempts = [entity]

    words = entity.split()
    if len(words) > 2:
        attempts.append(" ".join(words[:2]))

    if len(words) > 1:
        attempts.append(words[-1])

    for query in attempts:
        try:
            page = wikipedia.page(query, auto_suggest=True)
            return page.content
        except wikipedia.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return page.content
            except Exception:
                continue
        except Exception:
            continue

    return ""


def split_sentences(text: str) -> List[str]:
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]



def score_sentence_tf(sentence: str, claim_tokens: List[str]) -> float:
    
    sent_tokens = set(tokenize(sentence))
    return sum(1 for t in claim_tokens if t in sent_tokens)


def get_relevant_sentences(
    wiki_text: str,
    claim: str,
    main_entity: str = "",
    top_k: int = 8,
) -> List[str]:
    
    claim_variants = claim_with_entity_pronouns(claim, main_entity)
    claim_token_sets = [tokenize(c) for c in claim_variants]
    sentences    = split_sentences(wiki_text)

    scored = []
    for s in sentences:
        best = max(score_sentence_tf(s, tokens) for tokens in claim_token_sets)
        scored.append((s, best))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [s for s, sc in scored[:top_k] if sc > 0]



def normalize(text: str) -> str:
    """Lowercase + remove punctuation + collapse spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def is_found_in_evidence(detail: str, evidence: List[str]) -> bool:
    
    detail_norm   = normalize(detail)
    detail_tokens = set(detail_norm.split())

    for sent in evidence:
        sent_norm   = normalize(sent)
        sent_tokens = set(sent_norm.split())

        if detail_norm in sent_norm:
            return True
        if detail_tokens and detail_tokens.issubset(sent_tokens):
            return True

    return False


def check_details(claim: str, main_entity: str, evidence: List[str]) -> List[Dict]:
    
    claim_variants = claim_with_entity_pronouns(claim, main_entity)
    entities = {"dates": [], "names": [], "places": []}
    for variant in claim_variants:
        detected = find_entities(variant)
        entities["dates"].extend(detected["dates"])
        entities["names"].extend(detected["names"])
        entities["places"].extend(detected["places"])

    entities = {k: list(dict.fromkeys(v)) for k, v in entities.items()}
    main_norm    = normalize(main_entity)
    results      = []

    all_details = (
        [("date",  d) for d in entities["dates"]]  +
        [("place", p) for p in entities["places"]] +
        [("name",  n) for n in entities["names"]]
    )

    for dtype, detail in all_details:
        detail_norm = normalize(detail)

        # Skip if this detail IS the main entity or overlaps with it
        if (detail_norm == main_norm or
                detail_norm in main_norm or
                main_norm in detail_norm):
            continue

        found = is_found_in_evidence(detail, evidence)
        results.append({"detail": detail, "type": dtype, "found": found})

    return results



def detect_hallucination(claim: str, verbose: bool = True) -> Dict:
    
    if verbose:
        print(f"\nClaim : {claim}")
        print("-" * 55)

    # Step 1
    main_entity = find_main_entity(claim)
    if verbose:
        print(f"Subject       : {main_entity}")

    # Step 2
    wiki_text = fetch_wikipedia(main_entity)
    if not wiki_text:
        if verbose:
            print("Wikipedia     : not found")
        return {
            "claim": claim, "main_entity": main_entity,
            "evidence": [], "details": [],
            "hallucinated": [], "verdict": "no_evidence",
        }
    if verbose:
        print(f"Wikipedia     : fetched ({len(wiki_text.split())} words)")

    # Step 3
    evidence = get_relevant_sentences(wiki_text, claim, main_entity=main_entity, top_k=8)
    if verbose:
        print(f"Evidence      : {len(evidence)} relevant sentences")

    # Steps 4 & 5
    details      = check_details(claim, main_entity, evidence)
    hallucinated = [d for d in details if not d["found"]]

    verdict = (
        "no_evidence"   if not evidence else
        "hallucinated"  if hallucinated else
        "supported"
    )

    if verbose:
        if details:
            print("Checking details:")
            for d in details:
                mark = "✓" if d["found"] else "✗ HALLUCINATED"
                print(f"  [{d['type']:5}] {d['detail']:25} {mark}")
        else:
            print("No checkable details found in claim.")
        print(f"Verdict       : {verdict.upper()}")

    return {
        "claim":        claim,
        "main_entity":  main_entity,
        "evidence":     evidence,
        "details":      details,
        "hallucinated": hallucinated,
        "verdict":      verdict,
    }


def format_result(result: Dict) -> str:
    
    claim = result["claim"]
    for h in result["hallucinated"]:
        claim = claim.replace(h["detail"], f"[[{h['detail']}]]")

    icons = {
        "hallucinated": "HALLUCINATED",
        "supported":    "SUPPORTED   ",
        "no_evidence":  "NO EVIDENCE ",
    }
    icon = icons.get(result["verdict"], "UNKNOWN")

    lines = [
        f"[{icon}] {claim}",
        f"             Subject : {result['main_entity']}",
    ]
    if result["hallucinated"]:
        flagged = ", ".join(
            f"'{h['detail']}' ({h['type']})"
            for h in result["hallucinated"]
        )
        lines.append(f"             Flagged : {flagged}")
    if result["evidence"]:
        lines.append(f"             Evidence: {result['evidence'][0][:100]}...")
    return "\n".join(lines)


if __name__ == "__main__":
    tests = [
        "Albert Einstein was born in 2015.",                         # wrong date
        "Albert Einstein was born in Germany.",                      # correct
        "Albert Einstein won the Nobel Prize in Physics in 1925.",   # wrong year (should be 1921)
        "Alexander Graham Bell invented the telephone in France.",   # wrong place
        "The Eiffel Tower is located in Paris, France.",             # correct
        "William Shakespeare was born in London in 1564.",           # wrong place (Stratford)
    ]


    for claim in tests:
        result = detect_hallucination(claim, verbose=True)
        print()
        print(format_result(result))
