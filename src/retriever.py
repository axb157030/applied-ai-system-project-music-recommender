"""
Core Retriever class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1 + Phase 3 chunking)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)

A guardrail refuses to answer when nothing relevant is found.

HOW RAG WORKS IN THIS FILE
--------------------------
RAG = Retrieval-Augmented Generation. Instead of asking Gemini a question
cold, we first FIND the relevant paragraphs of our docs, then ask Gemini to
answer using ONLY those paragraphs.

  1. LOAD       load_documents()  -> read every .md/.txt file in docs/
  2. CHUNK      _build_chunks()   -> split each file into paragraphs and put
                                     them all in ONE list (self.chunks)
  3. INDEX      build_index()     -> build a word -> chunk ids lookup table
  4. RETRIEVE   retrieve()        -> clean the question, look each word up in
                                     the index, count matches per chunk, keep
                                     the best top_k chunks
  5. GUARDRAIL  retrieve()        -> if the best chunk matches too few words,
                                     return [] so we answer "I do not know"
  6. GENERATE   llm_client.answer_from_snippets()
                                  -> send question + retrieved chunks to
                                     Gemini, which answers from them only

Steps 1-3 run once at startup. Steps 4-6 run for every question.
Retrieval-only mode stops after step 5 and prints the raw chunks.
RAG mode continues to step 6 so Gemini can write a readable answer.
"""

# Thank you https://github.com/codepath/ai110-module4tinker-docubot-solution/blob/main/docubot.py

import os
import glob
import re

# Very common words that carry little retrieval signal. Dropping them from the
# query keeps a question like "Where is the auth token generated?" focused on
# the words that matter ("auth", "token", "generated").
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "do", "does", "did", "how", "what", "which", "where",
    "when", "who", "why", "i", "you", "it", "this", "that", "these", "those",
    "there", "any", "mention", "me", "my", "can", "with", "from", "into", "as",
    # Words about the corpus itself, not its content — they add noise because
    # "docs"/"documentation" literally appear throughout the documentation.
    "docs", "doc", "documentation",
    # Question filler: these words ask for an answer but say nothing about the
    # topic, so "what does valence mean" should be searched as just "valence".
    "mean", "means", "meaning", "explain", "tell", "give", "show", "about",
    "please", "define", "definition", "exist", "some", "something", "want",
    "good",
}


def _tokenize(text):
    """Lowercase and split into alphanumeric word tokens (punctuation removed)."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _stem(word):
    """Very light plural handling so "genres" matches "genre" and "works" matches "work".
    Only strips one trailing "s" from words longer than 3 letters (not "ss")."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class Retriever:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # STEP 1 - LOAD: list of (filename, full_text), one entry per file.
        self.documents = self.load_documents()

        # STEP 2 - CHUNK: one flat list of (filename, paragraph) holding the
        # paragraphs from ALL files, one after another. Order matters because
        # a chunk's "id" is simply its position in this list.
        self.chunks = self._build_chunks(self.documents)

        # STEP 3 - INDEX: must be built AFTER the chunks, because the index
        # stores chunk ids (positions in self.chunks).
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))

        if not docs:
            # Without this, a wrong folder path silently makes every
            # question answer "I do not know".
            print(f"Warning: no .md/.txt files found in '{self.docs_folder}'")
        return docs

    # -----------------------------------------------------------
    # Chunking (Phase 3)
    # -----------------------------------------------------------

    def _build_chunks(self, documents):
        """
        Split each document into paragraph-sized chunks (separated by blank
        lines). Returns a list of (filename, chunk_text). The filename is kept
        so retrieval results still point back to their source file.

        Example result (ids shown on the left are list positions):
            0: ("how_it_works.md", "Each song gets a score based on its mood.")
            1: ("attributes.md",   "mood: a label like happy or chill.")
            2: ("attributes.md",   "energy: a number from 0 to 1.")
        Chunks from different files sit in the SAME list. One file therefore
        owns many chunk ids (here, attributes.md owns 1 and 2).
        """
        chunks = []
        for filename, text in documents:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
            for paragraph in paragraphs:
                # Skip a lone "# Title" line: it holds no answer and only adds noise.
                is_title_only = paragraph.startswith("# ") and "\n" not in paragraph
                if paragraph and not is_title_only:
                    chunks.append((filename, paragraph))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build an inverted index: word -> set of chunk ids containing it.

            e.g. {"energy": {4, 17, 22}, "mood": {3, 4, 9}}

        WHAT A CHUNK ID IS
            A chunk id is a chunk's position in self.chunks. It is NOT a file
            and NOT a page of a file. "mood": {3, 4, 9} means the word "mood"
            appears in the paragraphs at positions 3, 4 and 9 of self.chunks.
            Those paragraphs can come from different files, and one file can
            own several of the ids.

        WHY "INVERTED"
            Normally you go chunk -> words ("what words are in chunk 4?").
            This flips it to word -> chunks ("which chunks contain 'mood'?"),
            like the index at the back of a textbook that lists page numbers
            for each term. Flipping it is what lets us answer a question
            quickly: look up each question word instead of reading every chunk.

        WHAT THE INDEX DOES NOT STORE
            Only WHICH chunks contain a word. Not how many times the word
            appears, and not where inside the chunk. That is enough, because
            we return the whole chunk.

        To get the actual text back from an id, use self.chunks:
            self.chunks[4]  ->  ("attributes.md", "mood: a label like ...")
        """
        index = {}
        for chunk_id, (_filename, text) in enumerate(chunks):
            # set() so a word repeated inside one chunk is only recorded once
            for token in {_stem(t) for t in _tokenize(text)}:
                index.setdefault(token, set()).add(chunk_id)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def _query_terms(self, query):
        """
        Meaningful words of the question: lowercased, stopwords removed, and
        de-duplicated (a set), so "mood mood" counts as one word, not two.
        """
        # Stopwords are checked on the raw word, then the word is stemmed.
        return {_stem(w) for w in _tokenize(query) if w not in STOPWORDS}

    def score_document(self, query, text):
        """
        Score one piece of text against the query: how many distinct query
        words appear in it as whole tokens. More overlap -> higher score.

        Note: retrieve() no longer calls this per chunk, because it gets the
        same counts from the index. Kept as a standalone helper.
        """
        # terms has the list of alphabetical words that are not stopwords 
        # from the user query. heading_words is a set of words in the first line 
        # of the chunk if it starts with a heading. The rank_score function returns 
        # the score for a chunk based on how many distinct query words it contains, 
        # with a bonus if any of those words appear in the heading. 
        # Between two sets, & means intersection: 
        # it returns only the items that are in both sets. 
        # As in the header has to have alphanumerical words that are not stopwords from query 
        # for a higher score.
        text_tokens = {_stem(t) for t in _tokenize(text)}
        return sum(1 for word in self._query_terms(query) if word in text_tokens)

    # Minimum number of distinct query words the best chunk must match for
    # Retriever to answer at all. It is capped by the query's own length (see
    # retrieve) so a one-keyword question like "What is energy?" can still be
    # answered.
    MIN_CONFIDENT_SCORE = 2

    def retrieve(self, query, top_k=3):
        """
        RETRIEVAL STEP: the "R" in RAG.

        Example for the question "How does mood affect the score?"
            terms (stopwords removed): {"mood", "affect", "score"}
            index lookups:
                "mood"   -> {3, 4, 12, 13}
                "affect" -> {13, 22}
                "score"  -> {12, 13, 30}
            matches per chunk:
                chunk 13 -> 3 words  (best)
                chunk 12 -> 2 words
                chunks 3, 4, 22, 30 -> 1 word each
            top_k=3 returns chunk 13, chunk 12, then the lowest-id 1-match
            chunk. Each id is turned back into (filename, text) via
            self.chunks.

        Guardrail: if the best chunk matches fewer than the required number of
        words, return an empty list so the answer modes respond with "I do not
        know based on these docs." instead of serving a weak, off-topic match.
        """
        terms = self._query_terms(query)
        if not terms:
            return []

        # Short questions (1-2 keywords) need only 1 matching word; longer ones need 2.
        # An off-topic question usually matches nothing at all and is refused below.
        required = 1 if len(terms) <= 2 else self.MIN_CONFIDENT_SCORE

        # For each query word, look up the chunk ids that contain it, and
        # tally how many distinct query words each chunk matched.
        counts = {}
        for term in terms:
            for chunk_id in self.index.get(term, ()):
                counts[chunk_id] = counts.get(chunk_id, 0) + 1

        if not counts:
            return []

        # GUARDRAIL: best chunk is too weak -> refuse rather than guess.
        # Checked on the plain match counts, before any heading bonus.
        if max(counts.values()) < required:
            return []

        # A chunk whose "## heading" contains a query word is almost certainly the
        # answer ("## energy" for "what is energy"), so it ranks above chunks that
        # only mention the word in passing.
        def rank_score(chunk_id):
            first_line = self.chunks[chunk_id][1].split("\n", 1)[0]
            heading_words = {_stem(t) for t in _tokenize(first_line)} if first_line.startswith("#") else set()
            return counts[chunk_id] + (1.5 if heading_words & terms else 0)

        # Highest score first; lower chunk id breaks ties so results are stable.
        ranked = sorted(counts, key=lambda chunk_id: (-rank_score(chunk_id), chunk_id))

        # Turn the winning chunk ids back into (filename, text) pairs.
        return [self.chunks[chunk_id] for chunk_id in ranked[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode (no LLM).
        Returns the raw retrieved chunks with their filenames.

        Example output:
            [attributes.md]
            mood: a label like happy or chill.

            ---
            [how_it_works.md]
            Each song gets a score based on its mood and energy.

        Off-topic question output:
            I do not know based on these docs.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode: Retrieve, then Generate.

        1. RETRIEVE: pick the most relevant chunks with self.retrieve().
        2. GENERATE: send the question plus ONLY those chunks to Gemini, which
           writes an answer grounded in them.

        Example output:
            Mood is a label such as happy or chill, and each song's score is
            based on how well its mood and energy match the user's profile.
            (Sources: attributes.md, how_it_works.md)
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        # If retrieval found nothing, do NOT call the LLM. This saves cost and
        # prevents a made-up answer from text that was never in the docs.
        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)

    # Thank you, https://github.com/codepath/ai110-module4tinker-docubot-solution/blob/main/docubot.py