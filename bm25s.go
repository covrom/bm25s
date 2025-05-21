package bm25s

import (
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/kljensen/snowball/english"
	"github.com/kljensen/snowball/russian"
)

const (
	ShortK1 = 1.2
	ShortB  = 0.3
	LongK1  = 1.5
	LongB   = 0.75
)

// BM25S implements a modified BM25 algorithm for short texts
type BM25S struct {
	docs          []string              // Document collection
	avgDocLength  float64               // Average document length (in terms)
	k1            float64               // Term frequency saturation parameter (1.2-2.0)
	b             float64               // Document length normalization parameter (0.3 for short texts)
	tokenizer     func(string) []string // Tokenization and stemming function
	termDocFreq   map[string]int        // DF: number of documents containing the term
	termTotalFreq map[string]int        // Total frequency of the term across all documents
	docTermFreqs  []map[string]int      // TF for each document
	docLengths    []int                 // Document lengths (in terms)
	totalTerms    int                   // Total number of terms across the collection
	autok1        bool
	autob         bool
	useIWF        bool // Use Inverse Word Frequency instead of IDF
}

// Option allows configuring BM25S parameters
type Option func(*BM25S)

// WithK1 sets the term frequency saturation parameter
func WithK1(k1 float64) Option {
	return func(b *BM25S) {
		b.k1 = k1
		b.autok1 = false
	}
}

// WithB sets the document length normalization parameter
func WithB(bParam float64) Option {
	return func(b *BM25S) {
		b.b = bParam
		b.autob = false
	}
}

// WithIWF enables the use of Inverse Word Frequency instead of IDF
func WithIWF() Option {
	return func(b *BM25S) {
		b.useIWF = true
	}
}

// WithTokenizer sets the document tokenizer
func WithTokenizer(f func(string) []string) Option {
	return func(b *BM25S) {
		b.tokenizer = f
	}
}

// New creates and initializes a new BM25S instance
func New(docs []string, opts ...Option) *BM25S {
	b := &BM25S{
		docs:          docs,
		k1:            ShortK1,
		b:             ShortB,
		autok1:        true,
		autob:         true,
		termDocFreq:   make(map[string]int),
		termTotalFreq: make(map[string]int),
	}

	b.tokenizer = b.tokenizeAndStem

	for _, opt := range opts {
		opt(b)
	}

	b.buildIndex()

	// Adjust parameters automatically for long documents
	if b.avgDocLength > 100.0 {
		if b.autok1 {
			b.k1 = LongK1
		}
		if b.autob {
			b.b = LongB
		}
	}

	return b
}

// tokenizeAndStem performs tokenization and stemming of the text
func (b *BM25S) tokenizeAndStem(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	terms := make([]string, 0, len(words))

	for _, word := range words {
		word = strings.TrimFunc(word, func(r rune) bool {
			return strings.ContainsRune(".,!?;:\"'()[]{}", r)
		})

		if utf8.RuneCountInString(word) < 2 {
			continue
		}

		terms = append(terms, b.stemWord(word))
	}

	return terms
}

// stemWord applies language-specific stemming
func (b *BM25S) stemWord(word string) string {
	// Count Cyrillic and Latin characters
	var cyrCount, latCount, digitCount int
	for _, r := range word {
		switch {
		case r >= 'а' && r <= 'я' || r >= 'А' && r <= 'Я':
			cyrCount++
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z':
			latCount++
		case r >= '0' && r <= '9':
			digitCount++
		}
	}

	// Apply stemming based on dominant script
	switch {
	case digitCount > 0:
		return word
	case cyrCount > latCount:
		return russian.Stem(word, false)
	case latCount > cyrCount:
		return english.Stem(word, false)
	}

	// Equal number of both scripts or undetermined - try both
	if stemmed := russian.Stem(word, false); stemmed != "" && stemmed != word {
		return stemmed
	}
	if stemmed := english.Stem(word, false); stemmed != "" && stemmed != word {
		return stemmed
	}

	return word
}

// buildIndex constructs the index for the document collection
func (b *BM25S) buildIndex() {
	b.docTermFreqs = make([]map[string]int, len(b.docs))
	b.docLengths = make([]int, len(b.docs))
	totalLength := 0

	for i, doc := range b.docs {
		terms := b.tokenizer(doc)
		b.docLengths[i] = len(terms)
		totalLength += len(terms)

		tf := make(map[string]int)
		for _, term := range terms {
			tf[term]++
			b.termTotalFreq[term]++
			b.totalTerms++
		}
		b.docTermFreqs[i] = tf

		for term := range tf {
			b.termDocFreq[term]++
		}
	}

	if len(b.docs) > 0 {
		b.avgDocLength = float64(totalLength) / float64(len(b.docs))
	}
}

// safeIDF calculates a stable Inverse Document Frequency
func (b *BM25S) safeIDF(term string) float64 {
	df := b.termDocFreq[term]
	n := len(b.docs)
	return math.Log(float64(n+1)/(float64(df)+0.5)) + 1.0
}

// iwf calculates Inverse Word Frequency
func (b *BM25S) iwf(term string) float64 {
	tf := b.termTotalFreq[term]
	if tf == 0 {
		return 0
	}
	return math.Log(float64(b.totalTerms) / float64(tf))
}

// termWeight returns the weight of the term using IDF or IWF
func (b *BM25S) termWeight(term string) float64 {
	if b.useIWF {
		return b.iwf(term)
	}
	return b.safeIDF(term)
}

// Score calculates the relevance score of a document to the query
// Automatically adjusts calculation for long documents
func (b *BM25S) Score(docIndex int, query string) float64 {
	queryTerms := b.tokenizer(query)
	docTF := b.docTermFreqs[docIndex]
	docLength := float64(b.docLengths[docIndex])
	score := 0.0

	// Determine if this is a long document
	isLongDoc := docLength > 2*b.avgDocLength

	for _, term := range queryTerms {
		if tf, ok := docTF[term]; ok && tf > 0 {
			weight := b.termWeight(term)
			tf := float64(tf)

			numerator := tf * (b.k1 + 1)
			denominator := tf + b.k1*(1-b.b+b.b*(docLength/b.avgDocLength))

			// Different calculation for long documents
			if isLongDoc {
				// Additional penalty for very long documents
				lengthPenalty := math.Min(1.0, b.avgDocLength/docLength)
				score += weight * numerator / denominator * lengthPenalty
			} else {
				// Standard BM25S calculation for short/medium documents
				score += weight * numerator / denominator
			}
		}
	}

	return score
}

// SearchResult holds the result of a search
type SearchResult struct {
	DocIndex int     // Document index in the collection
	Score    float64 // Relevance score
	Doc      string  // Document text
}

// Search performs a search and returns top-N results
func (b *BM25S) Search(query string, topN int) []SearchResult {
	results := make([]SearchResult, 0, len(b.docs))
	for i := range b.docs {
		score := b.Score(i, query)
		if score > 0 {
			results = append(results, SearchResult{
				DocIndex: i,
				Score:    score,
				Doc:      b.docs[i],
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topN > 0 && len(results) > topN {
		results = results[:topN]
	}

	return results
}
