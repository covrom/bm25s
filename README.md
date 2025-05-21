# BM25S - Optimized BM25 Algorithm for Short Texts

## Overview

BM25S is a Go implementation of a modified BM25 algorithm specifically optimized for short text documents. This package provides efficient full-text search capabilities with support for stemming, configurable ranking parameters, and automatic parameter tuning based on document length.

## Features

- **Optimized for short texts** with default parameters tuned for FAQ-style content
- **Automatic parameter adjustment** based on average document length
- **Support for multiple languages** (English and Russian included)
- **Flexible term weighting**:
  - Traditional IDF (Inverse Document Frequency)
  - Alternative IWF (Inverse Word Frequency)
- **Configurable tokenization** with built-in stemming support
- **Efficient indexing** for fast search operations

## Installation

```bash
go get github.com/yourusername/bm25s
```

## Usage

### Basic Usage

```go
package main

import (
	"fmt"
	"github.com/yourusername/bm25s"
)

func main() {
	// Document collection
	docs := []string{
		"How to reset password?",
		"Where to find security settings?",
		"How to change email address?",
		"Why am I receiving spam?",
	}

	// Create search index with default parameters
	bm := bm25s.New(docs, "en")

	// Perform search
	results := bm.Search("reset password", 3)

	// Print results
	for i, res := range results {
		fmt.Printf("%d. [%.2f] %s\n", i+1, res.Score, res.Doc)
	}
}
```

### Advanced Configuration

```go
// Create search index with custom parameters
bm := bm25s.New(docs, "en",
	bm25s.WithK1(1.3),      // Custom term frequency parameter
	bm25s.WithB(0.4),       // Custom length normalization
	bm25s.WithIWF(),        // Use Inverse Word Frequency
	bm25s.WithTokenizer(myCustomTokenizer), // Custom tokenizer
)
```

## API Reference

### Options

- `WithK1(k1 float64)` - Sets the term frequency saturation parameter
- `WithB(b float64)` - Sets the document length normalization parameter
- `WithIWF()` - Enables Inverse Word Frequency instead of IDF
- `WithTokenizer(f func(string) []string)` - Sets a custom tokenizer function

### Methods

- `New(docs []string, language string, opts ...Option)` - Creates a new BM25S instance
- `Score(docIndex int, query string) float64` - Calculates relevance score for a document
- `Search(query string, topN int) []SearchResult` - Performs search and returns top-N results

### SearchResult Structure

```go
type SearchResult struct {
	DocIndex int     // Document index in the collection
	Score    float64 // Relevance score
	Doc      string  // Document text
}
```

## Performance Considerations

- The implementation automatically adjusts parameters based on average document length
- For collections with average document length > 100 terms, it switches to standard BM25 parameters
- Custom tokenizers can significantly impact performance

## Supported Languages

- English (stemming via Snowball)
- Russian (stemming via Snowball)
- Other languages (basic tokenization without stemming)

## License

[MIT License](LICENSE)