package bm25s

import (
	"math"
	"reflect"
	"testing"
)

// TestNew tests the initialization of the BM25S struct
func TestNew(t *testing.T) {
	// Test case 1: English documents with default parameters
	englishDocs := []string{
		"The quick brown fox jumps over the lazy dog",
		"A fox fled from danger",
	}
	bm25En := New(englishDocs, "en")
	if bm25En.language != "en" {
		t.Errorf("Expected language 'en', got '%s'", bm25En.language)
	}
	if bm25En.k1 != ShortK1 {
		t.Errorf("Expected k1 %f, got %f", ShortK1, bm25En.k1)
	}
	if bm25En.b != ShortB {
		t.Errorf("Expected b %f, got %f", ShortB, bm25En.b)
	}

	// Test case 2: Russian documents with custom parameters
	russianDocs := []string{
		"Быстрая лисица перепрыгнула через ленивую собаку",
		"Лисица убегала от опасности",
	}
	bm25Ru := New(russianDocs, "ru", WithK1(1.5), WithB(0.5), WithIWF())
	if bm25Ru.language != "ru" {
		t.Errorf("Expected language 'ru', got '%s'", bm25Ru.language)
	}
	if bm25Ru.k1 != 1.5 {
		t.Errorf("Expected k1 1.5, got %f", bm25Ru.k1)
	}
	if bm25Ru.b != 0.5 {
		t.Errorf("Expected b 0.5, got %f", bm25Ru.b)
	}
	if !bm25Ru.useIWF {
		t.Error("Expected useIWF to be true")
	}
}

// TestBuildIndex tests the index building process
func TestBuildIndex(t *testing.T) {
	// Test case 1: English documents
	englishDocs := []string{
		"The quick brown fox jumps",
		"The fox fled from danger",
		"", // Empty document
	}
	bm25 := New(englishDocs, "en")

	// Check document lengths
	expectedLengths := []int{5, 5, 0}
	if !reflect.DeepEqual(bm25.docLengths, expectedLengths) {
		t.Errorf("Expected docLengths %v, got %v", expectedLengths, bm25.docLengths)
	}

	// Check average document length
	expectedAvg := (5.0 + 5.0 + 0.0) / 3.0
	if math.Abs(bm25.avgDocLength-expectedAvg) > 1e-9 {
		t.Errorf("Expected avgDocLength %f, got %f", expectedAvg, bm25.avgDocLength)
	}

	// Check term document frequencies
	if df, ok := bm25.termDocFreq["fox"]; !ok || df != 2 {
		t.Errorf("Expected document frequency for 'fox' to be 2, got %d", df)
	}

	// Test case 2: Russian documents
	russianDocs := []string{
		"Быстрая лисица перепрыгнула",
		"Лисица убегала от опасности",
	}
	bm25Ru := New(russianDocs, "ru")

	// Check term document frequencies after stemming
	if df, ok := bm25Ru.termDocFreq["лисиц"]; !ok || df != 2 {
		t.Errorf("Expected document frequency for 'лисиц' to be 2, got %d", df)
	}
}

// TestScore tests the scoring function
func TestScore(t *testing.T) {
	// Test case 1: English documents
	englishDocs := []string{
		"The quick brown fox jumps over the lazy dog",
		"A fox fled from danger",
	}
	bm25 := New(englishDocs, "en")

	// Query for "fox"
	scoreDoc0 := bm25.Score(0, "fox")
	scoreDoc1 := bm25.Score(1, "fox")
	if scoreDoc0 <= 0 || scoreDoc1 <= 0 {
		t.Errorf("Expected positive scores for 'fox', got %f and %f", scoreDoc0, scoreDoc1)
	}
	if scoreDoc0 > scoreDoc1 {
		t.Errorf("Expected lower score for doc0 (%f) than doc1 (%f) for 'fox'", scoreDoc0, scoreDoc1)
	}

	// Test case 2: Russian documents
	russianDocs := []string{
		"Быстрая лисица перепрыгнула через собаку",
		"Лисица убегала от опасности",
	}
	bm25Ru := New(russianDocs, "ru")

	// Query for "лисица"
	scoreDoc0Ru := bm25Ru.Score(0, "лисица")
	scoreDoc1Ru := bm25Ru.Score(1, "лисица")
	if scoreDoc0Ru <= 0 || scoreDoc1Ru <= 0 {
		t.Errorf("Expected positive scores for 'лисица', got %f and %f", scoreDoc0Ru, scoreDoc1Ru)
	}
}

// TestSearch tests the search functionality
func TestSearch(t *testing.T) {
	// Test case 1: English documents
	englishDocs := []string{
		"The quick brown fox jumps over the lazy dog",
		"A fox fled from danger",
		"Irrelevant document",
	}
	bm25 := New(englishDocs, "en")

	// Search for "fox" with topN=2
	results := bm25.Search("fox", 2)
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
	if results[0].DocIndex != 1 || results[1].DocIndex != 0 {
		t.Errorf("Expected doc indices [1, 0], got [%d, %d]", results[0].DocIndex, results[1].DocIndex)
	}
	if results[0].Score < results[1].Score {
		t.Errorf("Expected higher score for doc0 (%f) than doc1 (%f)", results[0].Score, results[1].Score)
	}

	// Test case 2: Russian documents
	russianDocs := []string{
		"Быстрая лисица перепрыгнула через собаку",
		"Лисица убегала от опасности",
		"Нерелевантный документ",
	}
	bm25Ru := New(russianDocs, "ru", WithIWF())

	// Search for "лисица" with topN=2
	resultsRu := bm25Ru.Search("лисица", 2)
	if len(resultsRu) != 2 {
		t.Errorf("Expected 2 results, got %d", len(resultsRu))
	}
	if resultsRu[0].DocIndex != 1 || resultsRu[1].DocIndex != 0 {
		t.Errorf("Expected doc indices [1, 0], got [%d, %d]", resultsRu[0].DocIndex, resultsRu[1].DocIndex)
	}

	// Test case 3: Empty query
	resultsEmpty := bm25.Search("", 2)
	if len(resultsEmpty) != 0 {
		t.Errorf("Expected 0 results for empty query, got %d", len(resultsEmpty))
	}
}

// TestEdgeCases tests edge cases like empty documents or queries
func TestEdgeCases(t *testing.T) {
	// Test case 1: Empty document collection
	bm25Empty := New([]string{}, "en")
	if bm25Empty.avgDocLength != 0 {
		t.Errorf("Expected avgDocLength 0 for empty collection, got %f", bm25Empty.avgDocLength)
	}
	results := bm25Empty.Search("test", 1)
	if len(results) != 0 {
		t.Errorf("Expected 0 results for empty collection, got %d", len(results))
	}

	// Test case 2: Collection with only empty documents
	bm25EmptyDocs := New([]string{"", ""}, "en")
	if bm25EmptyDocs.avgDocLength != 0 {
		t.Errorf("Expected avgDocLength 0 for empty documents, got %f", bm25EmptyDocs.avgDocLength)
	}
	if len(bm25EmptyDocs.docLengths) != 2 || bm25EmptyDocs.docLengths[0] != 0 || bm25EmptyDocs.docLengths[1] != 0 {
		t.Errorf("Expected docLengths [0, 0], got %v", bm25EmptyDocs.docLengths)
	}

	// Test case 3: Query with no valid terms
	b := New([]string{"The quick fox"}, "en")
	results = b.Search(".,!?", 1)
	if len(results) != 0 {
		t.Errorf("Expected 0 results for query with no valid terms, got %d", len(results))
	}
}
