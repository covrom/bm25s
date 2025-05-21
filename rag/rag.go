package rag

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/agnivade/levenshtein"
	"github.com/covrom/bm25s"
	"github.com/sashabaranov/go-openai"
)

// ResponseEntry represents an entry in the pre-prepared response database
type ResponseEntry struct {
	ID       string `json:"id"`        // Unique identifier for the entry
	Content  string `json:"content"`   // The actual text content of the entry
	IsPrompt bool   `json:"is_prompt"` // Flag indicating if this entry is a system prompt
}

// DocMatch implements a document match helper using BM25 algorithm
type DocMatch struct {
	documents []ResponseEntry // Collection of response entries
	bm        *bm25s.BM25S    // BM25S instance for document search
	ftr       float64         // Fuzzy matching threshold (0.0 to 1.0)
}

// RAGLLM handles the processing of user queries with BM25 and LLM integration
type RAGLLM struct {
	dm           *DocMatch      // Document matcher instance
	openaiClient *openai.Client // OpenAI API client
}

// NewRAGLLM initializes the processor with response database and OpenAI client
// Parameters:
//   - responseDB: collection of pre-prepared responses
//   - apiBaseURL: base URL for OpenAI API
//   - openaiAPIKey: authentication key for OpenAI API
func NewRAGLLM(responseDB []ResponseEntry, apiBaseURL, openaiAPIKey string) (*RAGLLM, error) {
	bm25 := &DocMatch{
		documents: responseDB,
		ftr:       0.75, // Default fuzzy matching threshold (75% similarity)
	}

	bm25.preprocessDocuments()

	config := openai.DefaultConfig(openaiAPIKey)
	config.BaseURL = apiBaseURL // Can be set to "https://api.openai.com/v1" or custom endpoint

	client := openai.NewClientWithConfig(config)
	return &RAGLLM{
		dm:           bm25,
		openaiClient: client,
	}, nil
}

// preprocessDocuments tokenizes and stems documents for BM25 search
func (dm *DocMatch) preprocessDocuments() {
	docs := make([]string, len(dm.documents))
	for i, doc := range dm.documents {
		docs[i] = doc.Content
	}
	dm.bm = bm25s.New(docs) // Initialize BM25 with document contents
}

// findFuzzyMatches performs fuzzy search for the query against document contents.
// It returns a slice of ResponseEntry that are above the similarity threshold.
func (dm *DocMatch) findFuzzyMatches(query string) []ResponseEntry {
	threshold := dm.ftr
	var matches []ResponseEntry

	for _, entry := range dm.documents {
		// Calculate Levenshtein distance between query and document content
		dist := levenshtein.ComputeDistance(strings.ToLower(query), strings.ToLower(entry.Content))

		// Calculate similarity score (higher is better)
		// Formula: 1.0 - (distance / max_length)
		maxLength := math.Max(float64(len(query)), float64(len(entry.Content)))
		if maxLength == 0 { // Avoid division by zero for empty strings
			if len(query) == 0 && len(entry.Content) == 0 {
				if threshold <= 1.0 { // Empty strings are considered 100% similar
					matches = append(matches, entry)
				}
			}
			continue
		}

		similarity := 1.0 - (float64(dist) / maxLength)

		if similarity >= threshold {
			matches = append(matches, entry)
		}
	}
	return matches
}

// readFileContent reads a file and ensures it's valid UTF-8 encoded
func readFileContent(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %w", filePath, err)
	}

	if !utf8.Valid(data) {
		return "", fmt.Errorf("file %s contains invalid UTF-8", filePath)
	}

	return string(data), nil
}

// ProcessQuery handles the complete query processing pipeline with file attachments
// Parameters:
//   - ctx: context for API calls
//   - model: OpenAI model name to use
//   - sysprompt: optional system prompt
//   - query: user's query
//   - filePaths: paths to attached files
//   - useFuzzy: whether to enable fuzzy matching
//
// Returns:
//   - generated response
//   - error if any occurred
func (p *RAGLLM) ProcessQuery(ctx context.Context, model, sysprompt, query string, filePaths []string, useFuzzy bool) (string, error) {
	// --- BM25 Retrieval ---
	scores := p.dm.bm.Search(query, 1) // Search with top 1 result
	bestBM25Idx := -1                  // Using -1 to indicate no match found
	bestBM25Score := 0.0
	for _, score := range scores {
		if score.Score > bestBM25Score {
			bestBM25Score = score.Score
			bestBM25Idx = score.DocIndex
		}
	}

	// --- Fuzzy Search Retrieval ---
	var fuzzyMatches []ResponseEntry
	if useFuzzy {
		// Find documents with similarity above threshold (0.75 by default)
		fuzzyMatches = p.dm.findFuzzyMatches(query)
	}

	// Prepare messages for OpenAI API
	var messages []openai.ChatCompletionMessage

	// Add system prompt if provided
	if len(sysprompt) > 0 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: sysprompt,
		})
	}

	sb := &strings.Builder{}

	// Flag to determine if "Using the following information:" prefix is needed
	needsInfoPrefix := len(filePaths) > 0 || bestBM25Score > 0 || len(fuzzyMatches) > 0

	if needsInfoPrefix {
		fmt.Fprintln(sb, "Using the following information:")
	}

	// Add BM25 best match content
	if bestBM25Score > 0 && bestBM25Idx != -1 {
		entry := p.dm.documents[bestBM25Idx]
		if entry.IsPrompt {
			fmt.Fprintln(sb, entry.Content)
		} else {
			// If it's not a prompt but a relevant response, return it directly
			return entry.Content, nil
		}
	}

	// Add Fuzzy matches content
	if len(fuzzyMatches) > 0 {
		fuzzyContent := "Fuzzy matches:\n"
		for i, match := range fuzzyMatches {
			// Avoid duplicates if BM25 already found this document
			if bestBM25Idx != -1 && p.dm.documents[bestBM25Idx].ID == match.ID {
				continue
			}
			fuzzyContent += fmt.Sprintf("  - %s\n", match.Content)
			if i >= 4 { // Limit number of fuzzy matches to avoid context overload
				break
			}
		}
		if fuzzyContent != "Fuzzy matches:\n" { // Add only if there are actual matches
			fmt.Fprintln(sb, fuzzyContent)
		}
	}

	// Add file contents
	for _, filePath := range filePaths {
		content, err := readFileContent(filePath)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(sb, "Content of file %s:\n%s\n", filepath.Base(filePath), content)
	}

	// Detect language based on character counts
	var cyrCount, latCount int
	for _, r := range query {
		switch {
		case r >= 'а' && r <= 'я' || r >= 'А' && r <= 'Я':
			cyrCount++
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z':
			latCount++
		}
	}

	lang := "english"
	if cyrCount > latCount {
		lang = "russian"
	}

	// Add the main query prompt
	fmt.Fprintf(sb, "Answer the question in %s language:\n%s\n", lang, query)

	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: sb.String(),
	})

	// Call OpenAI API
	resp, err := p.openaiClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
	})
	if err != nil {
		return "", fmt.Errorf("failed to call OpenAI API: %w", err)
	}

	return resp.Choices[0].Message.Content, nil
}
