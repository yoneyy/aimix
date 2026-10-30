package aimix

import "net/http"

// CacheRetention selects how long providers keep the prompt cache.
type CacheRetention string

const (
	// CacheRetentionNone disables prompt caching.
	CacheRetentionNone CacheRetention = "none"
	// CacheRetentionShort uses the provider's default cache lifetime. It is the default.
	CacheRetentionShort CacheRetention = "short"
	// CacheRetentionLong asks for an extended lifetime (Anthropic 1h, OpenAI 24h).
	CacheRetentionLong CacheRetention = "long"
)

// ToolChoice controls whether the model may or must call tools.
type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// StreamOptions are the request options shared by every Api.
type StreamOptions struct {
	// APIKey authenticates the request. Providers fall back to their environment
	// variables when it is empty.
	APIKey string
	// Headers are added after Model.Headers.
	Headers map[string]string
	// Temperature is omitted when nil, when thinking is enabled on Anthropic, and
	// when the model sets Compat.NoTemperature.
	Temperature *float64
	// MaxTokens caps the response. 0 uses Model.MaxTokens.
	MaxTokens int
	// CacheRetention defaults to CacheRetentionShort.
	CacheRetention CacheRetention
	// SessionID keys the provider prompt cache where supported.
	SessionID string
	// ToolChoice is empty for the provider default (auto).
	ToolChoice ToolChoice
	// ParallelToolCalls allows (true) or forbids (false) several tool calls in
	// one turn. Nil leaves the provider default, which allows them. It is only
	// sent when the request has tools.
	ParallelToolCalls *bool
	// MaxRetries is passed to the SDK. The default 0 leaves retrying to the caller.
	MaxRetries int
	// HTTPClient sends the requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// SimpleStreamOptions adds a provider-neutral thinking level to StreamOptions.
type SimpleStreamOptions struct {
	StreamOptions
	// Reasoning is the requested thinking level. Empty means ThinkingOff. It is
	// clamped to the levels the model supports.
	Reasoning ThinkingLevel
	// ThinkingBudgets overrides the default token budgets of budget-based thinking.
	ThinkingBudgets map[ThinkingLevel]int
}
