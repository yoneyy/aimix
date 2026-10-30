package aimix

import "slices"

// Modality is a kind of input a model accepts.
type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
)

// Model describes one model on one provider. aimix ships no model catalog: callers
// build Model values themselves.
type Model struct {
	// ID is the model id sent to the API (e.g. "claude-sonnet-4-5", "gpt-5").
	ID string `json:"id"`
	// Name is a human-readable name.
	Name string `json:"name,omitempty"`
	// Api selects the wire protocol implementation.
	Api Api `json:"api"`
	// Provider is the provider name (e.g. "anthropic", "openai").
	Provider ProviderID `json:"provider"`
	// BaseURL overrides the provider's default base URL.
	BaseURL string `json:"base_url,omitempty"`
	// Reasoning Reports whether the model exposes a reasoning/thinking channel.
	Reasoning bool `json:"reasoning"`
	// Input lists the accepted input modalities. Nil means no restriction.
	Input []Modality `json:"input,omitempty"`
	// ContextWindow is the maximum input+output token window, 0 if unknown.
	ContextWindow int `json:"context_window,omitempty"`
	// MaxTokens is the max tokens the model may emit per response, 0 if unknown.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Headers are sent with every request for this model.
	Headers map[string]string `json:"headers,omitempty"`
	// ThinkingLevels maps unified thinking levels to this model's wire values.
	ThinkingLevels ThinkingLevelMap `json:"thinking_levels,omitempty"`
	// Compat switches protocol details this model needs.
	Compat ModelCompat `json:"compat,omitzero"`
}

// SupportsImages reports whether the model accepts image input.
func (m Model) SupportsImages() bool {
	if m.Input == nil {
		return true
	}
	return slices.Contains(m.Input, ModalityImage)
}

// ModelCompat holds protocol switches for models that deviate from the API defaults.
type ModelCompat struct {
	// AdaptiveThinking makes Anthropic requests use adaptive thinking with
	// output_config.effort instead of a budget_tokens thinking budget.
	AdaptiveThinking bool `json:"adaptive_thinking,omitempty"`
	// NoTemperature drops the temperature parameter for models that reject it.
	NoTemperature bool `json:"no_temperature,omitempty"`
	// LegacyMaxTokens makes Chat Completions requests send max_tokens instead of
	// max_completion_tokens.
	LegacyMaxTokens bool `json:"legacy_max_tokens,omitempty"`
	// NoReasoningEffort drops reasoning_effort from Chat Completions requests.
	NoReasoningEffort bool `json:"no_reasoning_effort,omitempty"`
}

type Context struct {
	SystemPrompt string   `json:"system_prompt,omitempty"`
	Messages     Messages `json:"messages"`
	Tools        []Tool   `json:"tools,omitempty"`
}
