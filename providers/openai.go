package providers

import (
	"github.com/yoneyy/aimix"
	"github.com/yoneyy/aimix/api"
)

// OpenAIConfig is the configuration of OpenAI(): https://api.openai.com/v1 with
// the key from OPENAI_API_KEY, serving the Responses and Chat Completions APIs.
func OpenAIConfig() Config {
	return Config{
		ID:      aimix.ProviderOpenAI,
		Name:    "OpenAI",
		BaseURL: "https://api.openai.com/v1",
		APIKeyEnv: []string{
			"OPENAI_API_KEY",
		},
		Apis: map[aimix.Api]aimix.ApiStreamer{
			aimix.ApiOpenAIResponses:   api.OpenAIResponses(),
			aimix.ApiOpenAICompletions: api.OpenAICompletions(),
		},
	}
}

// OpenAI returns the OpenAI provider.
func OpenAI() aimix.Provider {
	return New(OpenAIConfig())
}
