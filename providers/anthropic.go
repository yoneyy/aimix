package providers

import (
	"github.com/yoneyy/aimix"
	"github.com/yoneyy/aimix/api"
)

// AnthropicConfig is the configuration of Anthropic(): https://api.anthropic.com
// with the key from ANTHROPIC_API_KEY, serving the Messages API.
func AnthropicConfig() Config {
	return Config{
		ID:      aimix.ProviderAnthropic,
		Name:    "Anthropic",
		BaseURL: "https://api.anthropic.com",
		APIKeyEnv: []string{
			"ANTHROPIC_API_KEY",
		},
		Apis: map[aimix.Api]aimix.ApiStreamer{
			aimix.ApiAnthropicMessages: api.AnthropicMessages(),
		},
	}
}

// Anthropic returns the Anthropic provider.
func Anthropic() aimix.Provider {
	return New(AnthropicConfig())
}
