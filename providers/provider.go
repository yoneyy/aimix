// Package providers defines the built-in providers and a way to assemble others
// from the implementations in the api package.
package providers

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/yoneyy/aimix"
)

// Config describes a provider.
//
// An OpenAI-compatible server, for example:
//
//	deepseek := providers.New(providers.Config{
//		ID:        "deepseek",
//		Name:      "DeepSeek",
//		BaseURL:   "https://api.deepseek.com",
//		APIKeyEnv: []string{"DEEPSEEK_API_KEY"},
//		Apis: map[aimix.Api]aimix.ApiStreamer{
//			aimix.ApiOpenAICompletions: api.OpenAICompletions(),
//		},
//	})
type Config struct {
	ID   aimix.ProviderID
	Name string
	// BaseURL is used for models without Model.BaseURL.
	BaseURL string
	// APIKey is used for requests without StreamOptions.APIKey.
	APIKey string
	// APIKeyEnv lists environment variables checked in order when neither the
	// request nor APIKey sets a key.
	APIKeyEnv []string
	// Apis are the implementations by Model.Api.
	Apis map[aimix.Api]aimix.ApiStreamer
}

type provider struct {
	cfg Config
}

// New returns a provider that fills in the API key and base URL and dispatches
// by Model.Api.
func New(cfg Config) aimix.Provider {
	return &provider{cfg: cfg}
}

func (p *provider) ID() aimix.ProviderID {
	return p.cfg.ID
}

func (p *provider) Name() string {
	return p.cfg.Name
}

func (p *provider) Stream(ctx context.Context, req aimix.StreamRequest) (*aimix.AssistantMessageEventStream, error) {
	impl, err := p.prepare(&req.Model, &req.Options)
	if err != nil {
		return nil, err
	}
	return impl.Stream(ctx, req)
}

func (p *provider) StreamSimple(ctx context.Context, req aimix.SimpleStreamRequest) (*aimix.AssistantMessageEventStream, error) {
	impl, err := p.prepare(&req.Model, &req.Options.StreamOptions)
	if err != nil {
		return nil, err
	}
	return impl.StreamSimple(ctx, req)
}

func (p *provider) prepare(model *aimix.Model, opts *aimix.StreamOptions) (aimix.ApiStreamer, error) {
	impl, ok := p.cfg.Apis[model.Api]
	if !ok {
		return nil, fmt.Errorf("aimix: provider %s does not support api %q", p.cfg.ID, model.Api)
	}
	if model.BaseURL == "" {
		model.BaseURL = p.cfg.BaseURL
	}
	if opts.APIKey == "" {
		opts.APIKey = p.apiKey()
	}
	if opts.APIKey == "" {
		if len(p.cfg.APIKeyEnv) == 0 {
			return nil, fmt.Errorf("aimix: missing API key for %s", p.cfg.ID)
		}
		return nil, fmt.Errorf("aimix: missing API key for %s (set %s)", p.cfg.ID, strings.Join(p.cfg.APIKeyEnv, " or "))
	}
	return impl, nil
}

func (p *provider) apiKey() string {
	if p.cfg.APIKey != "" {
		return p.cfg.APIKey
	}
	for _, name := range p.cfg.APIKeyEnv {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	return ""
}
