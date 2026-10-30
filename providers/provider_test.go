package providers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yoneyy/aimix"
)

// fakeApi records the request it receives and returns an empty stream.
type fakeApi struct {
	got aimix.StreamRequest
}

func (f *fakeApi) Stream(_ context.Context, req aimix.StreamRequest) (*aimix.AssistantMessageEventStream, error) {
	f.got = req
	return aimix.NewAssistantMessageEventStream(0), nil
}

func (f *fakeApi) StreamSimple(_ context.Context, req aimix.SimpleStreamRequest) (*aimix.AssistantMessageEventStream, error) {
	f.got = aimix.StreamRequest{
		Model:   req.Model,
		Context: req.Context,
		Options: req.Options.StreamOptions,
	}
	return aimix.NewAssistantMessageEventStream(0), nil
}

func TestProviderResolvesKeyAndBaseURL(t *testing.T) {
	fake := &fakeApi{}
	cfg := Config{
		ID:      "acme",
		BaseURL: "https://acme.example/v1",
		APIKeyEnv: []string{
			"ACME_KEY",
			"ACME_FALLBACK_KEY",
		},
		Apis: map[aimix.Api]aimix.ApiStreamer{
			aimix.ApiOpenAICompletions: fake,
		},
	}
	model := aimix.Model{
		ID:  "m",
		Api: aimix.ApiOpenAICompletions,
	}

	tests := []struct {
		name    string
		env     map[string]string
		cfgKey  string
		reqKey  string
		model   aimix.Model
		wantKey string
		wantURL string
		wantErr string
	}{
		{
			name:    "env in order",
			env:     map[string]string{"ACME_KEY": "", "ACME_FALLBACK_KEY": "from-env"},
			model:   model,
			wantKey: "from-env",
			wantURL: "https://acme.example/v1",
		},
		{
			name:    "config key before env",
			env:     map[string]string{"ACME_KEY": "from-env"},
			cfgKey:  "from-config",
			model:   model,
			wantKey: "from-config",
			wantURL: "https://acme.example/v1",
		},
		{
			name:   "request key and model base URL win",
			env:    map[string]string{"ACME_KEY": "from-env"},
			reqKey: "from-request",
			model: aimix.Model{
				ID:      "m",
				Api:     aimix.ApiOpenAICompletions,
				BaseURL: "http://localhost:8000/v1",
			},
			wantKey: "from-request",
			wantURL: "http://localhost:8000/v1",
		},
		{
			name:    "missing key",
			env:     map[string]string{"ACME_KEY": "", "ACME_FALLBACK_KEY": ""},
			model:   model,
			wantErr: "aimix: missing API key for acme (set ACME_KEY or ACME_FALLBACK_KEY)",
		},
		{
			name:   "unsupported api",
			reqKey: "k",
			model: aimix.Model{
				ID:  "m",
				Api: aimix.ApiAnthropicMessages,
			},
			wantErr: `aimix: provider acme does not support api "anthropic-messages"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			c := cfg
			c.APIKey = tt.cfgKey
			p := New(c)
			_, err := p.StreamSimple(context.Background(), aimix.SimpleStreamRequest{
				Model: tt.model,
				Options: aimix.SimpleStreamOptions{
					StreamOptions: aimix.StreamOptions{
						APIKey: tt.reqKey,
					},
				},
			})
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if fake.got.Options.APIKey != tt.wantKey || fake.got.Model.BaseURL != tt.wantURL {
				t.Fatalf("key %q url %q", fake.got.Options.APIKey, fake.got.Model.BaseURL)
			}
		})
	}
}

// TestClientEndToEnd runs a request through Client, the Anthropic provider and
// the anthropic-messages implementation against a local server.
func TestClientEndToEnd(t *testing.T) {
	var gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"pong\"}}\n\n",
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		}, ""))
	}))
	defer srv.Close()
	t.Setenv("ANTHROPIC_API_KEY", "env-key")

	client := aimix.NewClient(OpenAI(), Anthropic())
	model := aimix.Model{
		ID:        "claude-test",
		Api:       aimix.ApiAnthropicMessages,
		Provider:  aimix.ProviderAnthropic,
		BaseURL:   srv.URL,
		MaxTokens: 1024,
	}
	msg, err := client.CompleteSimple(context.Background(), model, aimix.Context{
		Messages: aimix.Messages{
			aimix.UserMessage{
				Role: aimix.RoleUser,
				Content: aimix.Contents{
					aimix.TextContent{
						Type: aimix.ContentTypeText,
						Text: "ping",
					},
				},
			},
		},
	}, aimix.SimpleStreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if gotKey != "env-key" {
		t.Fatalf("api key %q", gotKey)
	}
	if msg.StopReason != aimix.StopReasonStop || msg.Content[0].(aimix.TextContent).Text != "pong" || msg.Usage.Output != 2 {
		t.Fatalf("message %+v", msg)
	}

	_, err = client.Complete(context.Background(), aimix.Model{Provider: "nope"}, aimix.Context{}, aimix.StreamOptions{})
	if err == nil || err.Error() != `aimix: no provider registered for "nope"` {
		t.Fatalf("unknown provider err = %v", err)
	}
}
