package api

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yoneyy/aimix"
)

// TestHTMLResponse covers a base URL that points at a website: the server
// answers 200 with an HTML page instead of an event stream.
func TestHTMLResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, "<!doctype html><html></html>")
	}))
	defer srv.Close()

	opts := aimix.StreamOptions{
		APIKey: "k",
	}
	tests := []struct {
		name   string
		stream func() (*aimix.AssistantMessageEventStream, error)
	}{
		{
			name: "anthropic",
			stream: func() (*aimix.AssistantMessageEventStream, error) {
				return StreamAnthropicMessages(context.Background(), anthropicModel(srv.URL), aimix.TranscriptContext{}, AnthropicOptions{
					StreamOptions: opts,
				})
			},
		},
		{
			name: "responses",
			stream: func() (*aimix.AssistantMessageEventStream, error) {
				return StreamOpenAIResponses(context.Background(), responsesModel(srv.URL), aimix.TranscriptContext{}, OpenAIResponsesOptions{
					StreamOptions: opts,
				})
			},
		},
		{
			name: "completions",
			stream: func() (*aimix.AssistantMessageEventStream, error) {
				return StreamOpenAICompletions(context.Background(), completionsModel(srv.URL), aimix.TranscriptContext{}, OpenAICompletionsOptions{
					StreamOptions: opts,
				})
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := tt.stream()
			if err != nil {
				t.Fatal(err)
			}
			msg, _ := aimix.Collect(context.Background(), s)
			if msg.StopReason != aimix.StopReasonError || !strings.Contains(msg.ErrorReason, "returned an HTML page instead of an event stream") {
				t.Fatalf("stop %q error %q", msg.StopReason, msg.ErrorReason)
			}
		})
	}
}
